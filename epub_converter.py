import json
import os
import re
from typing import Iterable, List


try:
    import ebooklib
    import lxml
    from bs4 import BeautifulSoup, NavigableString
    from ebooklib import epub
except ImportError:  # pragma: no cover - exercised through require_epub_dependencies
    ebooklib = None
    lxml = None
    BeautifulSoup = None
    NavigableString = None
    epub = None


EPUB_DEPENDENCY_MESSAGE = (
    "EPUB support requires ebooklib, beautifulsoup4, and lxml. "
    "Install them with: pip install ebooklib beautifulsoup4 lxml"
)

BLOCK_TAGS = (
    "article",
    "aside",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "div",
    "figcaption",
    "main",
    "pre",
    "section",
    "td",
    "th",
    "caption",
    "dt",
    "dd",
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
MAX_TRANSLATION_UNIT_CHARS = 600
EXCLUDED_TEXT_PARENTS = {
    "head",
    "link",
    "meta",
    "script",
    "style",
    "svg",
    "title",
}


def require_epub_dependencies() -> None:
    if ebooklib is None or lxml is None or BeautifulSoup is None or epub is None:
        raise ImportError(EPUB_DEPENDENCY_MESSAGE)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = max(
            remaining.rfind("; ", 0, max_chars),
            remaining.rfind(", ", 0, max_chars),
            remaining.rfind(" ", 0, max_chars),
        )
        if split_at < max_chars // 2:
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def split_translation_units(text: str) -> List[str]:
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    units = []
    for sentence in sentences:
        units.extend(_split_long_text(sentence, MAX_TRANSLATION_UNIT_CHARS))
    return units


def _item_media_type(item) -> str:
    media_type = getattr(item, "media_type", None)
    if media_type is None and hasattr(item, "get_media_type"):
        media_type = item.get_media_type()
    return (media_type or "").lower()


def _is_text_document_item(item) -> bool:
    if item is None:
        return False
    item_name = (item.get_name() or "").lower()
    media_type = _item_media_type(item)
    return (
        item.get_type() == ebooklib.ITEM_DOCUMENT
        or item_name.endswith((".xhtml", ".html", ".htm"))
        or media_type in {"application/xhtml+xml", "text/html"}
    )


def _document_items_in_reading_order(book) -> Iterable:
    seen = set()
    for spine_item in book.spine:
        item_id = spine_item[0] if isinstance(spine_item, tuple) else spine_item
        item = book.get_item_with_id(item_id)
        if _is_text_document_item(item):
            seen.add(item.get_id())
            yield item
    for item in book.get_items():
        if _is_text_document_item(item) and item.get_id() not in seen:
            yield item


def _iter_translatable_blocks(soup: BeautifulSoup) -> Iterable:
    for block in soup.find_all(BLOCK_TAGS):
        if block.find(BLOCK_TAGS):
            continue
        text = _normalize_text(block.get_text(" ", strip=True))
        if text:
            yield block


def _iter_translatable_text_nodes(soup: BeautifulSoup) -> Iterable:
    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent is None:
            continue
        if parent.name and parent.name.lower() in EXCLUDED_TEXT_PARENTS:
            continue
        if parent.find_parent(EXCLUDED_TEXT_PARENTS):
            continue
        text = _normalize_text(str(node))
        if text:
            yield node


def epub_to_text(epub_path: str, text_path: str, manifest_path: str) -> None:
    """Extract translatable EPUB blocks to one UTF-8 text line per block."""
    require_epub_dependencies()

    book = epub.read_epub(epub_path)
    manifest = {
        "source_epub": os.path.abspath(epub_path),
        "items": [],
    }
    lines: List[str] = []

    for item in _document_items_in_reading_order(book):
        content = item.get_content()
        soup = BeautifulSoup(content, "lxml")
        blocks = []
        candidates = list(_iter_translatable_blocks(soup))
        visible_text = _normalize_text(soup.get_text(" ", strip=True))
        candidate_text_length = sum(
            len(_normalize_text(block.get_text(" ", strip=True)))
            for block in candidates
        )
        extraction_mode = "blocks"
        if visible_text and candidate_text_length < len(visible_text) * 0.8:
            candidates = list(_iter_translatable_text_nodes(soup))
            extraction_mode = "text_nodes"

        for block_index, block in enumerate(candidates):
            if extraction_mode == "text_nodes":
                text = _normalize_text(str(block))
                tag = block.parent.name if block.parent is not None else None
            else:
                text = _normalize_text(block.get_text(" ", strip=True))
                tag = block.name
            text_lines = split_translation_units(text)
            blocks.append(
                {
                    "line": len(lines),
                    "lines": len(text_lines),
                    "block_index": block_index,
                    "tag": tag,
                    "original_text": text,
                }
            )
            lines.extend(text_lines)

        if blocks:
            manifest["items"].append(
                {
                    "item_id": item.get_id(),
                    "file_name": item.get_name(),
                    "extraction_mode": extraction_mode,
                    "blocks": blocks,
                }
            )

    os.makedirs(os.path.abspath(os.path.dirname(text_path)), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.dirname(manifest_path)), exist_ok=True)

    with open(text_path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(lines))
        if lines:
            text_file.write("\n")

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)


def _load_translated_lines(translated_text_path: str) -> List[str]:
    with open(translated_text_path, "r", encoding="utf-8") as translated_file:
        return [line.rstrip("\n") for line in translated_file]


def text_to_epub(
    original_epub_path: str,
    translated_text_path: str,
    manifest_path: str,
    output_epub_path: str,
) -> None:
    """Rebuild an EPUB by replacing extracted blocks with translated lines."""
    require_epub_dependencies()

    book = epub.read_epub(original_epub_path)
    translated_lines = _load_translated_lines(translated_text_path)
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    expected_lines = sum(
        sum(block.get("lines", 1) for block in item["blocks"])
        for item in manifest["items"]
    )
    if len(translated_lines) != expected_lines:
        raise ValueError(
            "Translated text line count does not match EPUB manifest: "
            f"expected {expected_lines}, found {len(translated_lines)}"
        )

    for manifest_item in manifest["items"]:
        item = book.get_item_with_id(manifest_item["item_id"])
        if not _is_text_document_item(item):
            raise ValueError(
                "EPUB manifest references a missing document item: "
                f"{manifest_item['item_id']}"
            )

        soup = BeautifulSoup(item.get_content(), "lxml")
        extraction_mode = manifest_item.get("extraction_mode", "blocks")
        if extraction_mode == "text_nodes":
            blocks = list(_iter_translatable_text_nodes(soup))
        else:
            blocks = list(_iter_translatable_blocks(soup))
        if len(blocks) != len(manifest_item["blocks"]):
            raise ValueError(
                "EPUB structure changed while rebuilding "
                f"{manifest_item['file_name']}: expected "
                f"{len(manifest_item['blocks'])} blocks, found {len(blocks)}"
            )

        for block, block_info in zip(blocks, manifest_item["blocks"]):
            line_start = block_info["line"]
            line_count = block_info.get("lines", 1)
            translated_text = " ".join(
                translated_lines[line_start : line_start + line_count]
            )
            if extraction_mode == "text_nodes":
                block.replace_with(translated_text)
            else:
                block.clear()
                block.append(translated_text)

        item.set_content(str(soup).encode("utf-8"))

    os.makedirs(os.path.abspath(os.path.dirname(output_epub_path)), exist_ok=True)
    epub.write_epub(output_epub_path, book)

import json
import os
import re
from typing import Iterable, List


try:
    import ebooklib
    import lxml
    from bs4 import BeautifulSoup
    from ebooklib import epub
except ImportError:  # pragma: no cover - exercised through require_epub_dependencies
    ebooklib = None
    lxml = None
    BeautifulSoup = None
    epub = None


EPUB_DEPENDENCY_MESSAGE = (
    "EPUB support requires ebooklib, beautifulsoup4, and lxml. "
    "Install them with: pip install ebooklib beautifulsoup4 lxml"
)

BLOCK_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "figcaption",
    "caption",
    "dt",
    "dd",
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
MAX_TRANSLATION_UNIT_CHARS = 600


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


def _is_document_item(item) -> bool:
    return item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT


def _spine_document_items(book) -> Iterable:
    for spine_item in book.spine:
        item_id = spine_item[0] if isinstance(spine_item, tuple) else spine_item
        item = book.get_item_with_id(item_id)
        if _is_document_item(item):
            yield item


def _iter_translatable_blocks(soup: BeautifulSoup) -> Iterable:
    for block in soup.find_all(BLOCK_TAGS):
        if block.find(BLOCK_TAGS):
            continue
        text = _normalize_text(block.get_text(" ", strip=True))
        if text:
            yield block


def epub_to_text(epub_path: str, text_path: str, manifest_path: str) -> None:
    """Extract translatable EPUB blocks to one UTF-8 text line per block."""
    require_epub_dependencies()

    book = epub.read_epub(epub_path)
    manifest = {
        "source_epub": os.path.abspath(epub_path),
        "items": [],
    }
    lines: List[str] = []

    for item in _spine_document_items(book):
        content = item.get_content()
        soup = BeautifulSoup(content, "lxml")
        blocks = []
        for block_index, block in enumerate(_iter_translatable_blocks(soup)):
            text = _normalize_text(block.get_text(" ", strip=True))
            text_lines = split_translation_units(text)
            blocks.append(
                {
                    "line": len(lines),
                    "lines": len(text_lines),
                    "block_index": block_index,
                    "tag": block.name,
                    "original_text": text,
                }
            )
            lines.extend(text_lines)

        if blocks:
            manifest["items"].append(
                {
                    "item_id": item.get_id(),
                    "file_name": item.get_name(),
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
        if not _is_document_item(item):
            raise ValueError(
                "EPUB manifest references a missing document item: "
                f"{manifest_item['item_id']}"
            )

        soup = BeautifulSoup(item.get_content(), "lxml")
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
            block.clear()
            block.append(" ".join(translated_lines[line_start : line_start + line_count]))

        item.set_content(str(soup).encode("utf-8"))

    os.makedirs(os.path.abspath(os.path.dirname(output_epub_path)), exist_ok=True)
    epub.write_epub(output_epub_path, book)

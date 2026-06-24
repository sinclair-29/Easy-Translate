import json
import os
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List
from zipfile import ZipFile


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
DEFAULT_TRANSLATED_EPUB_LANGUAGE = "zh-Hans"
CHINESE_STYLE_ID = "easytranslate-chinese-style"
CHINESE_STYLE = """
html[lang|="zh"], body {
  text-align: start;
  line-height: 1.65;
}
p, div, li, blockquote, dd, dt, td, th {
  text-align: start;
  word-break: break-word;
}
a[style*="vertical-align: super"],
a[class] {
  text-decoration: none;
}
""".strip()
EXCLUDED_TEXT_PARENTS = {
    "head",
    "link",
    "meta",
    "script",
    "style",
    "svg",
    "title",
}
NOTE_LINK_CLASSES = {
    "class_16563",
    "class_16588",
    "class_16870",
    "class_18946",
    "class_19502",
    "class_19515",
}
LINK_PRESERVING_TAGS = {
    "a",
    "audio",
    "embed",
    "iframe",
    "img",
    "object",
    "source",
    "track",
    "video",
}


def require_epub_dependencies() -> None:
    if ebooklib is None or lxml is None or BeautifulSoup is None or epub is None:
        raise ImportError(EPUB_DEPENDENCY_MESSAGE)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_output_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\s+([，。！？；：、）】》」』])", r"\1", text)
    text = re.sub(r"([（【《「『])\s+", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _tag_name_matches(tag, name: str) -> bool:
    tag_name = (getattr(tag, "name", "") or "").lower()
    name = name.lower()
    return tag_name == name or tag_name.endswith(f":{name}")


def _find_first_tag(soup: BeautifulSoup, name: str):
    return soup.find(lambda tag: _tag_name_matches(tag, name))


def _find_all_tags(soup: BeautifulSoup, name: str):
    return soup.find_all(lambda tag: _tag_name_matches(tag, name))


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
        if _is_structural_note_link(parent):
            continue
        text = _normalize_text(str(node))
        if text:
            yield node


def _is_numeric_text(text: str) -> bool:
    return bool(re.fullmatch(r"[\[\(]?\d{1,4}[\]\)]?", _normalize_text(text)))


def _is_structural_note_link(tag) -> bool:
    if tag is None or getattr(tag, "name", None) != "a":
        return False
    text = _normalize_text(tag.get_text("", strip=True))
    if not _is_numeric_text(text):
        return False
    classes = set(tag.get("class") or [])
    if classes & NOTE_LINK_CLASSES:
        return True
    href = tag.get("href", "")
    return "#" in href and len(text) <= 4


def _strip_duplicate_note_number(text: str, next_node) -> str:
    sibling = next_node
    while sibling is not None:
        if isinstance(sibling, NavigableString):
            if _normalize_text(str(sibling)):
                return text
            sibling = sibling.next_sibling
            continue
        if getattr(sibling, "name", None) == "span" and not _normalize_text(
            sibling.get_text(" ", strip=True)
        ):
            sibling = sibling.next_sibling
            continue
        if _is_structural_note_link(sibling):
            note_number = re.escape(_normalize_text(sibling.get_text("", strip=True)))
            return re.sub(rf"\s*{note_number}\s*$", "", text).rstrip()
        return text
    return text


def _has_linked_descendant(block) -> bool:
    return bool(block.find(LINK_PRESERVING_TAGS))


def _replace_block_text_preserving_links(block, translated_text: str) -> None:
    first_text_node = None
    for node in block.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent is not None and parent.name in LINK_PRESERVING_TAGS:
            continue
        if not _normalize_text(str(node)):
            continue
        if first_text_node is None:
            first_text_node = node
        else:
            node.replace_with("")

    if first_text_node is not None:
        first_text_node.replace_with(translated_text)
    else:
        block.insert(0, translated_text)


def epub_to_text(epub_path: str, text_path: str, manifest_path: str) -> None:
    """Extract translatable EPUB blocks to one UTF-8 text line per block."""
    require_epub_dependencies()

    book = epub.read_epub(epub_path)
    manifest = {
        "source_epub": os.path.abspath(epub_path),
        "items": [],
        "metadata": [],
        "target_language_code": DEFAULT_TRANSLATED_EPUB_LANGUAGE,
    }
    lines: List[str] = []

    _append_epub_metadata_units(epub_path, manifest, lines)

    for item in _document_items_in_reading_order(book):
        content = item.get_content()
        soup = BeautifulSoup(content, "lxml")
        blocks = []
        candidates = list(_iter_translatable_text_nodes(soup))
        extraction_mode = "text_nodes"

        for block_index, block in enumerate(candidates):
            text = _normalize_text(str(block))
            tag = block.parent.name if block.parent is not None else None
            text_lines = split_translation_units(text)
            if not text_lines:
                continue
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


def _ensure_toc_uids(book) -> None:
    """ebooklib cannot write NCX entries whose TOC uid is None."""
    counter = 0

    def ensure_entry_uid(entry) -> None:
        nonlocal counter
        if isinstance(entry, (list, tuple)):
            for child in entry:
                ensure_entry_uid(child)
            return

        if hasattr(entry, "uid") and not getattr(entry, "uid", None):
            counter += 1
            entry.uid = f"toc-{counter}"

        children = getattr(entry, "children", None)
        if children:
            ensure_entry_uid(children)

    ensure_entry_uid(getattr(book, "toc", None))


def _epub_root_dir(epub_path: str) -> str:
    with ZipFile(epub_path) as epub_zip:
        container = epub_zip.read("META-INF/container.xml")
    root = ET.fromstring(container)
    for element in root.iter():
        if element.tag.endswith("rootfile"):
            full_path = element.attrib.get("full-path", "")
            return posixpath.dirname(full_path)
    return ""


def _epub_opf_path(epub_path: str) -> str:
    with ZipFile(epub_path) as epub_zip:
        container = epub_zip.read("META-INF/container.xml")
    root = ET.fromstring(container)
    for element in root.iter():
        if element.tag.endswith("rootfile"):
            return element.attrib.get("full-path", "")
    return ""


def _manifest_item_hrefs(opf_content: bytes) -> Dict[str, str]:
    root = ET.fromstring(opf_content)
    items = {}
    for element in root.iter():
        if element.tag.endswith("item"):
            item_id = element.attrib.get("id")
            href = element.attrib.get("href")
            if item_id and href:
                items[item_id] = href
    return items


def _append_translation_unit(
    manifest_units: List[Dict],
    lines: List[str],
    text: str,
    **metadata,
) -> None:
    text_lines = split_translation_units(_normalize_text(text))
    if not text_lines:
        return
    manifest_units.append(
        {
            "line": len(lines),
            "lines": len(text_lines),
            "original_text": _normalize_text(text),
            **metadata,
        }
    )
    lines.extend(text_lines)


def _append_epub_metadata_units(
    epub_path: str,
    manifest: Dict,
    lines: List[str],
) -> None:
    root_dir = _epub_root_dir(epub_path)
    opf_path = _epub_opf_path(epub_path)
    if not opf_path:
        return

    with ZipFile(epub_path) as epub_zip:
        if opf_path not in epub_zip.namelist():
            return
        opf_content = epub_zip.read(opf_path)
        opf_soup = BeautifulSoup(opf_content, "xml")
        title = _find_first_tag(opf_soup, "title")
        if title is not None:
            _append_translation_unit(
                manifest["metadata"],
                lines,
                title.get_text(" ", strip=True),
                kind="opf_title",
                zip_name=opf_path,
            )

        item_hrefs = _manifest_item_hrefs(opf_content)
        toc_href = item_hrefs.get("toc")
        if toc_href:
            ncx_path = posixpath.normpath(posixpath.join(root_dir, toc_href))
            if ncx_path in epub_zip.namelist():
                ncx_soup = BeautifulSoup(epub_zip.read(ncx_path), "xml")
                for index, label in enumerate(_find_all_tags(ncx_soup, "navLabel")):
                    text_tag = _find_first_tag(label, "text")
                    if text_tag is None:
                        continue
                    _append_translation_unit(
                        manifest["metadata"],
                        lines,
                        text_tag.get_text(" ", strip=True),
                        kind="ncx_label",
                        zip_name=ncx_path,
                        index=index,
                    )

        nav_href = None
        for item_id, href in item_hrefs.items():
            if item_id.lower() == "nav":
                nav_href = href
                break
        if nav_href:
            nav_path = posixpath.normpath(posixpath.join(root_dir, nav_href))
            if nav_path in epub_zip.namelist():
                nav_soup = BeautifulSoup(epub_zip.read(nav_path), "lxml")
                for index, node in enumerate(_iter_translatable_text_nodes(nav_soup)):
                    _append_translation_unit(
                        manifest["metadata"],
                        lines,
                        str(node),
                        kind="nav_text",
                        zip_name=nav_path,
                        index=index,
                    )


def _zip_name_for_item(epub_zip: ZipFile, file_name: str, root_dir: str) -> str:
    normalized_file_name = file_name.lstrip("/")
    candidates = [normalized_file_name]
    if root_dir:
        candidates.append(posixpath.join(root_dir, normalized_file_name))

    names = set(epub_zip.namelist())
    for candidate in candidates:
        if candidate in names:
            return candidate

    matches = [
        name
        for name in names
        if name == normalized_file_name or name.endswith(f"/{normalized_file_name}")
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(f"Could not find EPUB item in package: {file_name}")


def _write_epub_preserving_package(
    original_epub_path: str,
    output_epub_path: str,
    replacements: Dict[str, bytes],
) -> None:
    output_dir = os.path.abspath(os.path.dirname(output_epub_path))
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = os.path.abspath(output_epub_path)
    source_path = os.path.abspath(original_epub_path)
    if final_output_path == source_path:
        temp_file = tempfile.NamedTemporaryFile(
            prefix=".easytranslate-",
            suffix=".epub",
            dir=output_dir,
            delete=False,
        )
        temp_file.close()
        write_path = temp_file.name
    else:
        write_path = final_output_path

    try:
        with ZipFile(original_epub_path, "r") as source_zip:
            infos = source_zip.infolist()
            mimetype_info = next(
                (info for info in infos if info.filename == "mimetype"),
                None,
            )
            ordered_infos = []
            if mimetype_info is not None:
                ordered_infos.append(mimetype_info)
            ordered_infos.extend(
                info for info in infos if info.filename != "mimetype"
            )

            with ZipFile(write_path, "w") as output_zip:
                for info in ordered_infos:
                    content = replacements.get(
                        info.filename,
                        source_zip.read(info.filename),
                    )
                    output_zip.writestr(info, content)
        if write_path != final_output_path:
            os.replace(write_path, final_output_path)
    except Exception:
        if write_path != final_output_path and os.path.exists(write_path):
            os.unlink(write_path)
        raise


def _translated_text_for_unit(unit: Dict, translated_lines: List[str]) -> str:
    line_start = unit["line"]
    line_count = unit.get("lines", 1)
    return _normalize_output_text(
        " ".join(translated_lines[line_start : line_start + line_count])
    )


def _set_xhtml_language(soup: BeautifulSoup, language_code: str) -> None:
    html = soup.find("html")
    if html is not None:
        html["lang"] = language_code
        html["xml:lang"] = language_code


def _append_chinese_css_overrides(content: bytes) -> bytes:
    text = content.decode("utf-8", errors="replace")
    if CHINESE_STYLE_ID in text:
        return content
    text = f"{text.rstrip()}\n\n/* {CHINESE_STYLE_ID} */\n{CHINESE_STYLE}\n"
    return text.encode("utf-8")


def _apply_metadata_replacements(
    replacements: Dict[str, bytes],
    epub_zip: ZipFile,
    metadata_units: List[Dict],
    translated_lines: List[str],
    language_code: str,
) -> None:
    units_by_zip: Dict[str, List[Dict]] = {}
    for unit in metadata_units:
        units_by_zip.setdefault(unit["zip_name"], []).append(unit)

    for zip_name, units in units_by_zip.items():
        if zip_name not in epub_zip.namelist():
            continue
        parser = "xml" if zip_name.endswith((".opf", ".ncx")) else "lxml"
        soup = BeautifulSoup(
            replacements.get(zip_name, epub_zip.read(zip_name)),
            parser,
        )
        for unit in units:
            translated_text = _translated_text_for_unit(unit, translated_lines)
            if unit["kind"] == "opf_title":
                title = _find_first_tag(soup, "title")
                if title is not None:
                    title.clear()
                    title.append(translated_text)
            elif unit["kind"] == "ncx_label":
                labels = _find_all_tags(soup, "navLabel")
                if unit["index"] < len(labels):
                    text_tag = _find_first_tag(labels[unit["index"]], "text")
                    if text_tag is not None:
                        text_tag.clear()
                        text_tag.append(translated_text)
            elif unit["kind"] == "nav_text":
                nodes = list(_iter_translatable_text_nodes(soup))
                if unit["index"] < len(nodes):
                    nodes[unit["index"]].replace_with(translated_text)

        language_tag = _find_first_tag(soup, "language")
        if language_tag is not None:
            language_tag.clear()
            language_tag.append(language_code)
        _set_xhtml_language(soup, language_code)
        replacements[zip_name] = str(soup).encode("utf-8")


def _apply_language_and_css_replacements(
    replacements: Dict[str, bytes],
    epub_zip: ZipFile,
    language_code: str,
) -> None:
    for zip_name in epub_zip.namelist():
        if zip_name.endswith((".xhtml", ".html")):
            soup = BeautifulSoup(
                replacements.get(zip_name, epub_zip.read(zip_name)),
                "lxml",
            )
            _set_xhtml_language(soup, language_code)
            replacements[zip_name] = str(soup).encode("utf-8")
        elif zip_name.endswith(".css"):
            replacements[zip_name] = _append_chinese_css_overrides(
                replacements.get(zip_name, epub_zip.read(zip_name))
            )


def text_to_epub(
    original_epub_path: str,
    translated_text_path: str,
    manifest_path: str,
    output_epub_path: str,
    target_language_code: str = DEFAULT_TRANSLATED_EPUB_LANGUAGE,
) -> None:
    """Rebuild an EPUB by replacing extracted blocks with translated lines."""
    require_epub_dependencies()

    translated_lines = _load_translated_lines(translated_text_path)
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    expected_lines = sum(
        sum(block.get("lines", 1) for block in item["blocks"])
        for item in manifest["items"]
    ) + sum(
        unit.get("lines", 1) for unit in manifest.get("metadata", [])
    )
    if len(translated_lines) != expected_lines:
        raise ValueError(
            "Translated text line count does not match EPUB manifest: "
            f"expected {expected_lines}, found {len(translated_lines)}"
        )

    root_dir = _epub_root_dir(original_epub_path)
    replacements: Dict[str, bytes] = {}
    with ZipFile(original_epub_path) as epub_zip:
        for manifest_item in manifest["items"]:
            zip_name = _zip_name_for_item(
                epub_zip,
                manifest_item["file_name"],
                root_dir,
            )
            soup = BeautifulSoup(epub_zip.read(zip_name), "lxml")
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
                translated_text = _translated_text_for_unit(block_info, translated_lines)
                if extraction_mode == "text_nodes":
                    translated_text = _strip_duplicate_note_number(
                        translated_text,
                        block.next_sibling,
                    )
                    block.replace_with(translated_text)
                elif _has_linked_descendant(block):
                    _replace_block_text_preserving_links(block, translated_text)
                else:
                    block.clear()
                    block.append(translated_text)

            replacements[zip_name] = str(soup).encode("utf-8")

        _apply_metadata_replacements(
            replacements,
            epub_zip,
            manifest.get("metadata", []),
            translated_lines,
            target_language_code,
        )
        _apply_language_and_css_replacements(
            replacements,
            epub_zip,
            target_language_code,
        )

    _write_epub_preserving_package(
        original_epub_path,
        output_epub_path,
        replacements,
    )

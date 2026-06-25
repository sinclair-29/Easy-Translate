import json
import os
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
from zipfile import ZipFile


try:
    import ebooklib
    import lxml
    from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
    from ebooklib import epub
except ImportError:  # pragma: no cover - exercised through require_epub_dependencies
    ebooklib = None
    lxml = None
    BeautifulSoup = None
    NavigableString = None
    XMLParsedAsHTMLWarning = None
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

DEFAULT_TRANSLATED_EPUB_LANGUAGE = "zh-Hans"
CHINESE_STYLE_ID = "easytranslate-chinese-style"
ZH_BODY_CLASS = "easytranslate-zh-body"
NOTEREF_CLASS = "easytranslate-noteref"
CHINESE_STYLE = """
html[lang|="zh"], body {
  text-align: start;
  line-height: 1.65;
}
p, div, li, blockquote, dd, dt, td, th {
  text-align: start;
  word-break: break-word;
}
.easytranslate-zh-body {
  text-indent: 2em;
  text-align: start;
  hyphens: none;
  -webkit-hyphens: none;
}
a.noteref,
a.easytranslate-noteref,
a[epub\\:type~="noteref"] {
  color: #0645ad;
  font-size: 0.75em;
  line-height: 1;
  vertical-align: super;
  text-decoration: none;
  text-indent: 0;
  padding: 0 0.15em;
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
    "noteref",
    NOTEREF_CLASS,
    "class_16563",
    "class_16588",
    "class_16870",
    "class_18946",
    "class_19502",
    "class_19515",
}
BACKLINK_CLASSES = {
    "backlink",
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
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TABLE_CELL_TAGS = {"td", "th"}
CAPTION_TAGS = {"caption", "figcaption"}
TOC_HINTS = {"nav", "toc", "contents", "table-of-contents"}
NOTE_HINTS = {"note", "notes", "footnote", "footnotes", "endnote", "endnotes"}
BIBLIOGRAPHY_HINTS = {
    "bibliography",
    "biblio",
    "references",
    "reference",
    "works-cited",
    "workscited",
}
INDEX_HINTS = {"index", "indices"}
STRUCTURAL_TEXT_RESIDUES = {
    "body",
    "head",
    "html",
    "xhtml",
}
ROMAN_PAGE_MARKERS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
    "xvii",
    "xviii",
    "xix",
    "xx",
}
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
XML_DECLARATION_RE = re.compile(r"^\??xml\b|^<\?xml\b", re.IGNORECASE)


def require_epub_dependencies() -> None:
    if ebooklib is None or lxml is None or BeautifulSoup is None or epub is None:
        raise ImportError(EPUB_DEPENDENCY_MESSAGE)


def _content_to_text(content) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _ensure_epub_namespace(content) -> str:
    text = _content_to_text(content)
    if "epub:" not in text or "xmlns:epub" in text:
        return text
    return re.sub(
        r"(<html\b(?![^>]*\bxmlns:epub=)[^>]*)(>)",
        r'\1 xmlns:epub="http://www.idpf.org/2007/ops"\2',
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _parse_html(content):
    if XMLParsedAsHTMLWarning is None:
        return BeautifulSoup(_ensure_epub_namespace(content), "lxml")
    import warnings

    safe_content = _ensure_epub_namespace(content)
    probe = safe_content[:512].lower()
    if "<?xml" in probe:
        return BeautifulSoup(safe_content, "xml")
    if "xmlns=" in probe:
        return BeautifulSoup(safe_content, "xml")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        return BeautifulSoup(safe_content, "lxml")


def _parse_legacy_html(content):
    if XMLParsedAsHTMLWarning is None:
        return BeautifulSoup(content, "lxml")
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        return BeautifulSoup(content, "lxml")


def _parse_xml(content):
    return BeautifulSoup(content, "xml")


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_source_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def _normalize_output_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\s+([，。！？；：、）】》」』])", r"\1", text)
    text = re.sub(r"([（【《「『])\s+", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _is_punctuation_only(text: str) -> bool:
    return bool(re.fullmatch(r"[\W_]+", text, flags=re.UNICODE))


def _is_short_roman_page_marker(text: str) -> bool:
    marker = text.strip().strip("[](){}.,;:").lower()
    return marker in ROMAN_PAGE_MARKERS


def _is_translatable_text(text: str) -> bool:
    text = _normalize_text(text)
    if not text:
        return False
    lowered = text.lower()
    if lowered in STRUCTURAL_TEXT_RESIDUES:
        return False
    if XML_DECLARATION_RE.search(text):
        return False
    if URL_RE.fullmatch(text):
        return False
    if _is_numeric_text(text):
        return False
    if _is_short_roman_page_marker(text):
        return False
    if _is_punctuation_only(text):
        return False
    return True


def _tag_name_matches(tag, name: str) -> bool:
    tag_name = (getattr(tag, "name", "") or "").lower()
    name = name.lower()
    return tag_name == name or tag_name.endswith(f":{name}")


def _find_first_tag(soup: BeautifulSoup, name: str):
    return soup.find(lambda tag: _tag_name_matches(tag, name))


def _find_all_tags(soup: BeautifulSoup, name: str):
    return soup.find_all(lambda tag: _tag_name_matches(tag, name))


def _local_tag_name(tag) -> str:
    tag_name = (getattr(tag, "name", "") or "").lower()
    return tag_name.rsplit(":", 1)[-1]


def _contains_hint(text: str, hints: set) -> bool:
    normalized = (text or "").lower().replace("_", "-")
    tokens = set(re.split(r"[^a-z0-9]+", normalized))
    return any(
        hint in tokens or ("-" in hint and hint in normalized)
        for hint in hints
    )


def _item_hint_text(item) -> str:
    return " ".join(
        str(value)
        for value in (
            getattr(item, "id", ""),
            item.get_id() if hasattr(item, "get_id") else "",
            item.get_name() if hasattr(item, "get_name") else "",
        )
        if value
    )


def _block_hint_text(block) -> str:
    values = []
    current = block
    while current is not None and getattr(current, "name", None):
        for attr in ("id", "role", "epub:type", "type"):
            value = current.get(attr)
            if value:
                values.append(" ".join(value) if isinstance(value, list) else str(value))
        classes = current.get("class") or []
        if classes:
            values.append(" ".join(str(value) for value in classes))
        current = current.parent
    return " ".join(values)


def _kind_from_hints(hint_text: str):
    if _contains_hint(hint_text, TOC_HINTS):
        return "toc"
    if _contains_hint(hint_text, NOTE_HINTS):
        return "note"
    if _contains_hint(hint_text, BIBLIOGRAPHY_HINTS):
        return "bibliography"
    if _contains_hint(hint_text, INDEX_HINTS):
        return "index"
    return None


def _infer_block_kind(block, item) -> str:
    item_kind = _kind_from_hints(_item_hint_text(item))
    if item_kind is not None:
        return item_kind

    tag_name = _local_tag_name(block)
    if tag_name in HEADING_TAGS:
        return "heading"
    if tag_name in TABLE_CELL_TAGS:
        return "table_cell"
    if tag_name in CAPTION_TAGS:
        return "caption"

    block_kind = _kind_from_hints(_block_hint_text(block))
    if block_kind is not None:
        return block_kind
    return "body"


def split_translation_units(text: str) -> List[str]:
    text = _normalize_source_text(text)
    if not _is_translatable_text(text):
        return []
    return [text]


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


def _visible_text_for_block(block) -> str:
    parts = []
    for node in block.find_all(string=True):
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
            parts.append(text)
    return _normalize_source_text(" ".join(parts))


def _is_visible_text_node(node) -> bool:
    if not isinstance(node, NavigableString):
        return False
    parent = node.parent
    if parent is None:
        return False
    if parent.name and parent.name.lower() in EXCLUDED_TEXT_PARENTS:
        return False
    if parent.find_parent(EXCLUDED_TEXT_PARENTS):
        return False
    if _is_structural_note_link(parent):
        return False
    return bool(_normalize_text(str(node)))


def _iter_translatable_blocks(soup: BeautifulSoup) -> Iterable:
    for block in soup.find_all(BLOCK_TAGS):
        if block.find(BLOCK_TAGS):
            continue
        text = _visible_text_for_block(block)
        if _is_translatable_text(text):
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
    return bool(re.fullmatch(r"[\[\(]?\d{1,4}[\]\)]?\.?", _normalize_text(text)))


def _tag_classes(tag) -> set:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return {str(value).lower() for value in classes}


def _epub_type_values(tag) -> set:
    values = []
    for attr in ("epub:type", "type"):
        value = tag.get(attr)
        if value:
            values.extend(str(value).lower().split())
    return set(values)


def _is_noteref_link(tag) -> bool:
    if tag is None or _local_tag_name(tag) != "a":
        return False
    text = _normalize_text(tag.get_text("", strip=True))
    if not _is_numeric_text(text):
        return False
    classes = _tag_classes(tag)
    epub_types = _epub_type_values(tag)
    href = str(tag.get("href", "")).lower()
    link_context = " ".join(
        str(value).lower()
        for value in (
            tag.get("id", ""),
            tag.get("role", ""),
            href,
        )
        if value
    )
    return (
        bool(classes & NOTE_LINK_CLASSES)
        or "noteref" in epub_types
        or "footnote" in epub_types
        or "endnote" in epub_types
        or ("#" in href and any(hint in link_context for hint in NOTE_HINTS))
    )


def _is_backlink_note_link(tag) -> bool:
    if tag is None or _local_tag_name(tag) != "a":
        return False
    text = _normalize_text(tag.get_text("", strip=True))
    if not _is_numeric_text(text):
        return False
    classes = _tag_classes(tag)
    href = str(tag.get("href", "")).lower()
    return bool(classes & BACKLINK_CLASSES) and bool(href)


def _is_structural_note_link(tag) -> bool:
    if tag is None or _local_tag_name(tag) != "a":
        return False
    text = _normalize_text(tag.get_text("", strip=True))
    if not _is_numeric_text(text):
        return False
    if _is_noteref_link(tag) or _is_backlink_note_link(tag):
        return True
    href = tag.get("href", "")
    return "#" in href and len(text) <= 4


def _ensure_noteref_class(tag) -> None:
    if not _is_noteref_link(tag):
        return
    _ensure_tag_class(tag, NOTEREF_CLASS)


def _ensure_tag_class(tag, class_name: str) -> None:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    classes = list(classes)
    if class_name not in classes:
        classes.append(class_name)
    tag["class"] = classes


def _apply_noteref_classes(block) -> None:
    for link in block.find_all(lambda tag: _is_noteref_link(tag)):
        _ensure_noteref_class(link)


def _apply_chinese_body_class(block, block_info: Dict) -> None:
    if block_info.get("kind", "body") != "body":
        return
    if _local_tag_name(block) != "p":
        return
    _ensure_tag_class(block, ZH_BODY_CLASS)


def _last_noteref_link(block):
    links = block.find_all(lambda tag: _is_noteref_link(tag))
    return links[-1] if links else None


def _strip_duplicate_block_note_number(text: str, block) -> str:
    note_link = _last_noteref_link(block)
    if note_link is None:
        return text
    note_number = re.escape(_normalize_text(note_link.get_text("", strip=True)))
    return re.sub(rf"(?<!\d)\s*{note_number}\s*$", "", text).rstrip()


def _strip_duplicate_note_number_for_link(text: str, note_link) -> str:
    if note_link is None:
        return text
    note_number = re.escape(_normalize_text(note_link.get_text("", strip=True)))
    return re.sub(rf"(?<!\d)\s*{note_number}\s*$", "", text).rstrip()


def _leading_structural_note_link(block):
    for node in block.descendants:
        if _is_structural_note_link(node):
            return node
        if _is_visible_text_node(node):
            return None
    return None


def _strip_leading_duplicate_note_number(text: str, block) -> str:
    note_link = _leading_structural_note_link(block)
    if note_link is None:
        return text
    note_number = re.escape(_normalize_text(note_link.get_text("", strip=True)))
    return re.sub(rf"^\s*{note_number}\s*", "", text, count=1).lstrip()


def _has_noteref_descendant(block) -> bool:
    return bool(block.find(lambda tag: _is_noteref_link(tag)))


def _noteref_text_segments(block) -> List[Dict]:
    segments = []
    current_nodes = []
    current_parts = []

    def flush(next_note=None) -> None:
        nonlocal current_nodes, current_parts
        text = _normalize_source_text(" ".join(current_parts))
        if text:
            segments.append(
                {
                    "text": text,
                    "nodes": current_nodes,
                    "next_note": next_note,
                }
            )
        current_nodes = []
        current_parts = []

    for node in block.descendants:
        if _is_noteref_link(node):
            flush(next_note=node)
            continue
        if not _is_visible_text_node(node):
            continue
        current_nodes.append(node)
        current_parts.append(_normalize_text(str(node)))

    flush()
    return segments


def _replace_noteref_segmented_block(
    block,
    block_info: Dict,
    translated_lines: List[str],
) -> None:
    segments = _noteref_text_segments(block)
    segment_infos = block_info.get("segments", [])
    if len(segments) != len(segment_infos):
        raise ValueError(
            "EPUB note segment structure changed while rebuilding "
            f"block {block_info.get('block_index')}: expected "
            f"{len(segment_infos)} segments, found {len(segments)}"
        )

    for segment, segment_info in zip(segments, segment_infos):
        translated_text = _translated_text_for_unit(segment_info, translated_lines)
        translated_text = _strip_duplicate_note_number_for_link(
            translated_text,
            segment.get("next_note"),
        )
        nodes = segment["nodes"]
        if not nodes:
            continue
        nodes[0].replace_with(translated_text)
        for node in nodes[1:]:
            node.replace_with("")


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
            return re.sub(rf"(?<!\d)\s*{note_number}\s*$", "", text).rstrip()
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
        soup = _parse_html(content)
        blocks = []
        candidates = list(_iter_translatable_blocks(soup))
        extraction_mode = "blocks"

        for block_index, block in enumerate(candidates):
            text = _visible_text_for_block(block)
            tag = block.name
            kind = _infer_block_kind(block, item)
            if _has_noteref_descendant(block):
                segment_infos = []
                for segment in _noteref_text_segments(block):
                    text_lines = split_translation_units(segment["text"])
                    if not text_lines:
                        continue
                    segment_infos.append(
                        {
                            "line": len(lines),
                            "lines": len(text_lines),
                            "original_text": segment["text"],
                        }
                    )
                    lines.extend(text_lines)
                if not segment_infos:
                    continue
                blocks.append(
                    {
                        "line": segment_infos[0]["line"],
                        "lines": sum(
                            segment_info.get("lines", 1)
                            for segment_info in segment_infos
                        ),
                        "block_index": block_index,
                        "tag": tag,
                        "kind": kind,
                        "original_text": text,
                        "segment_mode": "noteref_text_segments",
                        "segments": segment_infos,
                    }
                )
            else:
                text_lines = split_translation_units(text)
                if not text_lines:
                    continue
                blocks.append(
                    {
                        "line": len(lines),
                        "lines": len(text_lines),
                        "block_index": block_index,
                        "tag": tag,
                        "kind": kind,
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


def _opf_manifest_items(opf_content: bytes) -> List[Dict[str, str]]:
    root = ET.fromstring(opf_content)
    items = []
    for element in root.iter():
        if element.tag.endswith("item"):
            item_id = element.attrib.get("id", "")
            href = element.attrib.get("href", "")
            if not item_id or not href:
                continue
            items.append(
                {
                    "id": item_id,
                    "href": href,
                    "media_type": element.attrib.get("media-type", ""),
                    "properties": element.attrib.get("properties", ""),
                }
            )
    return items


def _opf_spine_toc_id(opf_content: bytes) -> str:
    root = ET.fromstring(opf_content)
    for element in root.iter():
        if element.tag.endswith("spine"):
            return element.attrib.get("toc", "")
    return ""


def _dedupe_preserving_order(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _metadata_item_path(root_dir: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(root_dir, href))


def _ncx_paths_from_opf(opf_content: bytes, root_dir: str) -> List[str]:
    items = _opf_manifest_items(opf_content)
    items_by_id = {item["id"]: item for item in items}
    candidate_hrefs = []

    spine_toc_id = _opf_spine_toc_id(opf_content)
    if spine_toc_id in items_by_id:
        candidate_hrefs.append(items_by_id[spine_toc_id]["href"])

    for item in items:
        item_id = item["id"].lower()
        href = item["href"]
        media_type = item["media_type"].lower()
        if (
            media_type == "application/x-dtbncx+xml"
            or href.lower().endswith(".ncx")
            or item_id in {"toc", "ncx"}
        ):
            candidate_hrefs.append(href)

    return _dedupe_preserving_order(
        _metadata_item_path(root_dir, href) for href in candidate_hrefs
    )


def _nav_paths_from_opf(opf_content: bytes, root_dir: str) -> List[str]:
    candidate_hrefs = []
    for item in _opf_manifest_items(opf_content):
        item_id = item["id"].lower()
        href = item["href"]
        href_name = posixpath.basename(href).lower()
        media_type = item["media_type"].lower()
        properties = set(item["properties"].lower().split())
        if (
            "nav" in properties
            or item_id == "nav"
            or (
                media_type == "application/xhtml+xml"
                and href_name in {"nav.xhtml", "nav.html", "toc.xhtml", "toc.html"}
            )
        ):
            candidate_hrefs.append(href)

    return _dedupe_preserving_order(
        _metadata_item_path(root_dir, href) for href in candidate_hrefs
    )


def _append_translation_unit(
    manifest_units: List[Dict],
    lines: List[str],
    text: str,
    **metadata,
) -> None:
    text = _normalize_source_text(text)
    text_lines = split_translation_units(text)
    if not text_lines:
        return
    manifest_units.append(
        {
            "line": len(lines),
            "lines": len(text_lines),
            "original_text": text,
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
                unit_kind="metadata",
                zip_name=opf_path,
            )

        for ncx_path in _ncx_paths_from_opf(opf_content, root_dir):
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
                        unit_kind="metadata",
                        zip_name=ncx_path,
                        index=index,
                    )

        for nav_path in _nav_paths_from_opf(opf_content, root_dir):
            if nav_path in epub_zip.namelist():
                nav_soup = _parse_html(epub_zip.read(nav_path))
                for index, node in enumerate(_iter_translatable_text_nodes(nav_soup)):
                    _append_translation_unit(
                        manifest["metadata"],
                        lines,
                        str(node),
                        kind="nav_text",
                        unit_kind="metadata",
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
    for name, content in replacements.items():
        if name.endswith((".xhtml", ".html")):
            _validate_xhtml_bytes(name, content)
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
        for tag in soup.find_all(True):
            if any(str(attr).startswith("epub:") for attr in tag.attrs):
                html["xmlns:epub"] = "http://www.idpf.org/2007/ops"
                break


def _clean_xhtml_markup(markup: str) -> str:
    markup = markup.lstrip("\ufeff\r\n\t ")
    markup = re.sub(r"^\s*<!--\?xml[^>]*\?-->\s*", "", markup, count=1)
    if markup.startswith("?xml"):
        html_index = markup.find("<html")
        if html_index != -1:
            markup = markup[html_index:]
    elif not markup.startswith("<") and "<html" in markup:
        markup = markup[markup.find("<html") :]

    if "epub:" in markup and "xmlns:epub" not in markup:
        markup = re.sub(
            r"(<html\b(?![^>]*\bxmlns:epub=)[^>]*)(>)",
            r'\1 xmlns:epub="http://www.idpf.org/2007/ops"\2',
            markup,
            count=1,
            flags=re.IGNORECASE,
        )
    if "<html" in markup and "xmlns=" not in markup.split(">", 1)[0]:
        markup = re.sub(
            r"(<html\b[^>]*)(>)",
            r'\1 xmlns="http://www.w3.org/1999/xhtml"\2',
            markup,
            count=1,
            flags=re.IGNORECASE,
        )
    if not markup.startswith("<?xml"):
        markup = f'<?xml version="1.0" encoding="utf-8"?>\n{markup}'
    return markup


def _validate_xhtml_bytes(name: str, content: bytes) -> None:
    stripped = content.lstrip()
    if not stripped.startswith(b"<"):
        raise ValueError(f"Invalid EPUB XHTML does not start with '<': {name}")
    if b"<html" not in stripped.lower():
        raise ValueError(f"Invalid EPUB XHTML missing <html> root: {name}")
    try:
        ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid EPUB XHTML XML in {name}: {exc}") from exc


def _serialized_xhtml(soup: BeautifulSoup) -> bytes:
    html = soup.find("html")
    body = soup.find("body")
    if body is not None:
        for child in list(body.contents):
            if isinstance(child, NavigableString) and "?xml" in str(child).lower():
                child.extract()

    markup = str(html if html is not None else soup)
    content = _clean_xhtml_markup(markup).encode("utf-8")
    _validate_xhtml_bytes("<generated>", content)
    return content


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
        soup = (
            _parse_xml(replacements.get(zip_name, epub_zip.read(zip_name)))
            if parser == "xml"
            else _parse_html(replacements.get(zip_name, epub_zip.read(zip_name)))
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
        if zip_name.endswith((".xhtml", ".html")):
            replacements[zip_name] = _serialized_xhtml(soup)
        else:
            replacements[zip_name] = str(soup).encode("utf-8")


def _apply_language_and_css_replacements(
    replacements: Dict[str, bytes],
    epub_zip: ZipFile,
    language_code: str,
) -> None:
    for zip_name in epub_zip.namelist():
        if zip_name.endswith((".xhtml", ".html")):
            if zip_name in replacements:
                continue
            soup = _parse_html(replacements.get(zip_name, epub_zip.read(zip_name)))
            _set_xhtml_language(soup, language_code)
            replacements[zip_name] = _serialized_xhtml(soup)
        elif zip_name.endswith(".css"):
            replacements[zip_name] = _append_chinese_css_overrides(
                replacements.get(zip_name, epub_zip.read(zip_name))
            )


def _blocks_for_manifest_item(soup: BeautifulSoup, extraction_mode: str) -> List:
    if extraction_mode == "text_nodes":
        return list(_iter_translatable_text_nodes(soup))
    return list(_iter_translatable_blocks(soup))


@lru_cache(maxsize=16)
def _fallback_item_content(original_epub_path: str, item_id: str):
    book = epub.read_epub(original_epub_path)
    item = book.get_item_with_id(item_id)
    if item is None:
        return None
    return item.get_content()


def _soup_and_blocks_for_rebuild(
    original_epub_path: str,
    epub_zip: ZipFile,
    zip_name: str,
    manifest_item: Dict,
) -> Tuple[BeautifulSoup, List]:
    extraction_mode = manifest_item.get("extraction_mode", "blocks")
    expected_count = len(manifest_item["blocks"])

    raw_content = epub_zip.read(zip_name)
    raw_soup = _parse_html(raw_content)
    raw_blocks = _blocks_for_manifest_item(raw_soup, extraction_mode)
    if len(raw_blocks) == expected_count:
        return raw_soup, raw_blocks

    legacy_soup = _parse_legacy_html(raw_content)
    legacy_blocks = _blocks_for_manifest_item(legacy_soup, extraction_mode)
    if len(legacy_blocks) == expected_count:
        return legacy_soup, legacy_blocks

    fallback_content = _fallback_item_content(
        original_epub_path,
        manifest_item["item_id"],
    )
    if fallback_content is not None:
        fallback_soup = _parse_html(fallback_content)
        fallback_blocks = _blocks_for_manifest_item(fallback_soup, extraction_mode)
        if len(fallback_blocks) == expected_count:
            return fallback_soup, fallback_blocks

        fallback_legacy_soup = _parse_legacy_html(fallback_content)
        fallback_legacy_blocks = _blocks_for_manifest_item(
            fallback_legacy_soup,
            extraction_mode,
        )
        if len(fallback_legacy_blocks) == expected_count:
            return fallback_legacy_soup, fallback_legacy_blocks

    raise ValueError(
        "EPUB structure changed while rebuilding "
        f"{manifest_item['file_name']}: expected {expected_count} blocks, "
        f"found {len(raw_blocks)}"
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
            extraction_mode = manifest_item.get("extraction_mode", "blocks")
            soup, blocks = _soup_and_blocks_for_rebuild(
                original_epub_path,
                epub_zip,
                zip_name,
                manifest_item,
            )

            for block, block_info in zip(blocks, manifest_item["blocks"]):
                translated_text = _translated_text_for_unit(block_info, translated_lines)
                if extraction_mode == "text_nodes":
                    translated_text = _strip_duplicate_note_number(
                        translated_text,
                        block.next_sibling,
                    )
                    block.replace_with(translated_text)
                else:
                    _apply_noteref_classes(block)
                    _apply_chinese_body_class(block, block_info)
                    if block_info.get("segment_mode") == "noteref_text_segments":
                        _replace_noteref_segmented_block(
                            block,
                            block_info,
                            translated_lines,
                        )
                    elif _has_linked_descendant(block):
                        translated_text = _strip_leading_duplicate_note_number(
                            translated_text,
                            block,
                        )
                        translated_text = _strip_duplicate_block_note_number(
                            translated_text,
                            block,
                        )
                        _replace_block_text_preserving_links(block, translated_text)
                    else:
                        translated_text = _strip_leading_duplicate_note_number(
                            translated_text,
                            block,
                        )
                        translated_text = _strip_duplicate_block_note_number(
                            translated_text,
                            block,
                        )
                        block.clear()
                        block.append(translated_text)

            _set_xhtml_language(soup, target_language_code)
            replacements[zip_name] = _serialized_xhtml(soup)

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

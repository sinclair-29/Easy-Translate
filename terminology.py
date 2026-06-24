import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional


DEFAULT_CANDIDATE_LIMIT = 300
DEFAULT_MEMORY_LIMIT = 150
DEFAULT_RELEVANT_LIMIT = 20

ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&.-]{1,}\b")
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'’.-]+"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'’.-]+|(?:of|and|the|for|in|on|to|de|la|le|van|von|da|di|du)\b)){0,5}"
)
HYPHENATED_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z]+(?:-[A-Za-z][A-Za-z]+)+\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
URL_IN_TEXT_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PUNCTUATION_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
FLAT_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "between",
    "could",
    "during",
    "first",
    "from",
    "have",
    "into",
    "other",
    "over",
    "should",
    "such",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "which",
    "while",
    "with",
    "would",
}

PRIORITY_KINDS = {"heading", "toc", "metadata", "caption"}


def _clean_candidate(text: str) -> str:
    text = text.translate(DASH_TRANSLATION)
    return re.sub(r"\s+", " ", text.strip(" \t\r\n.,;:!?()[]{}<>\"'“”‘’")).strip()


def _normalize_key(text: str) -> str:
    text = _clean_candidate(text).casefold()
    text = re.sub(r"[._/]+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"[^\w\s'’-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [_singularize_word(word) for word in text.split()]
    return " ".join(words)


def _singularize_word(word: str) -> str:
    if len(word) <= 4:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("ses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _variant_keys(text: str):
    key = _normalize_key(text)
    if not key:
        return set()
    variants = {key, key.replace("-", " "), key.replace(" ", "-")}
    return {variant for variant in variants if variant}


def _is_candidate_allowed(candidate: str) -> bool:
    if not candidate or len(candidate) < 2 or len(candidate) > 90:
        return False
    if URL_RE.match(candidate):
        return False
    if PUNCTUATION_ONLY_RE.match(candidate):
        return False
    if candidate.lower() in STOPWORDS:
        return False
    if candidate.isdigit():
        return False
    if not any(character.isalpha() for character in candidate):
        return False
    words = candidate.split()
    if len(words) == 1 and words[0].lower() in STOPWORDS:
        return False
    if len(words) == 1 and len(words[0]) <= 2 and not words[0].isupper():
        return False
    return True


def _metadata_kind(unit_metadata: Optional[List[dict]], index: int) -> str:
    if unit_metadata is None or index >= len(unit_metadata) or not unit_metadata[index]:
        return "body"
    return unit_metadata[index].get("kind") or "body"


def _add_candidate(
    candidate: str,
    *,
    line_index: int,
    kind: str,
    counts: Counter,
    line_hits: Dict[str, set],
    priority: Counter,
) -> None:
    candidate = _clean_candidate(candidate)
    if not _is_candidate_allowed(candidate):
        return

    key = _normalize_key(candidate)
    if not key:
        return
    counts[key] += 1
    line_hits[key].add(line_index)
    if kind in PRIORITY_KINDS:
        priority[key] += 2
    elif kind != "body":
        priority[key] += 1


def _iter_repeated_phrase_candidates(line: str) -> Iterable[str]:
    words = WORD_RE.findall(line)
    if len(words) < 3:
        return []

    lowered = [word.lower() for word in words]
    phrases = []
    for size in (2, 3):
        for index in range(0, len(lowered) - size + 1):
            phrase_words = lowered[index : index + size]
            if phrase_words[0] in STOPWORDS or phrase_words[-1] in STOPWORDS:
                continue
            if sum(word in STOPWORDS for word in phrase_words) > 1:
                continue
            phrases.append(" ".join(phrase_words))
    return phrases


def collect_term_candidates(
    source_lines: List[str],
    unit_metadata: Optional[List[dict]] = None,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> List[dict]:
    counts = Counter()
    priority = Counter()
    display = {}
    line_hits = defaultdict(set)
    phrase_counts = Counter()
    phrase_line_hits = defaultdict(set)

    for index, line in enumerate(source_lines):
        line = URL_IN_TEXT_RE.sub(" ", line)
        kind = _metadata_kind(unit_metadata, index)
        for pattern in (ACRONYM_RE, CAPITALIZED_PHRASE_RE, HYPHENATED_TERM_RE):
            for match in pattern.finditer(line):
                candidate = _clean_candidate(match.group(0))
                key = _normalize_key(candidate)
                if key:
                    display.setdefault(key, candidate)
                _add_candidate(
                    candidate,
                    line_index=index,
                    kind=kind,
                    counts=counts,
                    line_hits=line_hits,
                    priority=priority,
                )

        for phrase in _iter_repeated_phrase_candidates(line):
            key = _normalize_key(phrase)
            if not key:
                continue
            display.setdefault(key, phrase)
            phrase_counts[key] += 1
            phrase_line_hits[key].add(index)

    for key, count in phrase_counts.items():
        if count < 2 or len(phrase_line_hits[key]) < 2:
            continue
        phrase = display.get(key, key)
        if not _is_candidate_allowed(phrase):
            continue
        counts[key] += count
        line_hits[key].update(phrase_line_hits[key])

    candidates = []
    for key, count in counts.items():
        distinct_lines = len(line_hits[key])
        score = count + distinct_lines * 2 + priority[key]
        candidate = display.get(key, key)
        word_count = len(candidate.split())
        is_acronym = candidate.upper() == candidate and any(ch.isalpha() for ch in candidate)
        if count < 2 and distinct_lines < 2 and priority[key] < 2:
            continue
        if word_count == 1 and not is_acronym and distinct_lines < 2:
            continue
        candidates.append(
            {
                "source": candidate,
                "score": score,
                "count": count,
                "lines": distinct_lines,
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["source"].lower()))
    return candidates[:limit]


def build_terminology_prompt(
    candidates: List[dict],
    target_language: str,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    strict: bool = False,
    previous_response: Optional[str] = None,
) -> str:
    candidate_payload = [
        {
            "source": candidate["source"],
            "score": candidate["score"],
            "count": candidate["count"],
            "lines": candidate["lines"],
        }
        for candidate in candidates
    ]

    base = (
        f"Prepare a compact terminology memory for translating a book into {target_language}.\n"
        "Choose only useful recurring names, proper nouns, acronyms, and domain terms.\n"
        "Do not invent source terms that are not in the candidates.\n"
        "Keep target translations concise and consistent.\n"
        f"Return at most {memory_limit} total entries.\n\n"
        "Return valid JSON with exactly this shape:\n"
        "{\n"
        '  "terms": [{"source": "...", "target": "..."}],\n'
        '  "proper_names": [{"source": "...", "target": "..."}]\n'
        "}\n\n"
        "Candidate source terms:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False)}"
    )

    if strict:
        retry = (
            "Your previous response was not valid JSON or did not match the required shape. "
            "Return JSON only, with no Markdown fences, no explanation, and no surrounding text."
        )
        if previous_response:
            retry += "\n\nPrevious response:\n" + previous_response[:2000]
        return retry + "\n\n" + base

    return base


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Terminology response did not contain a JSON object.")
    return _json_loads_with_repairs(stripped[start : end + 1])


def _json_loads_with_repairs(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_common_json_issues(text)
        return json.loads(repaired)


def _repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("\ufeff", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*{", "}, {", repaired)
    repaired = re.sub(
        r'("source"\s*:\s*"(?:\\.|[^"\\])*")\s+("target"\s*:)',
        r"\1, \2",
        repaired,
    )
    repaired = re.sub(
        r'("target"\s*:\s*"(?:\\.|[^"\\])*")\s+("source"\s*:)',
        r"\1, \2",
        repaired,
    )
    return repaired


def _parse_object_entries_fallback(text: str, limit: int) -> dict:
    terms = []
    proper_names = []
    seen = set()
    for match in FLAT_JSON_OBJECT_RE.finditer(text):
        raw_object = match.group(0)
        try:
            entry = _json_loads_with_repairs(raw_object)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("source", "")).strip()
        target = str(entry.get("target", "")).strip()
        if not source or not target:
            continue
        key = _normalize_key(source)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized_entry = {"source": source, "target": target}
        section_prefix = text[: match.start()].lower()
        if section_prefix.rfind("proper_names") > section_prefix.rfind("terms"):
            proper_names.append(normalized_entry)
        else:
            terms.append(normalized_entry)
        if len(terms) + len(proper_names) >= limit:
            break
    return {"terms": terms, "proper_names": proper_names}


def _normalize_entries(entries, limit: int) -> List[dict]:
    normalized = []
    if not isinstance(entries, list):
        return normalized

    for entry in entries:
        if isinstance(entry, str):
            source = entry.strip()
            target = ""
        elif isinstance(entry, dict):
            source = str(entry.get("source", "")).strip()
            target = str(entry.get("target", "")).strip()
        else:
            continue

        if not source or not target:
            continue
        normalized.append({"source": source, "target": target})
        if len(normalized) >= limit:
            break
    return normalized


def parse_terminology_memory(text: str, limit: int = DEFAULT_MEMORY_LIMIT) -> dict:
    try:
        payload = _extract_json_object(text)
    except (ValueError, json.JSONDecodeError) as error:
        fallback_memory = _parse_object_entries_fallback(text, limit)
        if fallback_memory["terms"] or fallback_memory["proper_names"]:
            return fallback_memory
        raise error

    if not isinstance(payload, dict):
        fallback_memory = _parse_object_entries_fallback(text, limit)
        if fallback_memory["terms"] or fallback_memory["proper_names"]:
            return fallback_memory
        raise ValueError("Terminology response JSON must be an object.")

    proper_names = _normalize_entries(payload.get("proper_names", []), limit)
    remaining = max(0, limit - len(proper_names))
    terms = _normalize_entries(payload.get("terms", []), remaining)
    return {"terms": terms, "proper_names": proper_names}


def save_terminology_memory(memory: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as terms_file:
        json.dump(memory, terms_file, ensure_ascii=False, indent=2)
        terms_file.write("\n")


def _all_entries(memory: Optional[dict]) -> List[dict]:
    if not memory:
        return []
    entries = []
    for section in ("proper_names", "terms"):
        for entry in memory.get(section, []) or []:
            source = str(entry.get("source", "")).strip()
            target = str(entry.get("target", "")).strip()
            if source and target:
                entries.append({"source": source, "target": target})
    return entries


def select_relevant_terms(
    memory: Optional[dict],
    text: str,
    limit: int = DEFAULT_RELEVANT_LIMIT,
) -> List[dict]:
    haystack = _normalize_key(text or "")
    if not haystack:
        return []

    relevant = []
    seen = set()
    for entry in _all_entries(memory):
        source = entry["source"]
        key = _normalize_key(source)
        if not key or key in seen:
            continue
        if any(variant in haystack for variant in _variant_keys(source)):
            seen.add(key)
            relevant.append(entry)

    relevant.sort(key=lambda item: (-len(item["source"]), item["source"].lower()))
    return relevant[:limit]


def format_terminology_section(entries: List[dict]) -> str:
    if not entries:
        return ""

    lines = [
        "Terminology memory for consistency. Use these entries when they fit the text; do not force them where they do not fit:",
    ]
    for entry in entries:
        lines.append(f"- {entry['source']} => {entry['target']}")
    return "\n".join(lines) + "\n\n"

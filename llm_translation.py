import re
from typing import Callable, Iterable, List, Optional


DEFAULT_LLM_SYSTEM_PROMPT = "You are a professional translator."

DEFAULT_LLM_PROMPT_TEMPLATE = """Translate the following text into fluent and accurate {TARGET_LANGUAGE}.

Requirements:

- Preserve all information.
- Do not summarize.
- Do not explain.
- Do not omit content.
- Preserve names, references, citations, and technical terminology.
- Maintain paragraph structure whenever possible.
- Produce natural and readable {TARGET_LANGUAGE}.
- If the text contains numbered items, return exactly the same item numbers in the same order.

{DOCUMENT_CONTEXT_SECTION}{CONTEXT_SECTION}{TERMINOLOGY_SECTION}Text:

{TEXT}"""

LLM_MODEL_TYPE_HINTS = {
    "baichuan",
    "bloom",
    "chatglm",
    "deepseek",
    "falcon",
    "gemma",
    "gpt2",
    "gpt_bigcode",
    "gptj",
    "gpt_neox",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "qwen2",
    "qwen3",
    "stablelm",
}

TRANSLATION_MODEL_TYPES = {
    "fsmt",
    "m2m_100",
    "marian",
    "mbart",
    "mt5",
    "nllb",
    "seamless_m4t",
    "t5",
}


def is_llm_translation_model(model, tokenizer, model_name: str) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "is_encoder_decoder", False):
        return False

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type in TRANSLATION_MODEL_TYPES:
        return False

    architectures = [
        str(architecture).lower()
        for architecture in getattr(config, "architectures", []) or []
    ]
    model_name = (model_name or "").lower()
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    has_llm_type = model_type in LLM_MODEL_TYPE_HINTS
    has_llm_name = any(
        hint in model_name
        for hint in (
            "instruct",
            "chat",
            "qwen",
            "-it",
        )
    )
    has_llm_architecture = any(
        any(hint in architecture for hint in ("causallm", "qwen", "llama"))
        for architecture in architectures
    )

    return has_chat_template or (has_llm_name and (has_llm_type or has_llm_architecture))


def read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as text_file:
        return [line.rstrip("\n") for line in text_file]


def build_context(lines: List[str], start_index: int, end_index: int, window: int) -> str:
    if window <= 0:
        return ""

    chunks = []
    previous_start = max(0, start_index - window)
    for index in range(previous_start, start_index):
        chunks.append(f"Previous block {index - start_index}: {lines[index]}")

    next_end = min(len(lines), end_index + window + 1)
    for index in range(end_index + 1, next_end):
        chunks.append(f"Next block +{index - end_index}: {lines[index]}")

    return "\n".join(chunks)


def build_numbered_text(texts: Iterable[str]) -> str:
    return "\n".join(f"[{index}] {text}" for index, text in enumerate(texts, start=1))


def split_text_for_llm(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = -1
        separator_length = 0
        for separator in ("\n", ". ", "? ", "! ", "; ", ", ", " "):
            candidate = remaining.rfind(separator, 0, max_chars)
            if candidate >= max_chars // 3:
                split_at = candidate
                separator_length = len(separator.rstrip() or separator)
                break
        if split_at < max_chars // 3:
            split_at = max_chars
            separator_length = 0
        chunk = remaining[: split_at + separator_length].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at + separator_length :].strip()

    if remaining:
        chunks.append(remaining)
    return chunks or [text]


def split_text_for_token_budget(
    text: str,
    fits_text,
    max_chars: int,
) -> List[str]:
    if fits_text(text):
        return [text]

    initial_chunks = split_text_for_llm(text, max_chars)
    split_chunks = []
    for chunk in initial_chunks:
        split_chunks.extend(_split_chunk_for_token_budget(chunk, fits_text))
    return split_chunks


def _split_chunk_for_token_budget(text: str, fits_text) -> List[str]:
    text = text.strip()
    if not text:
        return [text]
    if fits_text(text):
        return [text]
    if len(text) <= 1:
        raise ValueError("Prompt overhead exceeds the LLM input token budget.")

    split_at = _find_split_point(text)
    if split_at <= 0 or split_at >= len(text):
        split_at = len(text) // 2

    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if not left or not right:
        raise ValueError("Prompt overhead exceeds the LLM input token budget.")

    return _split_chunk_for_token_budget(left, fits_text) + _split_chunk_for_token_budget(
        right,
        fits_text,
    )


def _find_split_point(text: str) -> int:
    midpoint = len(text) // 2
    best_position = -1
    best_distance = len(text)
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        search_start = 0
        while True:
            position = text.find(separator, search_start)
            if position == -1:
                break
            candidate = position + len(separator)
            distance = abs(candidate - midpoint)
            if distance < best_distance:
                best_position = candidate
                best_distance = distance
            search_start = position + len(separator)
    return best_position


def build_llm_prompt(
    text: str,
    target_language: str,
    context: str = "",
    prompt_template: Optional[str] = None,
    terminology_section: str = "",
    document_context_section: str = "",
) -> str:
    template = prompt_template or DEFAULT_LLM_PROMPT_TEMPLATE
    if "{TEXT}" not in template:
        raise ValueError("--llm_prompt must include the {TEXT} placeholder.")

    context_section = ""
    if context:
        context_section = (
            "Context for consistency only. Do not translate this context unless it "
            "also appears in the Text section:\n\n"
            f"{context}\n\n"
        )

    return template.format(
        TEXT=text,
        CONTEXT=context,
        CONTEXT_SECTION=context_section,
        TERMINOLOGY_SECTION=terminology_section,
        DOCUMENT_CONTEXT=document_context_section.strip(),
        DOCUMENT_CONTEXT_SECTION=document_context_section,
        TARGET_LANGUAGE=target_language,
    )


def build_chat_messages(prompt_text: str):
    return [
        {"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def render_chat_prompt(tokenizer, prompt_text: str) -> str:
    messages = build_chat_messages(prompt_text)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{DEFAULT_LLM_SYSTEM_PROMPT}\n\n{prompt_text}"


def apply_chat_template_tokenized(tokenizer, prompt_text: str):
    messages = build_chat_messages(prompt_text)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template) or not getattr(tokenizer, "chat_template", None):
        return None

    try:
        return apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        return apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def _input_ids_to_list(tokenized) -> List[int]:
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "squeeze"):
        tokenized = tokenized.squeeze(0)
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, int):
        return [tokenized]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return list(tokenized)


def tokenize_llm_prompt(tokenizer, prompt_text: str) -> List[int]:
    tokenized_prompt = apply_chat_template_tokenized(tokenizer, prompt_text)
    if tokenized_prompt is not None:
        return _input_ids_to_list(tokenized_prompt)

    rendered_prompt = build_plain_prompt(tokenizer, prompt_text)
    tokenized_prompt = tokenizer(
        rendered_prompt,
        truncation=False,
        add_special_tokens=True,
    )
    return _input_ids_to_list(tokenized_prompt)


def estimate_llm_prompt_tokens(tokenizer, prompt_text: str) -> int:
    return len(tokenize_llm_prompt(tokenizer, prompt_text))


def build_plain_prompt(tokenizer, prompt_text: str) -> str:
    return render_chat_prompt(tokenizer, prompt_text)


def make_translation_groups(
    lines: List[str],
    merge_small_blocks: bool,
    merge_max_chars: int,
    unit_metadata: Optional[List[dict]] = None,
    fits_budget: Optional[Callable[[List[int]], bool]] = None,
):
    if not merge_small_blocks:
        return [[index] for index in range(len(lines))]

    groups = []
    current_group = []
    current_chars = 0

    for index, line in enumerate(lines):
        line_chars = len(line)
        candidate_group = current_group + [index]
        metadata_compatible = bool(current_group) and _can_merge_units(
            current_group[-1],
            index,
            unit_metadata,
        )
        if fits_budget is None:
            can_merge = line_chars <= merge_max_chars
            would_fit = current_chars + line_chars <= merge_max_chars
            should_merge = (
                can_merge
                and bool(current_group)
                and would_fit
                and metadata_compatible
            )
        else:
            should_merge = metadata_compatible and fits_budget(candidate_group)

        if should_merge:
            current_group.append(index)
            current_chars += line_chars
            continue

        if current_group:
            groups.append(current_group)

        current_group = [index]
        current_chars = line_chars

    if current_group:
        groups.append(current_group)

    return groups


def _metadata_for_index(unit_metadata: Optional[List[dict]], index: int) -> dict:
    if unit_metadata is None or index >= len(unit_metadata):
        return {}
    return unit_metadata[index] or {}


def _can_merge_units(
    previous_index: int,
    current_index: int,
    unit_metadata: Optional[List[dict]],
) -> bool:
    if unit_metadata is None:
        return True

    previous = _metadata_for_index(unit_metadata, previous_index)
    current = _metadata_for_index(unit_metadata, current_index)
    previous_kind = previous.get("kind", "body")
    current_kind = current.get("kind", "body")
    if previous_kind != current_kind:
        return False
    if current_kind == "heading":
        return False

    for key in ("file_name", "item_id", "section_group"):
        previous_value = previous.get(key)
        current_value = current.get(key)
        if previous_value is not None and current_value is not None:
            if previous_value != current_value:
                return False

    return True


NUMBERED_LINE_PATTERN = re.compile(
    r"^\s*(?:\[?\s*(\d+)\s*\]?[\).:：、-]?)\s*(.+?)\s*$"
)


def parse_numbered_translations(text: str, expected_count: int) -> List[str]:
    translations = {}
    for line in text.splitlines():
        match = NUMBERED_LINE_PATTERN.match(line)
        if not match:
            continue
        number = int(match.group(1))
        value = match.group(2).strip()
        if 1 <= number <= expected_count and value:
            translations[number] = value

    if len(translations) != expected_count:
        raise ValueError(
            f"Expected {expected_count} numbered translations, found {len(translations)}."
        )

    return [translations[index] for index in range(1, expected_count + 1)]

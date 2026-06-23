import re
from typing import Iterable, List, Optional


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

{CONTEXT_SECTION}Text:

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


def build_llm_prompt(
    text: str,
    target_language: str,
    context: str = "",
    prompt_template: Optional[str] = None,
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


def make_translation_groups(
    lines: List[str],
    merge_small_blocks: bool,
    merge_max_chars: int,
):
    if not merge_small_blocks:
        return [[index] for index in range(len(lines))]

    groups = []
    current_group = []
    current_chars = 0

    for index, line in enumerate(lines):
        line_chars = len(line)
        can_merge = line_chars <= merge_max_chars
        would_fit = current_chars + line_chars <= merge_max_chars

        if can_merge and current_group and would_fit:
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

import argparse
import glob
import hashlib
import json
import math
import os
import re
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator, find_executable_batch_size
from tqdm import tqdm

from epub_converter import epub_to_text, require_epub_dependencies, text_to_epub
from llm_translation import (
    apply_translategemma_chat_template_tokenized,
    build_context,
    build_llm_prompt,
    build_numbered_text,
    estimate_llm_prompt_tokens,
    is_llm_translation_model,
    is_translategemma_processor,
    make_translation_groups,
    parse_numbered_translations,
    read_text_lines,
    resolve_translategemma_language_codes,
    split_text_for_token_budget,
    tokenize_llm_prompt,
)
from model import load_model_for_inference
from terminology import (
    build_terminology_prompt,
    collect_term_candidates,
    format_terminology_section,
    parse_terminology_memory,
    save_terminology_memory,
    select_relevant_terms,
)


def encode_string(text):
    return text.replace("\r", r"\r").replace("\n", r"\n").replace("\t", r"\t")


PROMPT_LEAK_MARKERS = (
    "requirements:",
    "preserve all information",
    "do not summarize",
    "do not explain",
    "do not omit content",
    "document context for consistency only",
    "context for consistency only",
    "terminology memory for consistency",
    "locked terminology for consistency",
    "保留所有信息",
    "不要总结",
    "不要解释",
    "不要遗漏内容",
    "仅用于保持一致性的文档背景",
    "仅用于保持一致性的上下文",
    "用于保持一致性的术语记忆",
    "锁定术语",
    "块类型：",
    "块类型:",
)


def _translation_contains_prompt_leak(text: str) -> bool:
    probe = (text or "").replace(r"\n", "\n").replace(r"\r", "\n")
    lower_probe = probe.lower()
    return any(marker in lower_probe for marker in PROMPT_LEAK_MARKERS)


def clean_translation_output(text: str) -> str:
    """Remove accidental prompt/control text echoed by chat models."""
    text = (text or "").strip()
    if not text or not _translation_contains_prompt_leak(text):
        return text

    working = (
        text.replace(r"\r", "\n")
        .replace(r"\n", "\n")
        .replace(r"\t", " ")
    )
    cleaned_lines = []
    skip_section = None

    for raw_line in working.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        normalized_header = lower_line.rstrip(":：")

        if lower_line.startswith("translate the following text into") or lower_line.startswith(
            "将以下文本翻译成"
        ):
            skip_section = None
            continue

        if normalized_header in {"requirements", "要求"}:
            skip_section = "requirements"
            continue

        if (
            "document context for consistency only" in lower_line
            or "仅用于保持一致性的文档背景" in line
        ):
            skip_section = "document_context"
            continue

        if (
            "context for consistency only" in lower_line
            or "仅用于保持一致性的上下文" in line
        ):
            skip_section = "context"
            continue

        if (
            "terminology memory for consistency" in lower_line
            or "locked terminology for consistency" in lower_line
            or "用于保持一致性的术语记忆" in line
            or "锁定术语" in line
        ):
            skip_section = "terminology"
            continue

        if normalized_header in {"text", "文本", "translation", "译文"}:
            skip_section = None
            continue

        if skip_section == "requirements":
            if line.startswith(("-", "－", "—", "*", "•")) or any(
                phrase in lower_line
                for phrase in (
                    "preserve all information",
                    "do not summarize",
                    "do not explain",
                    "do not omit content",
                    "preserve names",
                    "maintain paragraph",
                    "produce natural",
                    "numbered items",
                )
            ) or any(
                phrase in line
                for phrase in (
                    "保留所有信息",
                    "不要总结",
                    "不要解释",
                    "不要遗漏内容",
                    "保留名称",
                    "保留姓名",
                    "保持段落",
                    "自然流畅",
                    "编号项目",
                    "相同的项目编号",
                )
            ):
                continue
            skip_section = None

        if skip_section == "document_context":
            if (
                lower_line.startswith(("book:", "chapter:", "epub item:", "block kind:"))
                or line.startswith(("书名：", "书名:", "章节：", "章节:", "EPUB 项目：", "EPUB项目：", "块类型：", "块类型:"))
            ):
                continue
            skip_section = None

        if skip_section == "context":
            if lower_line.startswith(("previous block", "next block")) or line.startswith(
                ("前一块", "后一块", "上一段", "下一段")
            ):
                continue
            skip_section = None

        if skip_section == "terminology":
            if line.startswith(("-", "－", "—", "*", "•")) or "=>" in line or "→" in line:
                continue
            skip_section = None

        if any(
            phrase in lower_line
            for phrase in (
                "preserve all information",
                "do not summarize",
                "do not explain",
                "do not omit content",
                "document context for consistency only",
                "context for consistency only",
                "terminology memory for consistency",
                "locked terminology for consistency",
            )
        ) or any(
            phrase in line
            for phrase in (
                "保留所有信息",
                "不要总结",
                "不要解释",
                "不要遗漏内容",
                "仅用于保持一致性的文档背景",
                "仅用于保持一致性的上下文",
                "用于保持一致性的术语记忆",
                "锁定术语",
            )
        ):
            continue

        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_epub_path(path: Optional[str]) -> bool:
    return path is not None and os.path.splitext(path)[1].lower() == ".epub"


def directory_may_include_epubs(directory: Optional[str], files_extension: str) -> bool:
    if directory is None:
        return False
    files_extension = files_extension or ""
    if files_extension.lower() == "epub":
        return True
    if files_extension:
        return False
    return any(is_epub_path(path) for path in glob.glob(os.path.join(directory, "*")))


def get_epub_work_paths(input_path: str, output_path: str, work_dir_root: Optional[str] = None):
    if work_dir_root:
        output_name = os.path.basename(os.path.abspath(output_path))
        work_dir = os.path.join(
            os.path.abspath(work_dir_root),
            output_name + ".easytranslate_epub",
        )
    else:
        work_dir = os.path.abspath(output_path) + ".easytranslate_epub"
    return {
        "work_dir": work_dir,
        "source_text": os.path.join(work_dir, "source.txt"),
        "translated_text": os.path.join(work_dir, "translated.txt"),
        "partial_text": os.path.join(work_dir, "translated.partial.jsonl"),
        "partial_meta": os.path.join(work_dir, "translated.partial.meta.json"),
        "manifest": os.path.join(work_dir, "manifest.json"),
        "terms": os.path.join(work_dir, "terms.json"),
        "input_epub": input_path,
        "output_path": output_path,
    }


def get_terms_path(output_path: str) -> str:
    return os.path.abspath(output_path) + ".easytranslate_terms.json"


def get_partial_paths(output_path: str):
    absolute_output_path = os.path.abspath(output_path)
    return {
        "partial_text": absolute_output_path + ".easytranslate_partial.jsonl",
        "partial_meta": absolute_output_path + ".easytranslate_partial.meta.json",
    }


def source_lines_sha256(source_lines: List[str]) -> str:
    digest = hashlib.sha256()
    for line in source_lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_resume_meta(source_lines: List[str], settings: Dict) -> Dict:
    return {
        "version": 1,
        "source_sha256": source_lines_sha256(source_lines),
        "line_count": len(source_lines),
        "settings": settings,
    }


def load_partial_translations(
    partial_text_path: Optional[str],
    partial_meta_path: Optional[str],
    expected_meta: Dict,
    total_lines: int,
) -> List[Optional[str]]:
    translations: List[Optional[str]] = [None] * total_lines
    if not partial_text_path or not partial_meta_path:
        return translations
    if not os.path.exists(partial_text_path) or not os.path.exists(partial_meta_path):
        return translations

    def clear_partial_text() -> None:
        try:
            if os.path.exists(partial_text_path):
                os.remove(partial_text_path)
        except OSError:
            pass

    try:
        with open(partial_meta_path, "r", encoding="utf-8") as meta_file:
            existing_meta = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        print("WARNING: Could not read resume metadata. Ignoring partial translations.")
        clear_partial_text()
        return translations

    if existing_meta != expected_meta:
        print("Resume metadata does not match current input/settings. Ignoring partial translations.")
        clear_partial_text()
        return translations

    loaded = 0
    try:
        with open(partial_text_path, "r", encoding="utf-8") as partial_file:
            for line in partial_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                index = record.get("index")
                text = record.get("text")
                if isinstance(index, int) and 0 <= index < total_lines and isinstance(text, str):
                    text = clean_translation_output(text)
                    if not text:
                        continue
                    if translations[index] is None:
                        loaded += 1
                    translations[index] = text
    except OSError:
        print("WARNING: Could not read partial translations. Ignoring resume cache.")
        clear_partial_text()
        return [None] * total_lines

    if loaded:
        print(f"Loaded {loaded}/{total_lines} completed translations from resume cache.")
    return translations


def write_partial_meta(partial_meta_path: Optional[str], resume_meta: Dict) -> None:
    if not partial_meta_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(partial_meta_path)), exist_ok=True)
    with open(partial_meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(resume_meta, meta_file, ensure_ascii=False, indent=2, sort_keys=True)


def append_partial_translation(partial_file, index: int, text: str) -> None:
    print(
        json.dumps({"index": index, "text": text}, ensure_ascii=False),
        file=partial_file,
    )
    partial_file.flush()


def get_epub_language_code(llm_target_language: str) -> str:
    if llm_target_language and "chinese" in llm_target_language.lower():
        if "traditional" in llm_target_language.lower():
            return "zh-Hant"
        return "zh-Hans"
    return "zh-Hans"


def _fill_unit_metadata(
    unit_metadata: List[dict],
    unit: dict,
    *,
    kind: str,
    file_name: Optional[str],
    item_id: Optional[str],
    section_group: int,
    book_title: Optional[str] = None,
    chapter_heading: Optional[str] = None,
) -> None:
    line_start = unit.get("line")
    if line_start is None:
        return
    line_count = unit.get("lines", 1)
    for line in range(line_start, min(line_start + line_count, len(unit_metadata))):
        unit_metadata[line] = {
            "line": line,
            "kind": kind,
            "file_name": file_name,
            "item_id": item_id,
            "section_group": section_group,
            "original_text": unit.get("original_text"),
            "tag": unit.get("tag"),
            "book_title": book_title,
            "chapter_heading": chapter_heading,
        }


def load_epub_unit_metadata(manifest_path: Optional[str], total_lines: int):
    if manifest_path is None or not os.path.exists(manifest_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    unit_metadata = [{} for _ in range(total_lines)]
    section_group = 0
    book_title = None
    for unit in manifest.get("metadata", []):
        if unit.get("kind") == "opf_title" and unit.get("original_text"):
            book_title = unit.get("original_text")
            break

    for unit in manifest.get("metadata", []):
        _fill_unit_metadata(
            unit_metadata,
            unit,
            kind=unit.get("unit_kind", "metadata"),
            file_name=unit.get("zip_name"),
            item_id=None,
            section_group=section_group,
            book_title=book_title,
        )

    for manifest_item in manifest.get("items", []):
        section_group += 1
        file_name = manifest_item.get("file_name")
        item_id = manifest_item.get("item_id")
        current_heading = None
        for block in manifest_item.get("blocks", []):
            kind = block.get("kind", "body")
            if kind == "heading" and block.get("original_text"):
                current_heading = block.get("original_text")
            _fill_unit_metadata(
                unit_metadata,
                block,
                kind=kind,
                file_name=file_name,
                item_id=item_id,
                section_group=section_group,
                book_title=book_title,
                chapter_heading=current_heading,
            )
            if kind == "heading":
                section_group += 1

    return unit_metadata


def generate_terminology_memory(
    source_lines: List[str],
    unit_metadata: Optional[List[dict]],
    target_language: str,
    terms_path: str,
    generate_prompts,
):
    candidates = collect_term_candidates(source_lines, unit_metadata)
    if len(candidates) < 3:
        return None

    prompt = build_terminology_prompt(candidates, target_language=target_language)
    first_response = generate_prompts([prompt])[0]
    try:
        memory = parse_terminology_memory(first_response)
    except (ValueError, json.JSONDecodeError) as first_error:
        retry_prompt = build_terminology_prompt(
            candidates,
            target_language=target_language,
            strict=True,
            previous_response=first_response,
        )
        retry_response = generate_prompts([retry_prompt])[0]
        try:
            memory = parse_terminology_memory(retry_response)
        except (ValueError, json.JSONDecodeError) as retry_error:
            print(
                "WARNING: Automatic terminology generation failed after one retry. "
                f"Continuing without terminology memory. First error: {first_error}. "
                f"Retry error: {retry_error}"
            )
            return None

    if not memory.get("terms") and not memory.get("proper_names"):
        print("WARNING: Automatic terminology generation returned no usable entries.")
        return None

    save_terminology_memory(memory, terms_path)
    print(f"Automatic terminology memory written to {terms_path}")
    return memory


def main(
    sentences_path: Optional[str],
    sentences_dir: Optional[str],
    files_extension: str,
    output_path: str,
    source_lang: Optional[str],
    target_lang: Optional[str],
    starting_batch_size: Optional[int] = None,
    model_name: str = "Qwen/Qwen3-14B-Instruct",
    lora_weights_name_or_path: str = None,
    force_auto_device_map: bool = False,
    precision: str = None,
    max_length: Optional[int] = None,
    num_beams: Optional[int] = None,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    keep_special_tokens: bool = False,
    keep_tokenization_spaces: bool = False,
    repetition_penalty: float = None,
    prompt: str = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    source_lang_code: str = "en",
    target_lang_code: str = "zh-CN",
    llm_target_language: str = "Simplified Chinese",
    llm_prompt: str = None,
    context_window: int = 0,
    merge_small_blocks: bool = False,
    merge_max_chars: int = 1200,
    llm_input_max_length: int = 8192,
    llm_chunk_chars: int = 3000,
    disable_auto_terms: bool = False,
    disable_resume: bool = False,
    work_dir: Optional[str] = None,
):
    accelerator = Accelerator()

    if sentences_path is None and sentences_dir is None:
        raise ValueError(
            "You must specify either --sentences_path or --sentences_dir. Use --help for more details."
        )

    if sentences_path is not None and sentences_dir is not None:
        raise ValueError(
            "You must specify either --sentences_path or --sentences_dir, not both. Use --help for more details."
        )

    has_epub_input = is_epub_path(sentences_path) or directory_may_include_epubs(
        sentences_dir, files_extension
    )
    if has_epub_input:
        require_epub_dependencies()
        if num_return_sequences != 1:
            raise ValueError(
                "EPUB input requires --num_return_sequences 1 so each extracted "
                "book block maps to exactly one translated line."
            )

    if precision is None:
        quantization = None
        dtype = None
    elif precision == "8" or precision == "4":
        quantization = int(precision)
        dtype = None
    elif precision == "fp16":
        quantization = None
        dtype = "float16"
    elif precision == "bf16":
        quantization = None
        dtype = "bfloat16"
    elif precision == "32":
        quantization = None
        dtype = "float32"
    else:
        raise ValueError(
            f"Precision {precision} not supported. Please choose between 8, 4, fp16, bf16, 32 or None."
        )

    model, tokenizer = load_model_for_inference(
        weights_path=model_name,
        quantization=quantization,
        lora_weights_name_or_path=lora_weights_name_or_path,
        torch_dtype=dtype,
        force_auto_device_map=force_auto_device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    use_translategemma = is_translategemma_processor(tokenizer)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    if not is_llm_translation_model(model, tokenizer, model_name):
        raise ValueError(
            "This fork/version of Easy-Translate is LLM-only and expects an "
            "instruction-tuned CausalLM/chat model. The loaded model does not look "
            "like a supported chat/instruction translation model. Use a model such "
            "as Qwen3-14B-Instruct, or provide a tokenizer with a chat template."
        )

    if max_length is None:
        max_length = 2048
    if num_beams is None:
        num_beams = 1
    if starting_batch_size is None:
        starting_batch_size = 1

    if starting_batch_size <= 0:
        raise ValueError("--starting_batch_size must be greater than 0.")
    if max_length <= 0:
        raise ValueError("--max_length must be greater than 0.")
    if num_beams <= 0:
        raise ValueError("--num_beams must be greater than 0.")

    if force_auto_device_map and starting_batch_size >= 64:
        print(
            f"WARNING: You are using a very large batch size ({starting_batch_size}) and the auto_device_map  flag. "
            f"auto_device_map will offload model parameters to the CPU when they don't fit on the GPU VRAM. "
            f"If you use a very large batch size, it will offload a lot of parameters to the CPU and slow down the "
            f"inference. You should consider using a smaller batch size, i.e '--starting_batch_size 8'"
        )

    if num_return_sequences != 1:
        raise ValueError("LLM translation mode requires --num_return_sequences 1.")

    if source_lang is not None or target_lang is not None:
        print(
            "WARNING: --source_lang and --target_lang are ignored in LLM-only mode. "
            "Use --llm_target_language to choose the translation target."
        )

    if prompt is not None:
        raise ValueError(
            "--prompt and the legacy %%SENTENCE%% prompting path are no longer "
            "supported in LLM-only mode. Use --llm_prompt with a {TEXT} placeholder instead."
        )
    if use_translategemma and llm_prompt is not None:
        raise ValueError(
            "TranslateGemma uses its own language-code chat template and does not "
            "support --llm_prompt. Remove --llm_prompt and use "
            "--source_lang_code/--target_lang_code instead."
        )
    if llm_prompt is not None and "{TEXT}" not in llm_prompt:
        raise ValueError("The --llm_prompt argument must include the {TEXT} placeholder.")
    requested_source_lang_code = source_lang_code
    requested_target_lang_code = target_lang_code
    if use_translategemma:
        source_lang_code, target_lang_code = resolve_translategemma_language_codes(
            tokenizer,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        )
        if (
            source_lang_code != requested_source_lang_code
            or target_lang_code != requested_target_lang_code
        ):
            print(
                "WARNING: TranslateGemma language codes were normalized from "
                f"{requested_source_lang_code}->{requested_target_lang_code} to "
                f"{source_lang_code}->{target_lang_code} for this model's chat template."
            )
    if context_window < 0:
        raise ValueError("--context_window must be greater than or equal to 0.")
    if merge_max_chars <= 0:
        raise ValueError("--merge_max_chars must be greater than 0.")
    if llm_input_max_length <= 0:
        raise ValueError("--llm_input_max_length must be greater than 0.")
    if llm_chunk_chars <= 0:
        raise ValueError("--llm_chunk_chars must be greater than 0.")

    gen_kwargs = {
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
    }

    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    stop_token_ids = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        convert_tokens_to_ids = getattr(text_tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert_tokens_to_ids):
            continue
        token_id = convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(text_tokenizer, "unk_token_id", None):
            stop_token_ids.append(token_id)
    if stop_token_ids:
        gen_kwargs["eos_token_id"] = stop_token_ids
    if getattr(text_tokenizer, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = text_tokenizer.pad_token_id

    if accelerator.is_main_process:
        print(
            f"** Translation **\n"
            f"Input file: {sentences_path}\n"
            f"Sentences dir: {sentences_dir}\n"
            f"Output file: {output_path}\n"
            f"EPUB work dir: {work_dir}\n"
            f"Deprecated source_lang argument: {source_lang}\n"
            f"Deprecated target_lang argument: {target_lang}\n"
            f"LLM-only translation mode: True\n"
            f"LLM target language: {llm_target_language}\n"
            f"TranslateGemma mode: {use_translategemma}\n"
            f"Source language code: {source_lang_code}\n"
            f"Target language code: {target_lang_code}\n"
            f"Automatic terminology memory: {not disable_auto_terms}\n"
            f"Resume partial translations: {not disable_resume}\n"
            f"Context window: {context_window}\n"
            f"Merge small blocks: {merge_small_blocks}\n"
            f"LLM input max length: {llm_input_max_length}\n"
            f"LLM chunk chars: {llm_chunk_chars}\n"
            f"Prompt: {prompt}\n"
            f"Starting batch size: {starting_batch_size}\n"
            f"Device: {str(accelerator.device).split(':')[0]}\n"
            f"Num. Devices: {accelerator.num_processes}\n"
            f"Distributed_type: {accelerator.distributed_type}\n"
            f"Max length: {max_length}\n"
            f"Quantization: {quantization}\n"
            f"Precision: {dtype}\n"
            f"Model: {model_name}\n"
            f"LoRA weights: {lora_weights_name_or_path}\n"
            f"Force auto device map: {force_auto_device_map}\n"
            f"Attention implementation: {attn_implementation}\n"
            f"Keep special tokens: {keep_special_tokens}\n"
            f"Keep tokenization spaces: {keep_tokenization_spaces}\n"
        )
        print("** Generation parameters **")
        print(f"max_new_tokens: dynamic <= {max_length}")
        print("\n".join(f"{k}: {v}" for k, v in gen_kwargs.items()))
        print("\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def llm_inference(
        batch_size,
        sentences_path,
        output_path,
        manifest_path=None,
        terms_path=None,
        partial_text_path=None,
        partial_meta_path=None,
    ):
        nonlocal model, tokenizer, max_length, gen_kwargs

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return

        print(f"Translating {sentences_path} with LLM batch size {batch_size}")
        source_lines = read_text_lines(sentences_path)
        unit_metadata = load_epub_unit_metadata(manifest_path, len(source_lines))
        resume_settings = {
            "model_family": "translategemma" if use_translategemma else "chat_causallm",
            "model_name": model_name,
            "source_lang_code": source_lang_code,
            "target_lang_code": target_lang_code,
            "llm_target_language": llm_target_language,
            "llm_prompt": llm_prompt,
            "context_window": context_window,
            "merge_small_blocks": merge_small_blocks,
            "merge_max_chars": merge_max_chars,
            "llm_input_max_length": llm_input_max_length,
            "llm_chunk_chars": llm_chunk_chars,
            "disable_auto_terms": disable_auto_terms,
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "keep_special_tokens": keep_special_tokens,
            "keep_tokenization_spaces": keep_tokenization_spaces,
            "terminology_consistency_version": 2,
        }
        resume_meta = build_resume_meta(source_lines, resume_settings)
        if disable_resume:
            translations = [None] * len(source_lines)
        else:
            translations = load_partial_translations(
                partial_text_path,
                partial_meta_path,
                resume_meta,
                len(source_lines),
            )
            write_partial_meta(partial_meta_path, resume_meta)
        completed_count = sum(translation is not None for translation in translations)
        if completed_count == len(source_lines):
            print("All translations loaded from resume cache.")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as output_file:
                for translated_text in translations:
                    print(
                        encode_string(clean_translation_output(translated_text or "")),
                        file=output_file,
                    )
            accelerator.wait_for_everyone()
            print(f"Translation done. Output written to {output_path}\n")
            return
        prepared_model = accelerator.prepare(model)

        def input_ids_to_list(tokenized):
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

        def build_llm_batch(prompt_texts, enforce_input_budget=True):
            if use_translategemma:
                encoded_prompts = [
                    input_ids_to_list(
                        apply_translategemma_chat_template_tokenized(
                            tokenizer,
                            prompt_text,
                            source_lang_code=source_lang_code,
                            target_lang_code=target_lang_code,
                        )
                    )
                    for prompt_text in prompt_texts
                ]
            else:
                encoded_prompts = [
                    tokenize_llm_prompt(tokenizer, prompt_text)
                    for prompt_text in prompt_texts
                ]
            longest_prompt = max((len(prompt) for prompt in encoded_prompts), default=0)
            if enforce_input_budget and longest_prompt > llm_input_max_length:
                raise ValueError(
                    f"LLM prompt has {longest_prompt} input tokens, which exceeds "
                    f"--llm_input_max_length {llm_input_max_length}. Increase "
                    "--llm_input_max_length, reduce --context_window, or disable auto terms."
                )
            return text_tokenizer.pad(
                {"input_ids": encoded_prompts},
                padding=True,
                return_tensors="pt",
            )

        def source_token_count(source_text):
            tokenized = text_tokenizer(
                source_text,
                add_special_tokens=False,
                truncation=False,
            )
            input_ids = tokenized["input_ids"]
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if isinstance(input_ids, int):
                return 1
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return len(input_ids)

        def prompt_token_count(prompt_text):
            if use_translategemma:
                return len(
                    input_ids_to_list(
                        apply_translategemma_chat_template_tokenized(
                            tokenizer,
                            prompt_text,
                            source_lang_code=source_lang_code,
                            target_lang_code=target_lang_code,
                        )
                    )
                )
            return estimate_llm_prompt_tokens(tokenizer, prompt_text)

        def dynamic_max_new_tokens(source_text, group_size=1):
            source_tokens = source_token_count(source_text)
            minimum = min(64, max_length)
            budget = math.ceil(source_tokens * 1.6) + 32 + 8 * group_size
            return min(max_length, max(minimum, budget))

        def generate_prompts(
            prompt_texts,
            enforce_input_budget=True,
            max_new_tokens=None,
        ):
            batch = build_llm_batch(
                prompt_texts,
                enforce_input_budget=enforce_input_budget,
            )
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            generation_kwargs = dict(gen_kwargs)
            generation_kwargs["max_new_tokens"] = max_new_tokens or max_length
            generated_tokens = accelerator.unwrap_model(prepared_model).generate(
                **batch,
                **generation_kwargs,
            )
            generated_tokens = generated_tokens[:, batch["input_ids"].shape[1] :]
            decoder = text_tokenizer
            return decoder.batch_decode(
                generated_tokens,
                skip_special_tokens=not keep_special_tokens,
                clean_up_tokenization_spaces=not keep_tokenization_spaces,
            )

        terminology_memory = None
        if use_translategemma and not disable_auto_terms:
            print(
                "WARNING: Automatic terminology memory is skipped for TranslateGemma "
                "because its official template is translation-specific."
            )
        elif not disable_auto_terms and source_lines:
            terminology_memory = generate_terminology_memory(
                source_lines=source_lines,
                unit_metadata=unit_metadata,
                target_language=llm_target_language,
                terms_path=terms_path or get_terms_path(output_path),
                generate_prompts=lambda prompts: generate_prompts(
                    prompts,
                    enforce_input_budget=False,
                ),
            )

        def terminology_section_for(text, context):
            if not terminology_memory:
                return ""
            relevant_entries = select_relevant_terms(
                terminology_memory,
                f"{text}\n{context}",
            )
            return format_terminology_section(relevant_entries)

        def _compact_context_value(value, limit=120):
            value = " ".join(str(value or "").split())
            if len(value) <= limit:
                return value
            return value[: limit - 1].rstrip() + "..."

        def document_context_section_for(index):
            if unit_metadata is None or index >= len(unit_metadata):
                return ""
            metadata = unit_metadata[index] or {}
            lines = []
            book_title = _compact_context_value(metadata.get("book_title"))
            chapter_heading = _compact_context_value(metadata.get("chapter_heading"))
            kind = metadata.get("kind")
            file_name = _compact_context_value(metadata.get("file_name"), limit=80)

            if book_title:
                lines.append(f"Book: {book_title}")
            if chapter_heading and chapter_heading != book_title:
                lines.append(f"Chapter: {chapter_heading}")
            elif file_name:
                lines.append(f"EPUB item: {file_name}")
            if kind:
                lines.append(f"Block kind: {kind}")
            if not lines:
                return ""
            return "Document context for consistency only:\n" + "\n".join(lines) + "\n\n"

        def build_translategemma_text(text, context, terminology_section, document_context_section):
            return text

        def build_group_text(group):
            return build_numbered_text(source_lines[index] for index in group)

        def build_prompt_for_group(group, text=None):
            context = build_context(
                source_lines,
                start_index=group[0],
                end_index=group[-1],
                window=context_window,
            )
            if text is None:
                text = build_group_text(group)
            terminology_section = terminology_section_for(text, context)
            document_context_section = document_context_section_for(group[0])
            if use_translategemma:
                return build_translategemma_text(
                    text,
                    context=context,
                    terminology_section=terminology_section,
                    document_context_section=document_context_section,
                )
            return build_llm_prompt(
                text=text,
                target_language=llm_target_language,
                context=context,
                prompt_template=llm_prompt,
                terminology_section=terminology_section,
                document_context_section=document_context_section,
            )

        def build_prompt_for_text(
            text,
            *,
            context,
            document_context_section,
        ):
            terminology_section = terminology_section_for(text, context)
            if use_translategemma:
                return build_translategemma_text(
                    text,
                    context=context,
                    terminology_section=terminology_section,
                    document_context_section=document_context_section,
                )
            return build_llm_prompt(
                text=text,
                target_language=llm_target_language,
                context=context,
                prompt_template=llm_prompt,
                terminology_section=terminology_section,
                document_context_section=document_context_section,
            )

        def prompt_fits_budget(prompt_text):
            return prompt_token_count(prompt_text) <= llm_input_max_length

        def group_fits_budget(group):
            return prompt_fits_budget(build_prompt_for_group(group))

        groups = make_translation_groups(
            source_lines,
            merge_small_blocks=merge_small_blocks,
            merge_max_chars=merge_max_chars,
            unit_metadata=unit_metadata,
            fits_budget=group_fits_budget,
        )
        resumable_groups = []
        for group in groups:
            missing_group = [index for index in group if translations[index] is None]
            if not missing_group:
                continue
            if len(missing_group) == len(group):
                resumable_groups.append(group)
            else:
                resumable_groups.extend([index] for index in missing_group)

        def build_group_record(group):
            source_text = build_group_text(group)
            prompt_text = build_prompt_for_group(group, source_text)
            token_count = prompt_token_count(prompt_text)
            if token_count > llm_input_max_length and len(group) > 1:
                raise ValueError(
                    f"Merged LLM prompt for lines {group[0] + 1}-{group[-1] + 1} "
                    f"has {token_count} input tokens, which exceeds "
                    f"--llm_input_max_length {llm_input_max_length}."
                )
            return {
                "group": group,
                "prompt": prompt_text if token_count <= llm_input_max_length else None,
                "token_count": token_count,
                "source_tokens": source_token_count(source_text),
                "max_new_tokens": dynamic_max_new_tokens(
                    source_text,
                    group_size=len(group),
                ),
            }

        group_records = [build_group_record(group) for group in resumable_groups]
        group_records = sorted(
            enumerate(group_records),
            key=lambda item: (item[1]["token_count"], item[0]),
        )
        group_records = [record for _, record in group_records]

        def translate_single_line(index):
            context = build_context(
                source_lines,
                start_index=index,
                end_index=index,
                window=context_window,
            )
            full_text = build_numbered_text([source_lines[index]])
            doc_context = document_context_section_for(index)
            full_prompt = build_prompt_for_text(
                full_text,
                context=context,
                document_context_section=doc_context,
            )
            if prompt_fits_budget(full_prompt):
                decoded_output = generate_prompts(
                    [full_prompt],
                    max_new_tokens=dynamic_max_new_tokens(full_text, group_size=1),
                )[0]
                try:
                    return parse_numbered_translations(
                        decoded_output,
                        expected_count=1,
                    )[0].strip()
                except ValueError:
                    return decoded_output.strip()

            def chunk_fits_budget(chunk):
                chunk_text = build_numbered_text([chunk])
                chunk_prompt = build_prompt_for_text(
                    chunk_text,
                    context=context,
                    document_context_section=doc_context,
                )
                return prompt_fits_budget(chunk_prompt)

            try:
                chunks = split_text_for_token_budget(
                    source_lines[index],
                    fits_text=chunk_fits_budget,
                    max_chars=llm_chunk_chars,
                )
            except ValueError as error:
                raise ValueError(
                    f"Source line {index + 1} cannot fit in the LLM input token "
                    f"budget even after splitting: {error} Increase "
                    "--llm_input_max_length, reduce --context_window, or disable auto terms."
                ) from error

            def build_chunk_group_prompt(chunk_group):
                text = build_numbered_text(
                    chunks[chunk_index] for chunk_index in chunk_group
                )
                return build_prompt_for_text(
                    text,
                    context=context,
                    document_context_section=doc_context,
                )

            chunk_groups = make_translation_groups(
                chunks,
                merge_small_blocks=True,
                merge_max_chars=llm_chunk_chars,
                fits_budget=lambda group: prompt_fits_budget(
                    build_chunk_group_prompt(group)
                ),
            )
            translated_chunks = []
            for chunk_group in chunk_groups:
                chunk_text = build_numbered_text(
                    chunks[chunk_index] for chunk_index in chunk_group
                )
                prompt_text = build_chunk_group_prompt(chunk_group)
                decoded_output = generate_prompts(
                    [prompt_text],
                    max_new_tokens=dynamic_max_new_tokens(
                        chunk_text,
                        group_size=len(chunk_group),
                    ),
                )[0]
                try:
                    parsed_chunks = parse_numbered_translations(
                        decoded_output,
                        expected_count=len(chunk_group),
                    )
                except ValueError:
                    parsed_chunks = []
                    for chunk_index in chunk_group:
                        chunk = chunks[chunk_index]
                        chunk_text = build_numbered_text([chunk])
                        chunk_prompt = build_prompt_for_text(
                            chunk_text,
                            context="",
                            document_context_section=doc_context,
                        )
                        parsed_chunks.append(
                            generate_prompts(
                                [chunk_prompt],
                                max_new_tokens=dynamic_max_new_tokens(
                                    chunk_text,
                                    group_size=1,
                                ),
                            )[0].strip()
                        )
                translated_chunks.extend(parsed_chunks)
            return " ".join(chunk.strip() for chunk in translated_chunks if chunk.strip())

        partial_file = None
        if not disable_resume and partial_text_path:
            os.makedirs(os.path.dirname(os.path.abspath(partial_text_path)), exist_ok=True)
            partial_file = open(partial_text_path, "a", encoding="utf-8")

        def record_translation(index: int, text: str) -> None:
            text = clean_translation_output(text)
            translations[index] = text
            if partial_file is not None:
                append_partial_translation(partial_file, index, text)

        try:
            with tqdm(
                total=len(source_lines),
                initial=completed_count,
                desc="LLM translation",
                leave=True,
                ascii=True,
            ) as pbar:
                with torch.no_grad():
                    for group_start in range(0, len(group_records), batch_size):
                        batch_records = group_records[group_start : group_start + batch_size]
                        ready_records = [
                            record
                            for record in batch_records
                            if record["prompt"] is not None
                        ]
                        if ready_records:
                            batch_max_new_tokens = max(
                                record["max_new_tokens"] for record in ready_records
                            )
                            decoded_outputs = generate_prompts(
                                [record["prompt"] for record in ready_records],
                                max_new_tokens=batch_max_new_tokens,
                            )
                        else:
                            decoded_outputs = []

                        decoded_by_group = {
                            tuple(record["group"]): decoded_output
                            for record, decoded_output in zip(
                                ready_records,
                                decoded_outputs,
                            )
                        }

                        for record in batch_records:
                            group = record["group"]
                            decoded_output = decoded_by_group.get(tuple(group))
                            if decoded_output is None:
                                record_translation(
                                    group[0],
                                    translate_single_line(group[0]),
                                )
                                pbar.update(1)
                                continue

                            if len(group) == 1:
                                try:
                                    parsed_outputs = parse_numbered_translations(
                                        decoded_output,
                                        expected_count=1,
                                    )
                                    record_translation(
                                        group[0],
                                        parsed_outputs[0],
                                    )
                                except ValueError:
                                    record_translation(
                                        group[0],
                                        translate_single_line(group[0]),
                                    )
                                pbar.update(1)
                                continue

                            try:
                                parsed_outputs = parse_numbered_translations(
                                    decoded_output,
                                    expected_count=len(group),
                                )
                            except ValueError:
                                parsed_outputs = [
                                    translate_single_line(index) for index in group
                                ]

                            for index, translated_text in zip(group, parsed_outputs):
                                record_translation(index, translated_text)
                            pbar.update(len(group))
        finally:
            if partial_file is not None:
                partial_file.close()

        missing_indexes = [
            index for index, translated_text in enumerate(translations)
            if translated_text is None
        ]
        if missing_indexes:
            raise RuntimeError(
                "Translation finished with missing cached lines: "
                + ", ".join(str(index + 1) for index in missing_indexes[:10])
            )

        with open(output_path, "w", encoding="utf-8") as output_file:
            for translated_text in translations:
                print(
                    encode_string(clean_translation_output(translated_text or "")),
                    file=output_file,
                )

        accelerator.wait_for_everyone()
        print(f"Translation done. Output written to {output_path}\n")

    def translate_file(input_path, final_output_path):
        epub_work_paths = None
        translation_manifest_path = None
        translation_input_path = input_path
        translation_output_path = final_output_path
        terms_path = get_terms_path(final_output_path)
        partial_paths = get_partial_paths(final_output_path)

        if is_epub_path(input_path):
            epub_work_paths = get_epub_work_paths(input_path, final_output_path, work_dir)
            translation_input_path = epub_work_paths["source_text"]
            translation_manifest_path = epub_work_paths["manifest"]
            terms_path = epub_work_paths["terms"]
            partial_paths = {
                "partial_text": epub_work_paths["partial_text"],
                "partial_meta": epub_work_paths["partial_meta"],
            }
            if is_epub_path(final_output_path):
                translation_output_path = epub_work_paths["translated_text"]

            if accelerator.is_main_process:
                os.makedirs(epub_work_paths["work_dir"], exist_ok=True)
                epub_to_text(
                    epub_path=input_path,
                    text_path=epub_work_paths["source_text"],
                    manifest_path=epub_work_paths["manifest"],
                )

            accelerator.wait_for_everyone()

        os.makedirs(os.path.abspath(os.path.dirname(final_output_path)), exist_ok=True)
        if disable_resume and accelerator.is_main_process:
            for partial_path in partial_paths.values():
                if partial_path and os.path.exists(partial_path):
                    os.remove(partial_path)
        accelerator.wait_for_everyone()

        llm_inference(
            sentences_path=translation_input_path,
            output_path=translation_output_path,
            manifest_path=translation_manifest_path,
            terms_path=terms_path,
            partial_text_path=partial_paths["partial_text"],
            partial_meta_path=partial_paths["partial_meta"],
        )
        accelerator.wait_for_everyone()

        if epub_work_paths is not None and is_epub_path(final_output_path):
            if accelerator.is_main_process:
                text_to_epub(
                    original_epub_path=epub_work_paths["input_epub"],
                    translated_text_path=epub_work_paths["translated_text"],
                    manifest_path=epub_work_paths["manifest"],
                    output_epub_path=epub_work_paths["output_path"],
                    target_language_code=get_epub_language_code(llm_target_language),
                )
            accelerator.wait_for_everyone()

    if sentences_path is not None:
        translate_file(sentences_path, output_path)

    if sentences_dir is not None:
        print(
            f"Translating all files in {sentences_dir}, with extension {files_extension}"
        )
        os.makedirs(os.path.abspath(output_path), exist_ok=True)
        for filename in glob.glob(
            os.path.join(
                sentences_dir, f"*.{files_extension}" if files_extension else "*"
            )
        ):
            output_filename = os.path.join(output_path, os.path.basename(filename))
            translate_file(filename, output_filename)

    print("Translation done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the translation experiments")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sentences_path",
        default=None,
        type=str,
        help="Path to a txt file containing the sentences to translate. One sentence per line.",
    )

    input_group.add_argument(
        "--sentences_dir",
        type=str,
        default=None,
        help="Path to a directory containing the sentences to translate. "
        "Sentences must be in  .txt files containing containing one sentence per line.",
    )

    parser.add_argument(
        "--files_extension",
        type=str,
        default="txt",
        help="If sentences_dir is specified, extension of the files to translate. Defaults to txt. "
        "If set to an empty string, we will translate all files in the directory.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to a txt file where the translated sentences will be written. If the input is a directory, "
        "the output will be a directory with the same structure.",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Optional root directory for EPUB extraction, manifest, resume cache, and temporary translated text.",
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        required=False,
        help="Deprecated and ignored in LLM-only mode. Use --llm_target_language instead.",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        required=False,
        help="Deprecated and ignored in LLM-only mode. Use --llm_target_language instead.",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=None,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
        "If you use multiple devices, we will divide this number by the number of devices. "
        "Defaults to 1.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-14B-Instruct",
        help="Path or Hugging Face model name for an instruction-tuned CausalLM/chat model.",
    )

    parser.add_argument(
        "--lora_weights_name_or_path",
        type=str,
        default=None,
        help="If the model uses LoRA weights, path to those weights. See: https://github.com/huggingface/peft",
    )

    parser.add_argument(
        "--force_auto_device_map",
        action="store_true",
        help=" Whether to force the use of the auto device map. If set to True, "
        "the model will be split across GPUs and CPU to fit the model in memory. "
        "If set to False, a full copy of the model will be loaded  into each GPU. Defaults to False.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum number of newly generated tokens per LLM call. Defaults to 2048.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams for beam search. Defaults to 1.",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of translations to return for each input. LLM-only mode requires 1.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["bf16", "fp16", "32", "4", "8"],
        help="Precision of the model. bf16, fp16 or 32, 8 , 4 "
        "(4bits/8bits quantification, requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes). "
        "If None, we will use the torch.dtype of the model weights.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of beam search.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling. Used only if --do_sample is set.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If --do_sample is set, sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="If --do_sample is set, sample from nucleus probability mass top_p.",
    )

    parser.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Keep special tokens in the decoded text.",
    )

    parser.add_argument(
        "--keep_tokenization_spaces",
        action="store_true",
        help="Do not clean spaces in the decoded text.",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Deprecated legacy prompt option. Use --llm_prompt with a {TEXT} placeholder instead.",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="If set we will trust remote code in HuggingFace models. This is required for some models.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["flash_attention_2", "sdpa"],
        help="Optional HuggingFace attention implementation for CausalLM loading.",
    )

    parser.add_argument(
        "--source_lang_code",
        type=str,
        default="en",
        help="Source language code for TranslateGemma, for example en or de-DE.",
    )

    parser.add_argument(
        "--target_lang_code",
        type=str,
        default="zh-CN",
        help="Target language code for TranslateGemma, for example zh-CN or de-DE.",
    )

    parser.add_argument(
        "--llm_target_language",
        type=str,
        default="Simplified Chinese",
        help="Human-readable target language for LLM translation mode.",
    )

    parser.add_argument(
        "--llm_prompt",
        type=str,
        default=None,
        help="Prompt template for LLM translation mode. Must include {TEXT}. "
        "May also include {CONTEXT}, {CONTEXT_SECTION}, {TERMINOLOGY_SECTION}, "
        "{DOCUMENT_CONTEXT_SECTION}, and {TARGET_LANGUAGE}.",
    )

    parser.add_argument(
        "--context_window",
        type=int,
        default=0,
        help="Number of neighboring text blocks to provide as context in LLM translation mode.",
    )

    parser.add_argument(
        "--merge_small_blocks",
        action="store_true",
        help="Merge neighboring short blocks into numbered LLM translation batches while preserving output line count.",
    )

    parser.add_argument(
        "--merge_max_chars",
        type=int,
        default=1200,
        help="Legacy character fallback for merged short-block groups. "
        "--llm_input_max_length is the main LLM prompt budget.",
    )

    parser.add_argument(
        "--llm_input_max_length",
        type=int,
        default=8192,
        help="Maximum tokenized input prompt length for LLM translation mode. "
        "This is separate from --max_length, which controls generated tokens.",
    )

    parser.add_argument(
        "--llm_chunk_chars",
        type=int,
        default=3000,
        help="Approximate character fallback used while splitting oversized source blocks. "
        "--llm_input_max_length is the main LLM prompt budget.",
    )

    parser.add_argument(
        "--disable_auto_terms",
        action="store_true",
        help="Disable automatic terminology memory generation and prompt injection.",
    )

    parser.add_argument(
        "--disable_resume",
        action="store_true",
        help="Disable partial translation resume cache and translate from scratch.",
    )

    args = parser.parse_args()

    main(
        sentences_path=args.sentences_path,
        sentences_dir=args.sentences_dir,
        files_extension=args.files_extension,
        output_path=args.output_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        starting_batch_size=args.starting_batch_size,
        model_name=args.model_name,
        lora_weights_name_or_path=args.lora_weights_name_or_path,
        force_auto_device_map=args.force_auto_device_map,
        max_length=args.max_length,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        precision=args.precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        keep_special_tokens=args.keep_special_tokens,
        keep_tokenization_spaces=args.keep_tokenization_spaces,
        repetition_penalty=args.repetition_penalty,
        prompt=args.prompt,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        llm_target_language=args.llm_target_language,
        llm_prompt=args.llm_prompt,
        context_window=args.context_window,
        merge_small_blocks=args.merge_small_blocks,
        merge_max_chars=args.merge_max_chars,
        llm_input_max_length=args.llm_input_max_length,
        llm_chunk_chars=args.llm_chunk_chars,
        disable_auto_terms=args.disable_auto_terms,
        disable_resume=args.disable_resume,
        work_dir=args.work_dir,
    )

import argparse
import glob
import json
import os
import shutil
from typing import List, Optional

import torch
from accelerate import Accelerator, find_executable_batch_size
from tqdm import tqdm

from epub_converter import epub_to_text, require_epub_dependencies, text_to_epub
from llm_translation import (
    apply_chat_template_tokenized,
    build_context,
    build_llm_prompt,
    build_numbered_text,
    build_plain_prompt,
    is_llm_translation_model,
    make_translation_groups,
    parse_numbered_translations,
    read_text_lines,
    split_text_for_llm,
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


def get_epub_work_paths(input_path: str, output_path: str):
    work_dir = os.path.abspath(output_path) + ".easytranslate_epub"
    return {
        "work_dir": work_dir,
        "source_text": os.path.join(work_dir, "source.txt"),
        "translated_text": os.path.join(work_dir, "translated.txt"),
        "manifest": os.path.join(work_dir, "manifest.json"),
        "terms": os.path.join(work_dir, "terms.json"),
        "input_epub": input_path,
        "output_path": output_path,
    }


def get_terms_path(output_path: str) -> str:
    return os.path.abspath(output_path) + ".easytranslate_terms.json"


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
        }


def load_epub_unit_metadata(manifest_path: Optional[str], total_lines: int):
    if manifest_path is None or not os.path.exists(manifest_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    unit_metadata = [{} for _ in range(total_lines)]
    section_group = 0

    for unit in manifest.get("metadata", []):
        _fill_unit_metadata(
            unit_metadata,
            unit,
            kind=unit.get("unit_kind", "metadata"),
            file_name=unit.get("zip_name"),
            item_id=None,
            section_group=section_group,
        )

    for manifest_item in manifest.get("items", []):
        section_group += 1
        file_name = manifest_item.get("file_name")
        item_id = manifest_item.get("item_id")
        for block in manifest_item.get("blocks", []):
            kind = block.get("kind", "body")
            _fill_unit_metadata(
                unit_metadata,
                block,
                kind=kind,
                file_name=file_name,
                item_id=item_id,
                section_group=section_group,
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
    llm_target_language: str = "Simplified Chinese",
    llm_prompt: str = None,
    context_window: int = 0,
    merge_small_blocks: bool = False,
    merge_max_chars: int = 1200,
    llm_input_max_length: int = 8192,
    llm_chunk_chars: int = 3000,
    disable_auto_terms: bool = False,
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
    )

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
    if llm_prompt is not None and "{TEXT}" not in llm_prompt:
        raise ValueError("The --llm_prompt argument must include the {TEXT} placeholder.")
    if context_window < 0:
        raise ValueError("--context_window must be greater than or equal to 0.")
    if merge_max_chars <= 0:
        raise ValueError("--merge_max_chars must be greater than 0.")
    if llm_input_max_length <= 0:
        raise ValueError("--llm_input_max_length must be greater than 0.")
    if llm_chunk_chars <= 0:
        raise ValueError("--llm_chunk_chars must be greater than 0.")

    gen_kwargs = {
        "max_new_tokens": max_length,
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
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_token_ids.append(token_id)
    if stop_token_ids:
        gen_kwargs["eos_token_id"] = stop_token_ids
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    if accelerator.is_main_process:
        print(
            f"** Translation **\n"
            f"Input file: {sentences_path}\n"
            f"Sentences dir: {sentences_dir}\n"
            f"Output file: {output_path}\n"
            f"Deprecated source_lang argument: {source_lang}\n"
            f"Deprecated target_lang argument: {target_lang}\n"
            f"LLM-only translation mode: True\n"
            f"LLM target language: {llm_target_language}\n"
            f"Automatic terminology memory: {not disable_auto_terms}\n"
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
            f"Keep special tokens: {keep_special_tokens}\n"
            f"Keep tokenization spaces: {keep_tokenization_spaces}\n"
        )
        print("** Generation parameters **")
        print("\n".join(f"{k}: {v}" for k, v in gen_kwargs.items()))
        print("\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def llm_inference(
        batch_size,
        sentences_path,
        output_path,
        manifest_path=None,
        terms_path=None,
    ):
        nonlocal model, tokenizer, max_length, gen_kwargs

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return

        print(f"Translating {sentences_path} with LLM batch size {batch_size}")
        source_lines = read_text_lines(sentences_path)
        unit_metadata = load_epub_unit_metadata(manifest_path, len(source_lines))
        translations = [None] * len(source_lines)
        prepared_model = accelerator.prepare(model)

        def build_llm_batch(prompt_texts):
            tokenized_prompts = [
                apply_chat_template_tokenized(tokenizer, prompt_text)
                for prompt_text in prompt_texts
            ]
            if all(prompt is not None for prompt in tokenized_prompts):
                encoded_prompts = []
                for prompt in tokenized_prompts:
                    if hasattr(prompt, "keys") and "input_ids" in prompt:
                        prompt = prompt["input_ids"]
                    if hasattr(prompt, "squeeze"):
                        prompt = prompt.squeeze(0)
                    if hasattr(prompt, "tolist"):
                        prompt = prompt.tolist()
                    encoded_prompts.append(prompt)
                return tokenizer.pad(
                    {"input_ids": encoded_prompts},
                    padding=True,
                    return_tensors="pt",
                )

            rendered_prompts = [
                build_plain_prompt(tokenizer, prompt_text) for prompt_text in prompt_texts
            ]
            return tokenizer(
                rendered_prompts,
                padding=True,
                truncation=True,
                max_length=llm_input_max_length,
                return_tensors="pt",
            )

        def generate_prompts(prompt_texts):
            batch = build_llm_batch(prompt_texts)
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            generated_tokens = accelerator.unwrap_model(prepared_model).generate(
                **batch,
                **gen_kwargs,
            )
            generated_tokens = generated_tokens[:, batch["input_ids"].shape[1] :]
            return tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=not keep_special_tokens,
                clean_up_tokenization_spaces=not keep_tokenization_spaces,
            )

        terminology_memory = None
        if not disable_auto_terms and source_lines:
            terminology_memory = generate_terminology_memory(
                source_lines=source_lines,
                unit_metadata=unit_metadata,
                target_language=llm_target_language,
                terms_path=terms_path or get_terms_path(output_path),
                generate_prompts=generate_prompts,
            )

        groups = make_translation_groups(
            source_lines,
            merge_small_blocks=merge_small_blocks,
            merge_max_chars=merge_max_chars,
            unit_metadata=unit_metadata,
        )

        def terminology_section_for(text, context):
            if not terminology_memory:
                return ""
            relevant_entries = select_relevant_terms(
                terminology_memory,
                f"{text}\n{context}",
            )
            return format_terminology_section(relevant_entries)

        def build_prompt_for_group(group):
            context = build_context(
                source_lines,
                start_index=group[0],
                end_index=group[-1],
                window=context_window,
            )
            text = build_numbered_text(source_lines[index] for index in group)
            terminology_section = terminology_section_for(text, context)
            return build_llm_prompt(
                text=text,
                target_language=llm_target_language,
                context=context,
                prompt_template=llm_prompt,
                terminology_section=terminology_section,
            )

        def translate_single_line(index):
            chunks = split_text_for_llm(source_lines[index], llm_chunk_chars)
            context = build_context(
                source_lines,
                start_index=index,
                end_index=index,
                window=context_window,
            )
            text = build_numbered_text(chunks)
            prompt_text = build_llm_prompt(
                text=text,
                target_language=llm_target_language,
                context=context,
                prompt_template=llm_prompt,
                terminology_section=terminology_section_for(text, context),
            )
            decoded_output = generate_prompts([prompt_text])[0]
            try:
                translated_chunks = parse_numbered_translations(
                    decoded_output,
                    expected_count=len(chunks),
                )
            except ValueError:
                if len(chunks) == 1:
                    translated_chunks = [decoded_output.strip()]
                else:
                    fallback_prompts = [
                        build_llm_prompt(
                            text=chunk,
                            target_language=llm_target_language,
                            context="",
                            prompt_template=llm_prompt,
                            terminology_section=terminology_section_for(chunk, ""),
                        )
                        for chunk in chunks
                    ]
                    translated_chunks = [
                        output.strip() for output in generate_prompts(fallback_prompts)
                    ]
            return " ".join(chunk.strip() for chunk in translated_chunks if chunk.strip())

        with tqdm(
            total=len(source_lines),
            desc="LLM translation",
            leave=True,
            ascii=True,
        ) as pbar, open(output_path, "w", encoding="utf-8") as output_file:
            with torch.no_grad():
                for group_start in range(0, len(groups), batch_size):
                    batch_groups = groups[group_start : group_start + batch_size]
                    prompt_texts = [build_prompt_for_group(group) for group in batch_groups]
                    decoded_outputs = generate_prompts(prompt_texts)

                    for group, decoded_output in zip(batch_groups, decoded_outputs):
                        if len(group) == 1:
                            try:
                                parsed_outputs = parse_numbered_translations(
                                    decoded_output,
                                    expected_count=1,
                                )
                                translations[group[0]] = parsed_outputs[0].strip()
                            except ValueError:
                                translations[group[0]] = translate_single_line(group[0])
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
                            translations[index] = translated_text.strip()
                        pbar.update(len(group))

            for translated_text in translations:
                print(encode_string(translated_text or ""), file=output_file)

        accelerator.wait_for_everyone()
        print(f"Translation done. Output written to {output_path}\n")

    def translate_file(input_path, final_output_path):
        epub_work_paths = None
        translation_manifest_path = None
        translation_input_path = input_path
        translation_output_path = final_output_path
        terms_path = get_terms_path(final_output_path)

        if is_epub_path(input_path):
            epub_work_paths = get_epub_work_paths(input_path, final_output_path)
            translation_input_path = epub_work_paths["source_text"]
            translation_manifest_path = epub_work_paths["manifest"]
            terms_path = epub_work_paths["terms"]
            if is_epub_path(final_output_path):
                translation_output_path = epub_work_paths["translated_text"]

            if accelerator.is_main_process:
                if os.path.exists(epub_work_paths["work_dir"]):
                    shutil.rmtree(epub_work_paths["work_dir"])
                os.makedirs(epub_work_paths["work_dir"], exist_ok=True)
                epub_to_text(
                    epub_path=input_path,
                    text_path=epub_work_paths["source_text"],
                    manifest_path=epub_work_paths["manifest"],
                )

            accelerator.wait_for_everyone()

        os.makedirs(os.path.abspath(os.path.dirname(final_output_path)), exist_ok=True)
        llm_inference(
            sentences_path=translation_input_path,
            output_path=translation_output_path,
            manifest_path=translation_manifest_path,
            terms_path=terms_path,
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
        "and {TARGET_LANGUAGE}.",
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
        help="Maximum total characters per merged short-block group in LLM translation mode.",
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
        help="Split oversized source blocks into chunks of about this many characters before LLM translation.",
    )

    parser.add_argument(
        "--disable_auto_terms",
        action="store_true",
        help="Disable automatic terminology memory generation and prompt injection.",
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
        llm_target_language=args.llm_target_language,
        llm_prompt=args.llm_prompt,
        context_window=args.context_window,
        merge_small_blocks=args.merge_small_blocks,
        merge_max_chars=args.merge_max_chars,
        llm_input_max_length=args.llm_input_max_length,
        llm_chunk_chars=args.llm_chunk_chars,
        disable_auto_terms=args.disable_auto_terms,
    )

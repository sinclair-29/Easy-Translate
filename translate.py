import argparse
import glob
import math
import os
import shutil
from typing import Optional

import torch
from accelerate import Accelerator, DistributedType, find_executable_batch_size
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from dataset import DatasetReader, count_lines
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
        "input_epub": input_path,
        "output_path": output_path,
    }


def model_supports_language_tokens(model, tokenizer) -> bool:
    return hasattr(tokenizer, "lang_code_to_id") or model.config.model_type in {
        "m2m_100",
    }


def get_language_token_id(tokenizer, language: str):
    if language is None:
        return None
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id.get(language)

    token_id = tokenizer.convert_tokens_to_ids(language)
    if token_id is None or token_id == tokenizer.unk_token_id:
        return None
    return token_id


def describe_supported_languages(tokenizer):
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id.keys()
    return "language tokens in this tokenizer vocabulary"


def get_dataloader(
    accelerator: Accelerator,
    filename: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
    prompt: str,
) -> DataLoader:
    dataset = DatasetReader(
        filename=filename,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt=prompt,
    )
    if accelerator.distributed_type == DistributedType.XLA:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            padding="max_length",
            max_length=max_length,
            # label_pad_token_id=tokenizer.pad_token_id,
            return_tensors="pt",
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            padding=True,
            # label_pad_token_id=tokenizer.pad_token_id,
            # max_length=max_length, No need to set max_length here, we already truncate in the preprocess function
            # pad_to_multiple_of=8,
            return_tensors="pt",
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,  # Disable multiprocessing
    )


def main(
    sentences_path: Optional[str],
    sentences_dir: Optional[str],
    files_extension: str,
    output_path: str,
    source_lang: Optional[str],
    target_lang: Optional[str],
    starting_batch_size: Optional[int] = None,
    model_name: str = "facebook/m2m100_1.2B",
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

    is_llm_translation = is_llm_translation_model(model, tokenizer, model_name)
    if max_length is None:
        max_length = 2048 if is_llm_translation else 256
    if num_beams is None:
        num_beams = 1 if is_llm_translation else 5
    if starting_batch_size is None:
        starting_batch_size = 1 if is_llm_translation else 128
    if temperature is None and not is_llm_translation:
        temperature = 0.8
    if top_k is None and not is_llm_translation:
        top_k = 100
    if top_p is None and not is_llm_translation:
        top_p = 0.75

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

    if is_llm_translation and num_return_sequences != 1:
        raise ValueError("LLM translation mode requires --num_return_sequences 1.")

    is_translation_model = (
        False if is_llm_translation else model_supports_language_tokens(model, tokenizer)
    )
    lang_code_to_idx = None

    if (
        is_translation_model
        and (source_lang is None or target_lang is None)
        and "small100" not in model_name
    ):
        raise ValueError(
            f"The model you are using requires a source and target language. "
            f"Please specify them with --source-lang and --target-lang. "
            f"The supported languages are: {describe_supported_languages(tokenizer)}"
        )
    if not is_llm_translation and not is_translation_model and (
        source_lang is not None or target_lang is not None
    ):
        if prompt is None:
            print(
                "WARNING: You are using a model that does not support source and target languages parameters "
                "but you specified them. You probably want to use m2m100/nllb200 for translation or "
                "set --prompt to define the task for you model. "
            )
        else:
            print(
                "WARNING: You are using a model that does not support source and target languages parameters "
                "but you specified them."
            )

    if prompt is not None and "%%SENTENCE%%" not in prompt:
        raise ValueError(
            f"The prompt must contain the %%SENTENCE%% token to indicate where the sentence should be inserted. "
            f"Your prompt: {prompt}"
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

    if is_translation_model and "small100" in model_name:
        lang_code_to_idx = get_language_token_id(tokenizer, target_lang)
        if lang_code_to_idx is None:
            raise KeyError(
                f"Language {target_lang} not found in tokenizer. Available languages: {describe_supported_languages(tokenizer)}"
            )
        tokenizer.tgt_lang = target_lang
        # We don't need to force the BOS token, so we set is_translation_model to False
        is_translation_model = False

    if is_translation_model:
        source_lang_idx = get_language_token_id(tokenizer, source_lang)
        if source_lang_idx is None:
            raise KeyError(
                f"Language {source_lang} not found in tokenizer. Available languages: {describe_supported_languages(tokenizer)}"
            )
        tokenizer.src_lang = source_lang

        lang_code_to_idx = get_language_token_id(tokenizer, target_lang)
        if lang_code_to_idx is None:
            raise KeyError(
                f"Language {target_lang} not found in tokenizer. Available languages: {describe_supported_languages(tokenizer)}"
            )

    if model.config.model_type == "seamless_m4t":
        # Loading a seamless_m4t model, we need to set a few things to ensure compatibility

        supported_langs = tokenizer.additional_special_tokens
        supported_langs = [lang.replace("__", "") for lang in supported_langs]

        if source_lang is None or target_lang is None:
            raise ValueError(
                f"The model you are using requires a source and target language. "
                f"Please specify them with --source-lang and --target-lang. "
                f"The supported languages are: {supported_langs}"
            )

        if source_lang not in supported_langs:
            raise ValueError(
                f"Language {source_lang} not found in tokenizer. Available languages: {supported_langs}"
            )
        if target_lang not in supported_langs:
            raise ValueError(
                f"Language {target_lang} not found in tokenizer. Available languages: {supported_langs}"
            )

        tokenizer.src_lang = source_lang

    gen_kwargs = {
        "max_new_tokens": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
    }

    if is_llm_translation:
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = top_p

    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    if is_llm_translation:
        stop_token_ids = []
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                stop_token_ids.append(token_id)
        if stop_token_ids:
            gen_kwargs["eos_token_id"] = stop_token_ids
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    if is_translation_model:
        gen_kwargs["forced_bos_token_id"] = lang_code_to_idx

    if model.config.model_type == "seamless_m4t":
        gen_kwargs["tgt_lang"] = target_lang

    if accelerator.is_main_process:
        print(
            f"** Translation **\n"
            f"Input file: {sentences_path}\n"
            f"Sentences dir: {sentences_dir}\n"
            f"Output file: {output_path}\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"LLM translation mode: {is_llm_translation}\n"
            f"LLM target language: {llm_target_language}\n"
            f"Context window: {context_window}\n"
            f"Merge small blocks: {merge_small_blocks}\n"
            f"LLM input max length: {llm_input_max_length}\n"
            f"LLM chunk chars: {llm_chunk_chars}\n"
            f"Force target lang as BOS token: {is_translation_model}\n"
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
    def inference(batch_size, sentences_path, output_path):
        nonlocal \
            model, \
            tokenizer, \
            max_length, \
            gen_kwargs, \
            precision, \
            prompt, \
            is_translation_model

        print(f"Translating {sentences_path} with batch size {batch_size}")

        total_lines: int = count_lines(sentences_path)

        data_loader = get_dataloader(
            accelerator=accelerator,
            filename=sentences_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            prompt=prompt,
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        samples_seen: int = 0

        with tqdm(
            total=total_lines,
            desc="Dataset translation",
            leave=True,
            ascii=True,
            disable=(not accelerator.is_main_process),
        ) as pbar, open(output_path, "w", encoding="utf-8") as output_file:
            with torch.no_grad():
                for step, batch in enumerate(data_loader):
                    batch["input_ids"] = batch["input_ids"]
                    batch["attention_mask"] = batch["attention_mask"]

                    generated_tokens = accelerator.unwrap_model(model).generate(
                        **batch,
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )

                    tgt_text = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=not keep_special_tokens,
                        clean_up_tokenization_spaces=not keep_tokenization_spaces,
                    )
                    if accelerator.is_main_process:
                        if (
                            step
                            == math.ceil(
                                math.ceil(total_lines / batch_size)
                                / accelerator.num_processes
                            )
                            - 1
                        ):
                            tgt_text = tgt_text[
                                : (total_lines * num_return_sequences) - samples_seen
                            ]
                        else:
                            samples_seen += len(tgt_text)

                        print(
                            "\n".join(
                                [encode_string(sentence) for sentence in tgt_text]
                            ),
                            file=output_file,
                        )

                    pbar.update(len(tgt_text) // gen_kwargs["num_return_sequences"])

        print(f"Translation done. Output written to {output_path}\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def llm_inference(batch_size, sentences_path, output_path):
        nonlocal model, tokenizer, max_length, gen_kwargs

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return

        print(f"Translating {sentences_path} with LLM batch size {batch_size}")
        source_lines = read_text_lines(sentences_path)
        groups = make_translation_groups(
            source_lines,
            merge_small_blocks=merge_small_blocks,
            merge_max_chars=merge_max_chars,
        )
        translations = [None] * len(source_lines)
        prepared_model = accelerator.prepare(model)

        def build_llm_batch(prompt_texts):
            tokenized_prompts = [
                apply_chat_template_tokenized(tokenizer, prompt_text)
                for prompt_text in prompt_texts
            ]
            if all(prompt is not None for prompt in tokenized_prompts):
                encoded_prompts = [
                    prompt.squeeze(0).tolist() if hasattr(prompt, "squeeze") else prompt
                    for prompt in tokenized_prompts
                ]
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

        def build_prompt_for_group(group):
            context = build_context(
                source_lines,
                start_index=group[0],
                end_index=group[-1],
                window=context_window,
            )
            text = build_numbered_text(source_lines[index] for index in group)
            return build_llm_prompt(
                text=text,
                target_language=llm_target_language,
                context=context,
                prompt_template=llm_prompt,
            )

        def translate_single_line(index):
            chunks = split_text_for_llm(source_lines[index], llm_chunk_chars)
            prompt_text = build_llm_prompt(
                text=build_numbered_text(chunks),
                target_language=llm_target_language,
                context=build_context(
                    source_lines,
                    start_index=index,
                    end_index=index,
                    window=context_window,
                ),
                prompt_template=llm_prompt,
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
        translation_input_path = input_path
        translation_output_path = final_output_path

        if is_epub_path(input_path):
            epub_work_paths = get_epub_work_paths(input_path, final_output_path)
            translation_input_path = epub_work_paths["source_text"]
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
        if is_llm_translation:
            llm_inference(
                sentences_path=translation_input_path,
                output_path=translation_output_path,
            )
        else:
            inference(
                sentences_path=translation_input_path,
                output_path=translation_output_path,
            )
        accelerator.wait_for_everyone()

        if epub_work_paths is not None and is_epub_path(final_output_path):
            if accelerator.is_main_process:
                text_to_epub(
                    original_epub_path=epub_work_paths["input_epub"],
                    translated_text_path=epub_work_paths["translated_text"],
                    manifest_path=epub_work_paths["manifest"],
                    output_epub_path=epub_work_paths["output_path"],
                )
                shutil.rmtree(epub_work_paths["work_dir"], ignore_errors=True)
            accelerator.wait_for_everyone()
        elif epub_work_paths is not None:
            if accelerator.is_main_process:
                shutil.rmtree(epub_work_paths["work_dir"], ignore_errors=True)
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
        help="Source language id. See: supported_languages.md. Required for m2m100 and nllb200",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        required=False,
        help="Source language id. See: supported_languages.md. Required for m2m100 and nllb200",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=None,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
        "If you use multiple devices, we will divide this number by the number of devices. "
        "Defaults to 128 for traditional MT models and 1 for LLM translation mode.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_1.2B",
        help="Path to the model to use. See: https://huggingface.co/models",
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
        help="Maximum number of tokens in the source sentence and generated sentence. "
        "Increase this value to translate longer sentences, at the cost of increasing memory usage. "
        "Defaults to 256 for traditional MT models and 2048 for LLM translation mode.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams for beam search. Defaults to 5 for traditional MT models and 1 for LLM translation mode.",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of possible translation to return for each sentence (num_return_sequences<=num_beams).",
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
        help="Temperature for sampling, value used only if do_sample is True. Defaults to 0.8 for traditional MT models.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If do_sample is True, will sample from the top k most likely tokens. Defaults to 100 for traditional MT models.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="If do_sample is True, will sample from nucleus probability mass top_p. Defaults to 0.75 for traditional MT models.",
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
        help="Prompt to use for generation. "
        "It must include the special token %%SENTENCE%% which will be replaced by the sentence to translate.",
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
        "May also include {CONTEXT}, {CONTEXT_SECTION}, and {TARGET_LANGUAGE}.",
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
    )

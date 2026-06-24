# Easy-Translate

Easy-Translate is a lightweight LLM-only translation tool for large text files
and EPUB books. This fork focuses on instruction-tuned CausalLM/chat models such
as Qwen3-14B-Instruct and Qwen2.5-Instruct.

Legacy Seq2Seq machine-translation models such as NLLB, M2M100, SeamlessM4T,
MBART, MarianMT, T5, and similar language-token based models are no longer
supported in this version.

## Features

- TXT input and TXT output, one source block per line.
- EPUB input with TXT or EPUB output.
- Instruction-tuned chat model prompting via tokenizer chat templates.
- Deterministic LLM generation by default.
- Optional neighboring-block context with `--context_window`.
- Optional short-block grouping with `--merge_small_blocks`.
- Automatic terminology memory for recurring names and terms.
- Long block splitting with `--llm_chunk_chars`.
- BF16 / FP16 / FP32 / 8-bit / 4-bit model loading.
- LoRA model loading.
- Accelerate-based runtime setup.

## Requirements

```bash
pip install torch accelerate transformers
```

Optional quantization support:

```bash
pip install bitsandbytes
```

Optional LoRA support:

```bash
pip install peft
```

Optional EPUB support:

```bash
pip install ebooklib beautifulsoup4 lxml
```

Qwen3 may require a recent Transformers version:

```bash
pip install --upgrade transformers
```

## Translate An EPUB

```bash
python3 translate.py \
  --sentences_path book.epub \
  --output_path book_zh.epub \
  --model_name /path/to/Qwen3-14B-Instruct \
  --precision fp16 \
  --starting_batch_size 1 \
  --context_window 1 \
  --merge_small_blocks \
  --llm_input_max_length 8192 \
  --llm_chunk_chars 4000 \
  --max_length 1536
```

If the EPUB output path ends in `.txt`, Easy-Translate writes one translated
line per extracted EPUB block instead of rebuilding an EPUB.

## Translate A TXT File

```bash
python3 translate.py \
  --sentences_path sample_text/en.txt \
  --output_path sample_text/en.zh.txt \
  --model_name /path/to/Qwen3-14B-Instruct \
  --precision fp16 \
  --starting_batch_size 1 \
  --llm_target_language "Simplified Chinese"
```

To translate every file in a directory:

```bash
python3 translate.py \
  --sentences_dir sample_text/ \
  --output_path sample_text/translations \
  --files_extension txt \
  --model_name /path/to/Qwen3-14B-Instruct \
  --precision fp16
```

Use `--files_extension epub` to translate EPUB files in a directory.

## LLM Prompting

The default prompt asks the model to produce fluent, accurate Simplified Chinese
without explanations or summaries.

Use `--llm_target_language` to choose another target language:

```bash
python3 translate.py \
  --sentences_path book.epub \
  --output_path book_ja.epub \
  --model_name /path/to/Qwen3-14B-Instruct \
  --precision fp16 \
  --llm_target_language "Japanese"
```

Use `--llm_prompt` to provide a custom prompt template. It must include `{TEXT}`;
it may also include `{CONTEXT}`, `{CONTEXT_SECTION}`, `{TERMINOLOGY_SECTION}`,
and `{TARGET_LANGUAGE}`.

```bash
python3 translate.py \
  --sentences_path sample_text/en.txt \
  --output_path sample_text/en.zh.txt \
  --model_name /path/to/Qwen3-14B-Instruct \
  --llm_prompt "Translate into {TARGET_LANGUAGE}. Return only the translation.\n\n{TEXT}"
```

The old `--prompt` / `%%SENTENCE%%` prompting path has been removed. Use
`--llm_prompt` instead.

## Automatic Terminology Memory

By default, Easy-Translate scans the extracted source blocks before translation,
asks the loaded LLM to prepare a compact JSON terminology memory, and injects
only relevant entries into each translation prompt.

For EPUB input, the memory is saved as `terms.json` in the
`.easytranslate_epub` work directory. For TXT input, it is saved next to the
output as `<output_path>.easytranslate_terms.json`.

Disable this step with `--disable_auto_terms`.

## Useful Runtime Options

- `--model_name`: local path or Hugging Face model name for an instruction-tuned
  CausalLM/chat model.
- `--precision`: `bf16`, `fp16`, `32`, `8`, or `4`.
- `--starting_batch_size`: starts at `1` by default and is reduced automatically
  if an OOM occurs.
- `--max_length`: hard upper limit for the dynamic newly generated token
  budget per LLM call.
- `--context_window`: neighboring source blocks passed as context.
- `--merge_small_blocks`: groups neighboring short blocks into numbered LLM
  calls while preserving output line count.
- `--merge_max_chars`: legacy character fallback for merged groups; the rendered
  prompt token budget is controlled by `--llm_input_max_length`.
- `--llm_input_max_length`: maximum tokenized prompt length.
- `--llm_chunk_chars`: approximate source characters per chunk for oversized
  source blocks after token-budget checks.
- `--disable_auto_terms`: skip automatic terminology memory generation and
  prompt injection.
- `--do_sample`, `--temperature`, `--top_k`, `--top_p`: optional sampling
  controls. Sampling parameters are only sent to the model when `--do_sample` is
  enabled.

`--source_lang` and `--target_lang` are deprecated and ignored in this LLM-only
version. Use `--llm_target_language` instead.

## Unsupported Models

This version fails early if you try to load encoder-decoder translation models
such as NLLB, M2M100, SeamlessM4T, MBART, MarianMT, T5, or similar Seq2Seq
models. Use an instruction-tuned CausalLM/chat model such as
Qwen3-14B-Instruct.

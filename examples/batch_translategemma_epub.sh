#!/usr/bin/env bash
set -uo pipefail

MODEL_NAME="${MODEL_NAME:-../LLMJailbreak/models/translategemma-12b-it}"
INPUT_DIR="${INPUT_DIR:-books/input}"
OUTPUT_DIR="${OUTPUT_DIR:-books/output}"
WORK_DIR="${WORK_DIR:-books/work}"
LOG_DIR="${LOG_DIR:-books/logs}"
SOURCE_LANG_CODE="${SOURCE_LANG_CODE:-en}"
TARGET_LANG_CODE="${TARGET_LANG_CODE:-zh-CN}"
PRECISION="${PRECISION:-bf16}"
STARTING_BATCH_SIZE="${STARTING_BATCH_SIZE:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LLM_INPUT_MAX_LENGTH="${LLM_INPUT_MAX_LENGTH:-1800}"
LLM_CHUNK_CHARS="${LLM_CHUNK_CHARS:-3000}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/batch_translategemma_epub.sh
  bash examples/batch_translategemma_epub.sh book1.epub book2.epub ...
  bash examples/batch_translategemma_epub.sh --list books.txt

With no arguments, the script translates all EPUB files in books/input.
Each non-empty, non-comment line in books.txt should be an EPUB path.

Useful environment overrides:
  MODEL_NAME=../LLMJailbreak/models/translategemma-12b-it
  INPUT_DIR=books/input
  OUTPUT_DIR=books/output
  WORK_DIR=books/work
  LOG_DIR=books/logs
  STARTING_BATCH_SIZE=4
  ATTN_IMPLEMENTATION=sdpa
EOF
}

books=()
if [[ $# -eq 0 ]]; then
  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory not found: $INPUT_DIR" >&2
    echo "Create it and put EPUB files there, or pass EPUB paths explicitly." >&2
    exit 1
  fi
  shopt -s nullglob
  books=("$INPUT_DIR"/*.epub "$INPUT_DIR"/*.EPUB)
  shopt -u nullglob
  if [[ ${#books[@]} -eq 0 ]]; then
    echo "No EPUB files found in $INPUT_DIR" >&2
    exit 1
  fi
fi

if [[ $# -eq 0 ]]; then
  :
elif [[ "${1:-}" == "--list" ]]; then
  if [[ $# -ne 2 ]]; then
    usage
    exit 1
  fi
  list_path="$2"
  if [[ ! -f "$list_path" ]]; then
    echo "Book list not found: $list_path" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    books+=("$line")
  done < "$list_path"
else
  books=("$@")
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$LOG_DIR"

total=${#books[@]}
failed=0

for index in "${!books[@]}"; do
  book="${books[$index]}"
  if [[ ! -f "$book" ]]; then
    echo "[$((index + 1))/$total] Missing input: $book" >&2
    failed=$((failed + 1))
    continue
  fi

  base="$(basename "$book")"
  stem="${base%.*}"
  output_path="$OUTPUT_DIR/${stem}_zh.epub"
  log_path="$LOG_DIR/${stem}_$(date +%Y%m%d_%H%M%S).log"

  echo "[$((index + 1))/$total] Translating: $book"
  echo "  Output: $output_path"
  echo "  Log:    $log_path"

  python3 translate.py \
    --sentences_path "$book" \
    --output_path "$output_path" \
    --work_dir "$WORK_DIR" \
    --model_name "$MODEL_NAME" \
    --precision "$PRECISION" \
    --starting_batch_size "$STARTING_BATCH_SIZE" \
    --source_lang_code "$SOURCE_LANG_CODE" \
    --target_lang_code "$TARGET_LANG_CODE" \
    --merge_small_blocks \
    --context_window "$CONTEXT_WINDOW" \
    --llm_input_max_length "$LLM_INPUT_MAX_LENGTH" \
    --llm_chunk_chars "$LLM_CHUNK_CHARS" \
    --max_length "$MAX_LENGTH" \
    --disable_auto_terms \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    2>&1 | tee "$log_path"

  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "  FAILED with status $status: $book" >&2
    failed=$((failed + 1))
  else
    echo "  DONE: $output_path"
  fi
done

echo "Batch finished: $((total - failed)) succeeded, $failed failed, $total total."
exit "$failed"

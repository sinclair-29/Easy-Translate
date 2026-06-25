# Easy-Translate Examples

This fork is LLM-only. The examples use instruction-tuned CausalLM/chat models
such as Qwen3-14B-Instruct.

> Run the examples from the root of the Easy-Translate repository as follows: `sh examples/<example>.sh`

## EPUB Translation

```bash
qwen3_epub.sh
```

## Batch TranslateGemma EPUB Translation

Translate multiple EPUB files sequentially and write per-book logs:

```bash
bash examples/batch_translategemma_epub.sh
```

By default this reads EPUB files from `books/input`, writes translated EPUBs to
`books/output`, writes intermediate resume files to `books/work`, and writes logs
to `books/logs`.

You can also pass paths explicitly:

```bash
bash examples/batch_translategemma_epub.sh book1.epub book2.epub
```

Or read paths from a text file:

```bash
bash examples/batch_translategemma_epub.sh --list books.txt
```

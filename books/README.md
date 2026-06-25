# Local Book Workspace

Use this folder to keep long-running EPUB translation jobs separated from the
repository code.

- `input/`: put source EPUB files here.
- `output/`: translated EPUB files are written here by the batch script.
- `work/`: extraction manifests, source text, translated text, and resume caches.
- `logs/`: one log file per translated book.

The real contents of these folders are ignored by git. The `.gitkeep` files only
keep the empty folder structure in the repository.

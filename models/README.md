# Local model store

`make download-model` downloads the public TinyStories `stories260K` checkpoint
from the `stories260K/` Hugging Face directory and its matching `tok512.bin`
tokenizer into this directory. The binary files are intentionally ignored by
Git. Pass `--model-size stories15M` to the downloader for the optional larger
checkpoint.

Run `make convert-model` to validate the llama2.c headers and tokenizer, then
emit the versioned kernel packages in `build/`. `make validate-model` checks
the package checksums without downloading or training anything. The normal
`make all`/`make bench` workflow still executes the compact character model;
`make llm-iso` is the explicit TinyStories boot path.

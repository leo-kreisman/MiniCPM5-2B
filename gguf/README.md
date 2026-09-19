# gguf/

`MiniCPM5-2B-Q4_K_M.gguf` lands here after `./assemble.sh`.

This directory carries no config files on purpose. Unlike the MLX and bf16
builds — which need `config.json` and a tokenizer sitting beside the weights —
a GGUF is self-contained: it carries its own tokenizer and chat template. That
is why `llama-server -m gguf/MiniCPM5-2B-Q4_K_M.gguf` needs no other argument to
produce correct output.

Do not copy `config.json` or the tokenizer JSON files in here to "match" the
other directories. llama.cpp does not read them, and a stale `config.json` next
to a GGUF is a good way to make a later reader think this build needs them.

The file is gitignored. See `SETUP.md` §3 for serving it.

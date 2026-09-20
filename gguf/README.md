# gguf/

Two files land here after `./assemble.sh`:

| File | Size | ~bpw | For |
| --- | ---: | ---: | --- |
| `MiniCPM5-2B-Q4_K_M.gguf` | 1,561,318,368 B | 4.96 | the memory-tight choice |
| `MiniCPM5-2B-Q8_0.gguf` | 2,679,710,688 B | 8.52 | precision, still with tool calling |

Both run on `llama-server`; only one at a time needs to be loaded. `Q8_0` costs
1.12 GB more and is the same precision and size as the MLX 8-bit build — the
difference being that this one can call tools. `SETUP.md` §1 has what each costs
against a 9 GB budget; §3 has the run lines.

This directory carries no config files on purpose. Unlike the MLX and bf16
builds — which need `config.json` and a tokenizer sitting beside the weights —
a GGUF is self-contained: it carries its own tokenizer and chat template. That
is why `llama-server -m gguf/MiniCPM5-2B-Q8_0.gguf` needs no other argument to
produce correct output.

Do not copy `config.json` or the tokenizer JSON files in here to "match" the
other directories. llama.cpp does not read them, and a stale `config.json` next
to a GGUF is a good way to make a later reader think this build needs them.

Both files are gitignored. See `SETUP.md` §3 for serving them.

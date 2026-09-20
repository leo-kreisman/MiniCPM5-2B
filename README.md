# MiniCPM5-2B

A GitHub-native mirror of the **MiniCPM5-2B** builds that run on Apple Silicon,
shipped as Release assets so a Mac can assemble them without a Hugging Face
token, without LFS, and without a compiler.

Five builds, four of them official OpenBMB releases:

| Build | Bytes | Upstream | Runtime | Ships as |
| --- | ---: | --- | --- | --- |
| bf16 | 5,033,557,096 | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) | Torch | 3 parts |
| MLX 4-bit | 1,416,035,216 | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX) | MLX | whole |
| MLX 8-bit | 2,674,327,290 | [`mlx-community/MiniCPM5-2B-8bit`](https://huggingface.co/mlx-community/MiniCPM5-2B-8bit) | MLX | 2 parts |
| GGUF Q4_K_M | 1,561,318,368 | [`openbmb/MiniCPM5-2B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) | llama.cpp | whole |
| GGUF Q8_0 | 2,679,710,688 | [`openbmb/MiniCPM5-2B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) | llama.cpp | 2 parts |

Only the 8-bit MLX build is community. The other four are official.

**`GGUF Q8_0` is the build to reach for if you want precision *and* tool calling.**
It is the same size and the same precision as the MLX 8-bit build — 2.68 GB
against 2.67 GB — but it runs on `llama-server`, which parses this model's tool
calls, where MLX has no parser for them at all. §1 and §2 of `SETUP.md`.

**Start with [`SETUP.md`](SETUP.md).** §1 is the memory arithmetic — MiniCPM5-2B
has 42 layers of full attention and no linear layers, so its KV cache is 43,008
bytes per token, which is what actually decides which build you can run at which
context length. Read it before downloading 10 GB.

> **This is a mirror.** The canonical copies live upstream on Hugging Face; links
> are in the table above. This repository redistributes the weights and vendors
> the upstream documentation it relies on. It does not modify any model file.

## Getting the weights

```bash
git clone https://github.com/leo-kreisman/MiniCPM5-2B.git
cd MiniCPM5-2B
./assemble.sh              # fetch, verify, assemble all five builds
./assemble.sh --check      # verify what is already here; download nothing
```

Each build lands in its own directory with its own `config.json` and tokenizer,
because the containers are not interchangeable — an MLX directory cannot be read
by llama.cpp and vice versa.

| Directory | Contents after `assemble.sh` |
| --- | --- |
| `bf16/` | `model-00000-of-00001.safetensors` + config + tokenizer |
| `mlx-4bit/` | `model.safetensors` + config + tokenizer |
| `mlx-8bit/` | `model.safetensors` + config + tokenizer |
| `gguf/` | `MiniCPM5-2B-Q4_K_M.gguf` and `MiniCPM5-2B-Q8_0.gguf` (self-contained) |

`assemble.sh` is resumable: a part already downloaded and verified is skipped, and
a part that fails its checksum is re-fetched. Every part and every reassembled
file is checked against the sha256 published by Hugging Face itself.

> **If a file fails its checksum, the download is wrong — not the hash.** Re-run
> `assemble.sh`; it deletes and re-fetches bad parts on its own. Do **not** edit
> the expected hashes to match a download. Those values pin the published bytes,
> and relaxing one turns a loud, catchable failure into a model that loads and
> silently produces wrong output.

## What is in this repository

| Path | What it is |
| --- | --- |
| **`SETUP.md`** | **start here** — the three routes, the 9 GB budget, and the traps |
| `assemble.sh` | fetch, verify and assemble all five builds |
| `MANIFEST.sha256` | every part's sha256, and every reassembled file's |
| `vendor/` | the **official** upstream docs, verbatim, at pinned commits |
| `VENDORED-FROM.md` | each vendored file's upstream commit, size, sha256 and licence |
| `scripts/` | `fetch_sources.py`, `vendor_docs.py`, and the release tooling |

`vendor/` holds the authoritative instructions for SemIf, MLX-LM, llama.cpp and
OMP. Nothing under it is hand-edited: `scripts/vendor_docs.py` writes those files
and `VENDORED-FROM.md` together, resolving each branch to a commit at run time so
a later upstream push cannot silently change what was vendored. Where this
repository's own guide disagrees with a vendored file, **the vendored file is
right.**

## The three routes

| Route | Runtime | For |
| --- | --- | --- |
| **A** | `mlx_lm.server` | chat and completion on MLX — §2 |
| **B** | `llama-server` + GGUF | chat and completion, no Python — §3 |
| **C** | SemIf | one-pass scoring; **not** a chat runtime — §4 |

Route C carries the finding worth knowing before you plan around it: **SemIf's
MLX backend refuses MiniCPM5-2B.** Its loader gates on
`model_type not in {"qwen3_5"}` and this model is `llama`, so it raises before
reading a weight. SemIf does support the model — it pins it twice, for the Torch
path and for its browser ladder — but not through MLX. §4 has the scoped version
of that claim, which matters: the over-broad version ("SemIf can't run this") is
wrong in the other direction.

## Provenance

Weight bytes come from the four Hugging Face repositories above, each fetched at
an immutable 40-character commit revision and verified against the `lfs.oid`
sha256 that Hugging Face publishes in its tree API. The pinned revisions and
hashes are recorded in `SOURCES.json` and checked in, so this mirror is
reproducible rather than merely recent.

## Licence

The weights carry the licence of their upstream repositories — see each model
card on Hugging Face. This repository's own scripts and documentation are
MIT. Vendored files under `vendor/` retain their upstream licences, listed per
file in [`VENDORED-FROM.md`](VENDORED-FROM.md).

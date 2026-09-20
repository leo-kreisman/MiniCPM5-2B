# Running MiniCPM5-2B on macOS

Everything below is assembled from the official documentation vendored in
[`vendor/`](vendor/) — see [`VENDORED-FROM.md`](VENDORED-FROM.md) for the exact
commit and sha256 of each source. Where this guide summarises an official
document, the vendored file wins. Where this guide adds something the official
documents do not contain (the memory arithmetic in §1, the OMP wiring in §2 and
§3), it is marked as ours.

> **Read §1 before you download anything.** MiniCPM5-2B has **42 layers of full
> attention and no hybrid/linear layers**, so its KV cache is unusually large for
> a 2.5B model: 43,008 bytes per token. Which build you can use depends on the
> context length you want, not on which build is "best".

---

## The builds

| Build | Bytes | ~bpw | Publisher | Runtime | Mirrored? | SemIf pin? |
| --- | ---: | ---: | --- | --- | --- | --- |
| bf16 | 5,033,557,096 | 16.00 | `openbmb` (official) | Torch / CPU | yes | **yes** — desktop reference |
| GGUF F16 | 5,039,006,688 | 16.02 | `openbmb` (official) | llama.cpp | **no** | no |
| MLX 4-bit | 1,416,035,216 | 4.50 | `openbmb` (official) | MLX | yes | no |
| GGUF Q4_K_M | 1,561,318,368 | 4.96 | `openbmb` (official) | llama.cpp | yes | **yes** — wllama pin |
| MLX 8-bit | 2,674,327,290 | 8.50 | `mlx-community` | MLX | yes | no |
| GGUF Q8_0 | 2,679,710,688 | 8.52 | `openbmb` (official) | llama.cpp | **no** | no |

Bytes/param is computed against **2,516,778,548 params** (bf16 ÷ 2), so it is the
real precision of each container, not its label. Note `Q4_K_M` is ~4.96 bpw, not
4.0 — a k-quant mixes bit widths across tensors. "4-bit" and "8-bit" are labels;
the bpw column is the fact.

All are official OpenBMB builds except the MLX 8-bit one, which is community. They
are the same weights in different containers — they do not share files and no
runtime can read another's.

**The mirror ships four of the six.** `GGUF Q8_0` and `GGUF F16` are published
upstream in [`openbmb/MiniCPM5-2B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
and are **not** in `weights-v1` — download them straight from Hugging Face if you
want them. Other quantizations (Q5_K_M, Q6_K, IQ4_XS …) exist in community repos
such as `bartowski/MiniCPM5-2B-GGUF`.

**`GGUF Q8_0` is the one to notice.** It costs 1.12 GB over Q4_K_M and buys
~8.52 bpw — and unlike the MLX 8-bit build, which is the *same precision at the
same size*, it runs on `llama-server` and therefore **keeps the tool loop**
(§2 fault 1). It is not true that 8-bit and tool calling are mutually exclusive
on this model. See §1 for what it costs against the 9 GB budget.

Get them with `./assemble.sh` (see §4). The bf16 and MLX 8-bit builds are over
GitHub's 2 GiB per-file cap and ship as byte-range parts; the other two ship whole.

---

## §1 — The 9 GB budget

MiniCPM5-2B is `LlamaForCausalLM`, `model_type: llama`, 42 layers, 16 query
heads and **2 KV heads** at head_dim 128. Because *every* layer is full
attention, the cache cost is:

```
42 layers x 2 (K and V) x 2 kv_heads x 128 head_dim x 2 bytes (fp16) = 43,008 B/token
```

| Context | KV cache alone |
| ---: | ---: |
| 8 K | 352 MB |
| 32 K | 1.41 GB |
| 64 K | 2.82 GB |
| 128 K (max) | **5.64 GB** |

Weights plus KV, against 9 GB:

| Build | Weights | @ 32 K | @ 64 K | @ 128 K |
| --- | ---: | ---: | ---: | ---: |
| MLX 4-bit | 1.42 GB | **2.83 GB** | 4.24 GB | **7.06 GB** |
| GGUF Q4_K_M | 1.56 GB | 2.97 GB | 4.38 GB | 7.20 GB |
| MLX 8-bit | 2.67 GB | 4.08 GB | 5.49 GB | 8.31 GB |
| GGUF Q8_0 | 2.68 GB | 4.09 GB | 5.50 GB | 8.32 GB |
| bf16 | 5.03 GB | 6.44 GB | 7.85 GB | **10.67 GB — does not fit** |

Allow roughly 0.6 GB for the Python runtime, MLX's allocator cache, and macOS
overhead on top of these. That puts the practical line at:

- **4-bit at 128 K is the only build with real headroom.** ~7.7 GB of 9 GB.
- **8-bit at 128 K is marginal** — ~8.9 GB of 9 GB, and this is true of **both**
  8-bit builds: MLX 8-bit and GGUF Q8_0 differ by 5 MB, so they land in the same
  place. Expect pressure.
- **bf16 fits only below ~64 K**, and then with almost nothing left over.
- At 32 K, every build fits. The choice only matters if you want long context.
- **8-bit is comfortable to ~64 K** (5.50 GB) and only becomes marginal past it.
  Combined with the KV-cache lever below, Q8_0 at 128 K comes back to ~5.5 GB.

Two levers, and they are different on each runtime:

- **MLX** — `--max-kv-size n` uses a rotating KV cache. It bounds memory but
  discards older context, so quality degrades for long prompts. There is no KV
  quantization in MLX. (`vendor/MLX-LM-README.md`, "Long Prompts and Generations")
- **llama.cpp** — quantize the cache itself, which keeps the whole context at
  half or quarter precision. Confirm the exact flag on your build with
  `llama-server --help | grep cache-type` before relying on it; it is not in the
  vendored llama.cpp docs.

**Decode speed tracks weight size** — it is bandwidth-bound. Relative to 4-bit,
8-bit is roughly **1.9x slower** and bf16 roughly **3.5x slower**. This matters
for chat and not at all for SemIf, which never decodes (§3).

---

## §2 — Route A: chat, via MLX

The shortest path to a chat endpoint. No compiler, no GGUF conversion.

```bash
python3 -m venv ~/.venvs/minicpm && source ~/.venvs/minicpm/bin/activate
pip install mlx-lm
```

`mlx_lm.server` is the OpenAI-compatible server (`vendor/MLX-LM-README.md`).
Point it at the assembled 4-bit build:

```bash
mlx_lm.server --model ./mlx-4bit --port 8080
```

For long context, bound the cache rather than letting it grow to 128 K:

```bash
mlx_lm.server --model ./mlx-4bit --port 8080 \
  --max-kv-size 32768 \
  --prefill-step-size 1024
```

`--prefill-step-size` lowers *peak* memory while reading a long prompt, at some
cost in prefill speed. The default is 2048.

### ⚠️ Two server faults that look like a stupid model

Run **both** checks before judging output quality. Each failure produces a symptom
that reads as "this model is bad" when the fault is in the server, not the weights.

#### 1. `mlx_lm` cannot parse MiniCPM's tool calls or thinking blocks

**Evidence, from a real run** with the model loaded and a tool-using client:

```
00:37:35,055 - WARNING - Received tools but model does not support tool calling.
```

That warning lives *inside* the `if tokenizer.has_chat_template:` branch
(`server.py:543`) — so it also proves the **chat template IS loading** and
`has_chat_template` is **True**. The template is not the problem, and neither is
`tokenizer_config.json`'s missing `chat_template` key: `mlx_lm` resolves the
standalone `chat_template.jinja` correctly.

The real cause is that `mlx_lm` identifies a model's tool format by
**string-matching the chat template** (`tokenizer_utils.py:619
_infer_tool_parser`). Its entire registry:

```
"<minimax:tool_call>"                     -> minimax_m2
"<|tool_call>" + "<tool_call|>"           -> gemma4
"<start_function_call>"                   -> function_gemma
"<longcat_tool_call>"                     -> longcat
"<arg_key>"                               -> glm47
"<|tool_list_start|>" / "<|tool_call_start|>" -> pythonic
"<tool_call>\n<function="                 -> qwen3_coder
"<|tool_calls_section_begin|>"            -> kimi_k2
"[TOOL_CALLS]"                            -> mistral
"<tool_call>" + "tool_call.name"          -> json_tools
                                          -> else: return None
```

MiniCPM5-2B emits `<function name="bash">…<param name="command">…`. **Nothing in
that list matches** — note `qwen3_coder` needs a literal `<tool_call>` wrapper and
`<function=` with no space, neither of which MiniCPM produces. So
`_infer_tool_parser` returns `None`, `tool_parser` is `None`, and
`has_tool_calling` is False (`tokenizer_utils.py:484`). `mlx_lm/tool_parsers/`
ships thirteen parsers — `function_gemma`, `gemma4`, `glm47`, `json_tools`,
`kimi_k2`, `kimi_k3`, `laguna`, `longcat`, `minimax_m2`, `mistral`, `pythonic`,
`qwen3_coder` — and **`grep -i minicpm` across `mlx_lm` returns zero hits.**

**Assert:**
- tool calls arrive as **raw text** in the message body instead of structured
  `tool_calls` — you see the literal `<function name=…><param name=…>` markup
  printed as content;
- reasoning leaks into the visible response instead of a thinking block
  (`has_thinking`, `tokenizer_utils.py:448`, is likewise False);
- the model appears to ramble about its own capabilities.

**Do not try to fix this by editing `tokenizer_config.json`.** You *can* force the
parser — `tokenizer_utils.py:709` reads a `tool_parser_type` key and imports
`mlx_lm.tool_parsers.<type>` — but **no shipped parser matches MiniCPM's syntax**,
so you would silently mis-parse tool calls into the wrong shape, which is worse
than leaving them as text. There is no supported route to tool calling for this
model on `mlx_lm.server`, and it is not a bug you can configure around.

**Use Route B (§3).** `llama-server` applies the template from GGUF metadata and
runs its own tool-call handling, so it is not limited to mlx_lm's parser registry.
For an agent harness this is the difference between a working tool loop and none.

#### 2. `GET /v1/models` returns 500 without a Hugging Face cache

`handle_models_request` calls `scan_cache_dir()` unguarded (`server.py:1690`;
`huggingface_hub` raises `CacheNotFound` at `utils/_cache_manager.py:777`). On a
machine that has never had an HF cache this endpoint 500s, and any client that
probes it for readiness — **OMP does** — hangs on "starting" indefinitely.

```bash
mkdir -p "$(python3 -c 'from huggingface_hub import constants as c; print(c.HF_HUB_CACHE)')"
```

`scan_cache_dir()` reads **`HF_HUB_CACHE`**, which derives from `HF_HOME` — *not*
`~/.cache/huggingface/hub`. If `HF_HOME` is exported, `mkdir -p
~/.cache/huggingface/hub` is a silent no-op. Ask the library for the path, above.

This is the same endpoint the OMP `id` check below uses. If that `curl` hangs or
500s, this is why.

**Known-good workaround, confirmed in practice:** start the server with offline
mode on and the endpoint stops failing, with no `mkdir` needed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 mlx_lm.server --model ./mlx-8bit --port 7777
```

This was observed to work on 2026-09-20 (macOS, Homebrew Python 3.14, mlx-lm
current). **The mechanism is not established** — `scan_cache_dir()` is documented
to raise `CacheNotFound` on a missing directory whether or not offline mode is set,
so offline mode is doing something indirect here. Treat it as an empirical
workaround, not an explanation, and keep the `mkdir` form above as the documented
one. Offline mode is worth setting regardless for a local-only server.

**Route B sidesteps this entirely** — `llama-server` has no Hugging Face dependency
at all. It is also the only route with usable tool calling (§2 fault 1).

### Wiring it into OMP

OMP is a coding agent, not an inference runtime — it talks to a server over
OpenAI-compatible HTTP. Its config is `~/.omp/agent/models.yml` (it falls back to
`.yaml`), and the root object contains **only** `providers`
(`vendor/OMP-MODELS.md`).

OMP has *implicit* discovery for `ollama`, `lm-studio`, `llama.cpp` and `vllm`.
There is **no implicit provider for MLX**, so an MLX server needs an explicit
provider with the model listed by hand:

```yaml
providers:
  minicpm-mlx:
    baseUrl: http://127.0.0.1:8080/v1
    api: openai-completions
    auth: none
    models:
      - id: /absolute/path/to/mlx-4bit
        name: MiniCPM5-2B MLX 4-bit
        contextWindow: 131072
        maxTokens: 8192
```

The `id` must match what the server reports at `GET /v1/models` — `mlx_lm.server`
uses the model path as the id, so read it back rather than guessing:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

---

## §3 — Route B: chat, via llama.cpp and GGUF

No Python, no virtualenv. The GGUF carries its own tokenizer and chat template.

```bash
brew install llama.cpp          # vendor/LLAMA-CPP-INSTALL.md
llama-server -m ./gguf/MiniCPM5-2B-Q4_K_M.gguf \
  --port 8080 -ngl 99 -c 32768
```

`-ngl 99` offloads every layer to Metal. `-c` sets the context; this is the knob
that controls the KV cache in §1. Lower it and the cache shrinks linearly.

### Wiring it into OMP

`llama.cpp` **does** have implicit discovery, so if the server is on the default
port you may need no config at all — OMP looks at `LLAMA_CPP_BASE_URL` then
`http://127.0.0.1:8080`. To be explicit, this is the schema from the official
doc:

```yaml
providers:
  llama.cpp:
    baseUrl: http://127.0.0.1:8080
    api: openai-responses
    auth: none
    discovery:
      type: llama.cpp
```

Note the official example pairs llama.cpp with `api: openai-responses`, not
`openai-completions` — that is upstream's choice, not a typo here. If your client
wants `/v1/chat/completions` specifically, declare a custom provider with
`api: openai-completions` and an explicit `models:` list as in §2.

---

## §4 — Route C: SemIf (scoring, not chat)

SemIf is not a chat runtime. It runs **one forward pass** and reads the logits at
a fixed answer token. It never decodes. It is not an alternative to §2 and §3 —
it answers a different question.

### Install

```bash
git clone https://github.com/TheoLeeCJ/SemIf && cd SemIf
pip install -e '.[test,mlx]'
```

SemIf requires **MLX-LM at commit `a63e24c389382619eb6d9af656e3b46024be217a`**
(package 0.32.0), with MLX pinned to 0.32.2. The released **0.31.3 is not
acceptable** — it applies the L2 epsilon incorrectly
(`vendor/SEMIF-MLX.md`). Install from the commit, not from PyPI:

```bash
pip install "mlx-lm @ git+https://github.com/ml-explore/mlx-lm@a63e24c389382619eb6d9af656e3b46024be217a"
```

### ⚠️ The MLX backend refuses MiniCPM5-2B

This is the finding that decides the whole SemIf question on this machine.
`src/semif_phase1/mlx_backend.py` gates the model at load:

```python
if config.get("model_file") or config.get("model_type") not in {"qwen3_5"}:
    raise ValueError("MLX backend supports native Qwen3.5 text scoring only; "
                     "custom model code is not allowed")
```

MiniCPM5-2B's config says `model_type: llama` and
`architectures: ["LlamaForCausalLM"]`. **It fails this check and raises before a
single weight is read.** This is a hard, deterministic refusal, not a warning.

Scope this precisely, because the opposite conclusion is easy and wrong:

- It is **not** "SemIf cannot run MiniCPM5-2B." SemIf pins MiniCPM5-2B twice —
  bf16 as the desktop reference and GGUF Q4_K_M for its browser ladder
  (`vendor/SEMIF-MODELS.json`).
- It is **not** "MLX cannot run MiniCPM5-2B." MLX runs it fine — that is §2.
- It is exactly this: **SemIf's *MLX backend* serves only `qwen3_5`.** The
  restriction is on that one entry point.

### What follows from that

Every MLX result SemIf publishes — including the bf16/q4/q8 precision sweep and
the `--mlx-bits` in-memory quantization this repo's earlier notes carried — was
produced on **`Qwen/Qwen3.5-4B`**, not on MiniCPM5-2B. Those numbers do not
transfer to this model, and `--mlx-bits 8` is not a MiniCPM5-2B option.

So on a Mac there is no fast SemIf path for MiniCPM5-2B. The MLX backend refuses
it, and the Torch backend — which is the default and does support it — runs on
CPU here, because SemIf's Torch path targets CUDA. It will work and it will be
slow. If your goal is SemIf *specifically*, the model it is built and benchmarked
around on Apple Silicon is **Qwen3.5-4B**; use MiniCPM5-2B with SemIf only if you
want the bf16 reference numbers and can afford the wait.

The `--mlx-cache-limit-mib` flag (default 256 MiB, and it bounds the MLX
*allocator* cache, not the KV cache) is a real and useful lever — it just applies
to the Qwen3.5-4B runs, not to this model.

---

## §5 — Traps

Each of these is a wrong conclusion this setup reliably produces.

1. **"8-bit is the best quality, so use it."** At 128 K, 8-bit needs ~8.9 GB of
   your 9 GB. Quality is not the only axis — see §1.
2. **"SemIf's MLX backend supports this model."** It does not; §4. Reading
   SemIf's MLX docs without reading the gate in `mlx_backend.py` produces this.
3. **"SemIf can't use MiniCPM5-2B at all."** Also wrong. Scope the refusal to the
   MLX entry point. It is pinned for the Torch path and the browser path.
4. **"The MLX 8-bit build is the SemIf 8-bit build."** SemIf quantizes in memory
   from the bf16 pin. A separate 8-bit artifact is a chat-path convenience, not a
   SemIf input.
5. **"Any of these containers can read another's files."** They cannot. Each
   variant directory needs its own `config.json` and tokenizer alongside the
   reassembled weights.
6. **"`mlx_lm.server` needs a `/v1` in `--model`."** It does not, and OMP's
   `baseUrl` does. Those are two different places.
7. **"This 2B model is just bad at tool calling."** Partly a capability ceiling,
   but the *reason* you see raw `<function name=…>` markup is §2 fault 1:
   `mlx_lm` has no tool parser matching MiniCPM's syntax, so tool calls reach the
   client as text no matter how well the model emits them. The chat template is
   fine and is not the cause — do not go editing `tokenizer_config.json`. If you
   need a real tool loop, use Route B.
8. **"I ran `mkdir ~/.cache/huggingface/hub` and `/v1/models` still 500s."** It
   reads `HF_HUB_CACHE`, which follows `HF_HOME`. If `HF_HOME` is exported, that
   mkdir is a no-op. Create the path the library actually reports — §2.
9. **"llama.cpp is faster than MLX."** Maybe — but compare like with like first,
   because the obvious comparison is confounded. Decode is **bandwidth-bound**
   (§1), and the three builds are not the same size:

   | Build | Bytes |
   | --- | ---: |
   | MLX 4-bit | 1,416,035,216 |
   | GGUF Q4_K_M | 1,561,318,368 |
   | MLX 8-bit | 2,674,327,290 |

   A speedup measured against **MLX 8-bit** is mostly the **1.71x** larger weight
   stream, not the runtime. Note MLX 4-bit is the *smallest* file of the three —
   so if llama.cpp still wins the 4-bit comparison, the cause is elsewhere. The
   real runtime difference is the **KV cache**: llama.cpp can quantize it
   (`--cache-type-k` / `--cache-type-v`; confirm on your build with
   `llama-server --help | grep cache-type`) and **MLX has no KV quantization at
   all** (§1). At 128 K that is 5.64 GB of cache at fp16 — so llama.cpp's
   advantage should **grow with context length** and be smallest at short
   context. If you see the opposite, something else is going on.

---

## §6 — What came from where

| Claim in this guide | Source |
| --- | --- |
| MLX-LM install, `--max-kv-size`, `--prefill-step-size`, `mlx_lm.server` | `vendor/MLX-LM-README.md` |
| Tool-parser registry, `has_tool_calling`, `has_thinking` (§2 fault 1) | `mlx-lm` `mlx_lm/tokenizer_utils.py:619` (`_infer_tool_parser`), `:448`, `:484`; `mlx_lm/tool_parsers/` |
| The unguarded `scan_cache_dir()` (§2 fault 2) | `mlx-lm` `mlx_lm/server.py:1690`; `huggingface_hub` `utils/_cache_manager.py:777` |
| The `HF_HUB_OFFLINE=1` workaround (§2 fault 2) | **empirical** — observed 2026-09-20; mechanism *not* established |
| SemIf install, MLX-LM commit pin, 0.31.3 L2-epsilon bug, `--mlx-bits`, cache cap | `vendor/SEMIF-MLX.md` |
| The `model_type not in {"qwen3_5"}` refusal | SemIf `src/semif_phase1/mlx_backend.py` @ `ca3ba65` |
| Which revisions SemIf pins | `vendor/SEMIF-MODELS.json` |
| OMP `providers` schema, local discovery, llama.cpp/ollama examples | `vendor/OMP-MODELS.md` |
| `brew install llama.cpp` | `vendor/LLAMA-CPP-INSTALL.md` |
| Build instructions, if you ever need to compile llama.cpp | `vendor/LLAMA-CPP-BUILD.md` |
| KV bytes/token, residency tables, OMP MLX wiring, traps | **ours** — derived from the pinned `config.json`, not from an upstream doc |

The KV arithmetic is ours because no upstream document states it for this model.
Check it against the `config.json` in each variant directory: 42 layers, 2 KV
heads, head_dim 128. If those numbers move, §1 moves with them.

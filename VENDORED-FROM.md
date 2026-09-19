# Vendored from

Verbatim copies of the official documentation this mirror relies on.
Nothing under `vendor/` is edited by hand -- `scripts/vendor_docs.py`
writes these files and this list together, at pinned commits.

Every file below is byte-identical to its upstream source. Where this
repository's own guide disagrees with a vendored file, the vendored file
wins; it is the authority, not our summary of it.

| Vendored file | Upstream path | Commit | Bytes | sha256 | License |
| --- | --- | --- | ---: | --- | --- |
| `vendor/SEMIF-MLX.md` | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/docs/MLX.md) | `ca3ba65f1429` | 8,755 | `b43d451bea31f3f1998b17461b1e7b922b74c699bef109971ee7a08c4cb96e23` | Apache-2.0 (see the upstream repository) |
| `vendor/SEMIF-REPRODUCE.md` | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/docs/REPRODUCE.md) | `ca3ba65f1429` | 6,034 | `8d219dbd50e82cf54316d46252f45ec7fa8f09019aae42c61f14da9440a77726` | Apache-2.0 (see the upstream repository) |
| `vendor/SEMIF-METHOD.md` | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/docs/METHOD.md) | `ca3ba65f1429` | 5,783 | `f328e1d8d11fd801132a409d5de8c73894ddaa4789a3601d072c756f9fb940c0` | Apache-2.0 (see the upstream repository) |
| `vendor/SEMIF-MODELS.json` | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/manifests/models.json) | `ca3ba65f1429` | 1,457 | `b7286bd0a41c15e7c7d39475a33f268ec6edf11066026bb4017bb79c1ca3a8f2` | Apache-2.0 (see the upstream repository) |
| `vendor/SEMIF-requirements.txt` | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf/blob/ca3ba65f142967030ecb453346e94d6f476a69df/requirements.txt) | `ca3ba65f1429` | 181 | `bc215c87d5eed29b4c4db100d53c1d763f807fbb7164a1c47ad8d89166fff87f` | Apache-2.0 (see the upstream repository) |
| `vendor/OMP-MODELS.md` | [can1357/oh-my-pi](https://github.com/can1357/oh-my-pi/blob/836048d81e088b4cddcd023780d6d769920e8525/docs/models.md) | `836048d81e08` | 43,790 | `e997ad228ceddb2686a06344b553e9e0d212834caeefec21ea5b5fe395a887ab` | MIT (see the upstream repository) |
| `vendor/MLX-LM-README.md` | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/blob/9d1e356e7cc6549e7d1697adabe2ea01ff8e062c/README.md) | `9d1e356e7cc6` | 8,590 | `625b4478800ad2fd5d393792f7ca8d4006be9c95b725cc239a9e67b7aa891875` | MIT (see the upstream repository) |
| `vendor/LLAMA-CPP-BUILD.md` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/e613ef2c81bae98d59850d061ac29e6e3e88cb00/docs/build.md) | `e613ef2c81ba` | 42,196 | `1410862ad38bcf3e8c3133cca1c2283e00a2f2af7cfa55286374f118b79767dc` | MIT (see the upstream repository) |
| `vendor/LLAMA-CPP-INSTALL.md` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/e613ef2c81bae98d59850d061ac29e6e3e88cb00/docs/install.md) | `e613ef2c81ba` | 1,830 | `36bd37248baeca0f52222884d98cf6c3b0a7017739dd68aca1a093ce34434f52` | MIT (see the upstream repository) |

## Why these

- **SEMIF-MLX.md** -- the only official Apple Silicon / MLX document.
  Notes the MLX-LM commit pin, the in-memory `--mlx-bits` quantization,
  the 256 MiB inactive-cache cap, and that MLX reranker mode is
  unsupported.
- **SEMIF-REPRODUCE.md** -- environment and scoring commands.
- **SEMIF-METHOD.md** -- what the method is, and what it is not.
- **SEMIF-MODELS.json** -- the frozen revision of every model SemIf pins.
  This is the authoritative answer to "which build does SemIf use".
- **SEMIF-requirements.txt** -- the Python pins for the Torch path.
- **OMP-MODELS.md** -- the `models.yml` schema for pointing OMP at a
  local OpenAI-compatible server.
- **MLX-LM-README.md** -- install, generate, chat, convert.
- **LLAMA-CPP-BUILD.md** -- building llama.cpp, including the macOS
  section. The GGUF path needs no build (a prebuilt release works), but
  this is the authority if one is ever needed.
- **LLAMA-CPP-INSTALL.md** -- the packaged install routes.

Regenerate with `python3 scripts/vendor_docs.py`.

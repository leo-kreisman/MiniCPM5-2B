#!/usr/bin/env python3
"""Vendor the official third-party documentation this mirror depends on.

The point of vendoring is that the Mac workstation can read the *authoritative*
instructions without network access and without an agent paraphrasing them. So
these are copied byte-for-byte, at a pinned commit, and every file records where
it came from and what it hashes to.

Do not hand-edit anything under vendor/. If a source moves, re-run this script
and let VENDORED-FROM.md record the new pin. A paraphrase of a loader instruction
is how the sibling Bonsai repos ended up with docs that described the wrong
loader; verbatim plus provenance is the fix.

Usage:  python3 scripts/vendor_docs.py [--check]
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "vendor")

# repo -> ref (branch); the exact commit is resolved at run time and pinned in
# VENDORED-FROM.md, so a later push to the branch cannot silently change what
# this mirror vendored.
SOURCES = [
    ("TheoLeeCJ/SemIf", "master", "docs/MLX.md",          "SEMIF-MLX.md"),
    ("TheoLeeCJ/SemIf", "master", "docs/REPRODUCE.md",    "SEMIF-REPRODUCE.md"),
    ("TheoLeeCJ/SemIf", "master", "docs/METHOD.md",       "SEMIF-METHOD.md"),
    ("TheoLeeCJ/SemIf", "master", "manifests/models.json", "SEMIF-MODELS.json"),
    ("TheoLeeCJ/SemIf", "master", "requirements.txt",     "SEMIF-requirements.txt"),
    ("can1357/oh-my-pi", "main",   "docs/models.md",      "OMP-MODELS.md"),
    ("ml-explore/mlx-lm", "main",  "README.md",           "MLX-LM-README.md"),
    ("ggml-org/llama.cpp", "master", "docs/build.md",     "LLAMA-CPP-BUILD.md"),
    ("ggml-org/llama.cpp", "master", "docs/install.md",   "LLAMA-CPP-INSTALL.md"),
]

LICENSE_NOTE = {
    "TheoLeeCJ/SemIf": "Apache-2.0 (see the upstream repository)",
    "can1357/oh-my-pi": "MIT (see the upstream repository)",
    "ml-explore/mlx-lm": "MIT (see the upstream repository)",
    "ggml-org/llama.cpp": "MIT (see the upstream repository)",
}


def api(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "mirror/1", "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def resolve_commit(repo: str, ref: str) -> str:
    return api(f"https://api.github.com/repos/{repo}/commits/{ref}")["sha"]


def fetch_blob(repo: str, commit: str, path: str) -> bytes:
    d = api(f"https://api.github.com/repos/{repo}/contents/{path}?ref={commit}")
    return base64.b64decode(d["content"])


def main() -> int:
    check = "--check" in sys.argv
    os.makedirs(OUT, exist_ok=True)

    pins: dict[tuple[str, str], str] = {}
    rows = []
    failed = []

    for repo, ref, path, dest in SOURCES:
        key = (repo, ref)
        if key not in pins:
            pins[key] = resolve_commit(repo, ref)
            print(f"pin {repo}@{ref} -> {pins[key]}", flush=True)
        commit = pins[key]

        try:
            data = fetch_blob(repo, commit, path)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {repo}/{path}: {e}", flush=True)
            failed.append(f"{repo}/{path}")
            continue

        digest = hashlib.sha256(data).hexdigest()
        target = os.path.join(OUT, dest)
        existing = None
        if os.path.exists(target):
            with open(target, "rb") as fh:
                existing = hashlib.sha256(fh.read()).hexdigest()

        verb = "verify" if check else "write"
        if check:
            state = "ok" if existing == digest else "MISMATCH"
            print(f"  {state:8s} {dest}  {len(data):>7,} B  {digest[:16]}")
            if existing != digest:
                failed.append(dest)
            rows.append((repo, commit, path, dest, len(data), digest,
                         LICENSE_NOTE.get(repo, "see upstream")))
            continue

        if existing == digest:
            print(f"  [same]   {dest}", flush=True)
        else:
            with open(target, "wb") as fh:
                fh.write(data)
            print(f"  [{verb}] {dest}  {len(data):,} B", flush=True)

        rows.append((repo, commit, path, dest, len(data), digest,
                     LICENSE_NOTE.get(repo, "see upstream")))

    if not check:
        lines = [
            "# Vendored from",
            "",
            "Verbatim copies of the official documentation this mirror relies on.",
            "Nothing under `vendor/` is edited by hand -- `scripts/vendor_docs.py`",
            "writes these files and this list together, at pinned commits.",
            "",
            "Every file below is byte-identical to its upstream source. Where this",
            "repository's own guide disagrees with a vendored file, the vendored file",
            "wins; it is the authority, not our summary of it.",
            "",
            "| Vendored file | Upstream path | Commit | Bytes | sha256 | License |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
        for repo, commit, path, dest, size, digest, lic in rows:
            link = f"[{repo}](https://github.com/{repo}/blob/{commit}/{path})"
            lines.append(f"| `vendor/{dest}` | {link} | `{commit[:12]}` | {size:,} | `{digest}` | {lic} |")
        lines += [
            "",
            "## Why these",
            "",
            "- **SEMIF-MLX.md** -- the only official Apple Silicon / MLX document.",
            "  Notes the MLX-LM commit pin, the in-memory `--mlx-bits` quantization,",
            "  the 256 MiB inactive-cache cap, and that MLX reranker mode is",
            "  unsupported.",
            "- **SEMIF-REPRODUCE.md** -- environment and scoring commands.",
            "- **SEMIF-METHOD.md** -- what the method is, and what it is not.",
            "- **SEMIF-MODELS.json** -- the frozen revision of every model SemIf pins.",
            "  This is the authoritative answer to \"which build does SemIf use\".",
            "- **SEMIF-requirements.txt** -- the Python pins for the Torch path.",
            "- **OMP-MODELS.md** -- the `models.yml` schema for pointing OMP at a",
            "  local OpenAI-compatible server.",
            "- **MLX-LM-README.md** -- install, generate, chat, convert.",
            "- **LLAMA-CPP-BUILD.md** -- building llama.cpp, including the macOS",
            "  section. The GGUF path needs no build (a prebuilt release works), but",
            "  this is the authority if one is ever needed.",
            "- **LLAMA-CPP-INSTALL.md** -- the packaged install routes.",
            "",
            "Regenerate with `python3 scripts/vendor_docs.py`.",
            "",
        ]
        with open(os.path.join(ROOT, "VENDORED-FROM.md"), "w") as fh:
            fh.write("\n".join(lines))
        print("wrote VENDORED-FROM.md", flush=True)

    if failed:
        print(f"\n{len(failed)} problem(s): {failed}", flush=True)
        return 1
    print("\nall vendored files match", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

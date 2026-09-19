#!/usr/bin/env python3
"""Turn the verified staged downloads into release parts + an assemble.sh that pins them.

This is the single place that decides part boundaries, asset names, and expected
hashes. It runs after `fetch_sources.py` has downloaded everything and verified
each file against the sha256 Hugging Face publishes, so the whole-file hashes it
works from are already known-good.

It produces, in PARTS (uploaded as release assets):

    MiniCPM5-2B-bf16.safetensors.part-{0,1,2}       5,033,557,096 B, split 3 ways
    MiniCPM5-2B-MLX-8bit.safetensors.part-{0,1}     2,674,327,290 B, split 2 ways
    MiniCPM5-2B-MLX-4bit.safetensors                1,416,035,216 B, under the cap
    MiniCPM5-2B-Q4_K_M.gguf                         1,561,318,368 B, under the cap
    MANIFEST.sha256                                 every asset + every rebuilt file

and in the repository, one directory per build holding that build's config and
tokenizer, because the containers are not interchangeable -- an MLX directory
cannot be read by llama.cpp, and the GGUF carries its own tokenizer.

Two mappings matter and they are deliberately separate: ASSETS maps an asset
name to the file it is cut from, and each entry names the TARGET path that asset
reassembles into. Conflating them is how a splitter produces parts that verify
individually but concatenate into a file at the wrong path with the wrong name.

It refuses to run if a staged file disagrees with the hash recorded in
SOURCES.json. Never relax that check to make a build proceed: the expected hashes
pin the published bytes, and editing one to match a bad download converts a loud
failure into a model that loads and silently produces wrong output.

Usage:  python3 scripts/build_release.py [--check]
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys

STAGING = os.environ.get("DEST", "/home/scribe/model-staging/MiniCPM5-2B")
SOURCES = os.path.join(STAGING, "SOURCES.json")
PARTS = os.environ.get("PARTS", "/home/scribe/model-parts/MiniCPM5-2B")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GitHub rejects any single file at or over 2 GiB, for Release assets and for Git
# LFS alike, on every plan. Stay clear of the boundary rather than at it.
CAP = 2 * 1024 ** 3
CHUNK = 1 << 22

# asset stem, ext, source repo, source path, part count, reassembled target path
ASSETS = [
    ("MiniCPM5-2B-bf16", ".safetensors", "openbmb/MiniCPM5-2B",
     "model-00000-of-00001.safetensors", 3,
     "bf16/model-00000-of-00001.safetensors"),
    ("MiniCPM5-2B-MLX-8bit", ".safetensors", "mlx-community/MiniCPM5-2B-8bit",
     "model.safetensors", 2, "mlx-8bit/model.safetensors"),
    ("MiniCPM5-2B-MLX-4bit", ".safetensors", "openbmb/MiniCPM5-2B-MLX",
     "model.safetensors", 1, "mlx-4bit/model.safetensors"),
    ("MiniCPM5-2B-Q4_K_M", ".gguf", "openbmb/MiniCPM5-2B-GGUF",
     "MiniCPM5-2B-Q4_K_M.gguf", 1, "gguf/MiniCPM5-2B-Q4_K_M.gguf"),
]

# repository build directory -> source repo whose config/tokenizer it needs
BUILDS = {
    "bf16": "openbmb/MiniCPM5-2B",
    "mlx-4bit": "openbmb/MiniCPM5-2B-MLX",
    "mlx-8bit": "mlx-community/MiniCPM5-2B-8bit",
}

# The GGUF needs none of these: a GGUF carries its own tokenizer and template.
KEEP = {
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "model.safetensors.index.json",
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def staged(repo: str, path: str) -> str:
    return os.path.join(STAGING, repo.replace("/", "__"), path)


def asset_names(stem: str, ext: str, parts: int) -> list[str]:
    if parts == 1:
        return [f"{stem}{ext}"]
    return [f"{stem}{ext}.part-{i}" for i in range(parts)]


def split(src: str, stem: str, ext: str, parts: int) -> list[tuple[str, str, int]]:
    """Cut src into `parts` pieces under PARTS. Returns [(asset_name, sha256, size)]."""
    size = os.path.getsize(src)
    part_size = math.ceil(size / parts)
    if part_size >= CAP:
        raise SystemExit(
            f"{os.path.basename(src)} is {human(size)}; {parts} parts would be "
            f"{human(part_size)} each, at or over the 2 GiB cap. Raise the count.")

    made = []
    names = asset_names(stem, ext, parts)
    with open(src, "rb") as fh:
        for i, name in enumerate(names):
            dest = os.path.join(PARTS, name)
            want = min(part_size, size - i * part_size)
            h = hashlib.sha256()
            with open(dest, "wb") as o:
                left = want
                while left:
                    b = fh.read(min(CHUNK, left))
                    if not b:
                        raise SystemExit(f"short read on {src} at part {i}")
                    o.write(b)
                    h.update(b)
                    left -= len(b)
            made.append((name, h.hexdigest(), want))
            print(f"    [part]   {name:<44} {human(want):>10}  {h.hexdigest()[:16]}",
                  flush=True)
    return made


def emit_assembler(assets: list[tuple[str, str]], targets: list[tuple[str, str, list[str]]]) -> None:
    """assets: [(asset_name, sha256)]  targets: [(path, sha256, [asset names])]

    The rows are emitted bare and unindented: no surrounding quotes, no leading
    whitespace. They are interpolated into a double-quoted multi-line assignment
    in the generated script, so a literal `"` on any row would close that
    assignment early and turn the rest of the data into stray commands, and any
    leading spaces would be read into the first field by `read -r`. Both failures
    pass `bash -n` -- they are valid syntax and only break at run time -- so they
    are caught by the fixture test in DEVELOPMENT_LOG.md, not by a syntax check.
    """
    rows = "\n".join(f"{n}:{d}" for n, d in assets)
    trows = "\n".join(f'{p}|{d}|{" ".join(names)}' for p, d, names in targets)

    body = f'''#!/usr/bin/env bash
#
# Fetch, verify and assemble every MiniCPM5-2B build this repository ships.
#
#   ./assemble.sh            fetch what is missing, verify, assemble
#   ./assemble.sh --check    verify what is here; download and write nothing
#
# Resumable: an asset already present and correct is skipped, and one that fails
# its checksum is deleted and re-fetched.
#
# The expected hashes below are generated by scripts/build_release.py from
# downloads that were themselves verified against the sha256 Hugging Face
# publishes. They are not hand-maintained. Re-run that script after any change
# to the release layout.
#
# DO NOT edit an expected hash to match a download. An asset that fails its
# checksum is a bad download, not a wrong hash, and relaxing the check turns a
# loud, catchable failure into a model that loads and silently produces wrong
# output.
#
set -euo pipefail

REPO="leo-kreisman/MiniCPM5-2B"
TAG="weights-v1"
BASE="https://github.com/${{REPO}}/releases/download/${{TAG}}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PARTS="$ROOT/.parts"
CHECK=0
[ "${{1:-}}" = "--check" ] && CHECK=1

say()  {{ printf '\\n==> %s\\n' "$*"; }}
note() {{ printf '    %s\\n' "$*"; }}
die()  {{ printf '\\n[STOP] %s\\n' "$*" >&2; exit 1; }}

# asset-name:sha256, one per line
ASSETS_SHA="
{rows}
"

# target-path|sha256|asset names in concatenation order
TARGETS="
{trows}
"

sha_of() {{
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}}

fetch() {{
  name="$1"; want="$2"
  dest="$PARTS/$name"
  if [ -f "$dest" ] && [ "$(sha_of "$dest")" = "$want" ]; then
    note "skip     $name"
    return 0
  fi
  if [ "$CHECK" -eq 1 ]; then
    die "$name is missing or corrupt (--check makes no changes)."
  fi
  rm -f "$dest"
  note "fetch    $name"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$dest" "$BASE/$name" || die \\
"download failed: $name
  A 404 here means the release asset is missing. Check:
    https://github.com/${{REPO}}/releases/tag/${{TAG}}"
  got="$(sha_of "$dest")"
  [ "$got" = "$want" ] || die "$name failed its checksum.
  expected $want
  actual   $got
  That is a bad download, not a wrong hash. Delete it and re-run; do not edit
  the expected value in this script."
  note "ok       $name"
}}

say "1/3  fetching assets"
mkdir -p "$PARTS"
while IFS=: read -r n h; do
  [ -n "$n" ] || continue
  fetch "$n" "$h"
done <<< "$ASSETS_SHA"

say "2/3  assembling"
while IFS='|' read -r target h names; do
  [ -n "$target" ] || continue
  out="$ROOT/$target"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(sha_of "$out")" = "$h" ]; then
    note "skip     $target"
    continue
  fi
  if [ "$CHECK" -eq 1 ]; then
    die "$target is missing or corrupt (--check makes no changes)."
  fi
  note "assemble $target"
  : > "$out"
  for n in $names; do
    cat "$PARTS/$n" >> "$out"
  done
  got="$(sha_of "$out")"
  [ "$got" = "$h" ] || die "$target failed its checksum after assembly.
  expected $h
  actual   $got
  Every asset verified individually, so the concatenation order is wrong or an
  asset belongs to a different build. Do not serve this file."
  note "ok       $target"
done <<< "$TARGETS"

say "3/3  ready"
cat <<'EOF'

  The assembled weights are gitignored and are what SETUP.md expects to find:
    bf16/      mlx-4bit/      mlx-8bit/      gguf/

  SETUP.md section 1 for which build fits your memory; sections 2-4 for how to
  serve it. The assets under .parts/ can be deleted once assembly has verified.
EOF
'''

    path = os.path.join(REPO, "assemble.sh")
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o755)
    print(f"    {path}  ({len(body):,} B, {len(assets)} assets, {len(targets)} targets)")


def main() -> int:
    check = "--check" in sys.argv
    with open(SOURCES) as fh:
        sources = json.load(fh)
    by_repo = {r: {f["path"]: f for f in i["files"]} for r, i in sources.items()}

    # ---------------------------------------------------------- 1. verify
    print("==> verifying staged sources against the pinned hashes")
    digests: dict[tuple[str, str], tuple[str, int]] = {}
    for stem, ext, repo, path, parts, target in ASSETS:
        src = staged(repo, path)
        if not os.path.exists(src):
            raise SystemExit(f"missing staged file: {src}\nRun fetch_sources.py first.")
        want = by_repo[repo][path]
        size = os.path.getsize(src)
        if size != want["size"]:
            raise SystemExit(
                f"{path}: {size:,} B on disk but {want['size']:,} B pinned. "
                "Re-run fetch_sources.py; do not proceed.")
        digest = sha256_of(src)
        if want.get("sha256") and digest != want["sha256"]:
            raise SystemExit(
                f"{path}: sha256 {digest} does not match the published "
                f"{want['sha256']}. This is a bad download, NOT a wrong hash. "
                "Re-run fetch_sources.py. Do not edit SOURCES.json to match it.")
        digests[(repo, path)] = (digest, size)
        print(f"    [ok]     {stem:<28} {human(size):>10}  {digest[:16]}")

    if check:
        print("\n--check: all staged weights verified; nothing written.")
        return 0

    # ---------------------------------------------------------- 2. split
    print(f"\n==> writing assets to {PARTS}")
    os.makedirs(PARTS, exist_ok=True)
    assets: list[tuple[str, str]] = []
    targets: list[tuple[str, str, list[str]]] = []

    for stem, ext, repo, path, parts, target in ASSETS:
        digest, size = digests[(repo, path)]
        names = asset_names(stem, ext, parts)
        if parts == 1:
            shutil.copyfile(staged(repo, path), os.path.join(PARTS, names[0]))
            print(f"    [whole]  {names[0]:<44} {human(size):>10}  {digest[:16]}",
                  flush=True)
            assets.append((names[0], digest))
        else:
            for name, ph, _psize in split(staged(repo, path), stem, ext, parts):
                assets.append((name, ph))
        targets.append((target, digest, names))

    with open(os.path.join(PARTS, "MANIFEST.sha256"), "w") as fh:
        fh.write("# MiniCPM5-2B release assets -- sha256\n")
        fh.write("# Assets are the parts; targets are what they rebuild.\n\n")
        for name, digest in assets:
            fh.write(f"{digest}  {name}\n")
        fh.write("\n# reassembled files (not uploaded -- built by assemble.sh)\n")
        for path, digest, _names in targets:
            fh.write(f"{digest}  {path}\n")
    print(f"\n==> wrote {os.path.join(PARTS, 'MANIFEST.sha256')}")

    # ---------------------------------------------------------- 3. repo dirs
    print("\n==> writing per-build config directories")
    for build, repo in BUILDS.items():
        d = os.path.join(REPO, build)
        os.makedirs(d, exist_ok=True)
        src_dir = os.path.dirname(staged(repo, "x"))
        n = 0
        for name in sorted(os.listdir(src_dir)):
            if name in KEEP:
                shutil.copyfile(os.path.join(src_dir, name), os.path.join(d, name))
                n += 1
        print(f"    {build}/  {n} config files")
    os.makedirs(os.path.join(REPO, "gguf"), exist_ok=True)

    # ---------------------------------------------------------- 4. assembler
    print("\n==> emitting assemble.sh")
    emit_assembler(assets, targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())

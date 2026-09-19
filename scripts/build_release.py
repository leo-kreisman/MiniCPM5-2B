#!/usr/bin/env python3
"""Turn the verified staged downloads into release parts + an assemble.sh that pins them.

This is the single place that decides part boundaries and expected hashes. It
runs after `fetch_sources.py` has downloaded everything and verified each file
against the sha256 Hugging Face publishes, so the whole-file hashes it works from
are already known-good.

What it produces, in PARTS:

    MiniCPM5-2B-bf16.safetensors.part-{0,1,2}      5,033,557,096 B split 3 ways
    MiniCPM5-2B-MLX-8bit.safetensors.part-{0,1}    2,674,327,290 B split 2 ways
    MiniCPM5-2B-MLX-4bit.safetensors               1,416,035,216 B, under the cap
    MiniCPM5-2B-Q4_K_M.gguf                        1,561,318,368 B, under the cap
    MANIFEST.sha256                                every part + every whole file

and in the repository, one directory per build holding that build's config and
tokenizer, because the containers are not interchangeable -- an MLX directory
cannot be read by llama.cpp.

It refuses to run if a staged file disagrees with the hash recorded in
SOURCES.json. Never relax that check to make a build proceed: the expected
hashes pin the published bytes, and editing one to match a bad download converts
a loud failure into a model that loads and silently produces wrong output.

Usage:  python3 scripts/build_release.py [--parts DIR] [--repo DIR] [--check]
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

# local_name (in PARTS)  <-  (repo, upstream path, parts)
WEIGHTS = {
    "MiniCPM5-2B-bf16.safetensors": (
        "openbmb/MiniCPM5-2B", "model-00000-of-00001.safetensors", 3),
    "MiniCPM5-2B-MLX-8bit.safetensors": (
        "mlx-community/MiniCPM5-2B-8bit", "model.safetensors", 2),
    "MiniCPM5-2B-MLX-4bit.safetensors": (
        "openbmb/MiniCPM5-2B-MLX", "model.safetensors", 1),
    "MiniCPM5-2B-Q4_K_M.gguf": (
        "openbmb/MiniCPM5-2B-GGUF", "MiniCPM5-2B-Q4_K_M.gguf", 1),
}

# repo directory  <-  (source repo, where the reassembled weights land)
BUILDS = {
    "bf16": ("openbmb/MiniCPM5-2B", "model-00000-of-00001.safetensors"),
    "mlx-4bit": ("openbmb/MiniCPM5-2B-MLX", "model.safetensors"),
    "mlx-8bit": ("mlx-community/MiniCPM5-2B-8bit", "model.safetensors"),
}

# config/tokenizer files worth committing. The GGUF needs none of these: a GGUF
# carries its own tokenizer and chat template.
KEEP = {
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "model.safetensors.index.json",
}

# Files that must not be committed: HF marks weights for Git LFS via
# .gitattributes, and this repository deliberately avoids LFS.
DROP = {".gitattributes"}


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


def split(src: str, out: str, parts: int) -> list[tuple[str, str, int]]:
    """Cut src into `parts` byte-range pieces. Returns [(name, sha256, size)]."""
    size = os.path.getsize(src)
    part_size = math.ceil(size / parts)
    if part_size >= CAP:
        raise SystemExit(
            f"{os.path.basename(src)} is {human(size)}; {parts} parts would be "
            f"{human(part_size)} each, at or over the 2 GiB cap. Raise the count.")

    made = []
    with open(src, "rb") as fh:
        for i in range(parts):
            name = f"{os.path.basename(out)}.part-{i}" if parts > 1 else os.path.basename(out)
            dest = os.path.join(os.path.dirname(out), name)
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


def main() -> int:
    check = "--check" in sys.argv
    with open(SOURCES) as fh:
        sources = json.load(fh)

    by_repo = {r: {f["path"]: f for f in i["files"]}
               for r, i in sources.items()}

    # ---------------------------------------------------------- 1. verify
    print("==> verifying staged sources against the pinned hashes")
    whole: dict[str, tuple[str, str, int]] = {}
    for local, (repo, path, parts) in WEIGHTS.items():
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
        whole[local] = (digest, repo, size)
        print(f"    [ok]     {local:<44} {human(size):>10}  {digest[:16]}")

    if check:
        print("\n--check: all staged weights verified; nothing written.")
        return 0

    # ---------------------------------------------------------- 2. split
    print(f"\n==> splitting into {PARTS}")
    os.makedirs(PARTS, exist_ok=True)
    manifest: list[tuple[str, str]] = []
    for local, (digest, repo, size) in whole.items():
        parts = WEIGHTS[local][2]
        if parts == 1:
            dest = os.path.join(PARTS, local)
            shutil.copyfile(staged(repo, WEIGHTS[local][1]), dest)
            print(f"    [whole]  {local:<44} {human(size):>10}  {digest[:16]}", flush=True)
            manifest.append((local, digest))
        else:
            for name, ph, psize in split(staged(repo, WEIGHTS[local][1]),
                                         os.path.join(PARTS, local), parts):
                manifest.append((name, ph))
            manifest.append((f"{local}  (reassembled)", digest))

    with open(os.path.join(PARTS, "MANIFEST.sha256"), "w") as fh:
        fh.write("# MiniCPM5-2B release assets -- sha256\n")
        fh.write("# Whole-file lines are marked (reassembled); parts are not.\n")
        for name, digest in manifest:
            fh.write(f"{digest}  {name}\n")
    print(f"\n==> wrote {os.path.join(PARTS, 'MANIFEST.sha256')}")

    # ---------------------------------------------------------- 3. repo dirs
    print("\n==> writing per-build config directories")
    for build, (repo, weight_path) in BUILDS.items():
        d = os.path.join(REPO, build)
        os.makedirs(d, exist_ok=True)
        src_dir = os.path.dirname(staged(repo, weight_path))
        n = 0
        for name in sorted(os.listdir(src_dir)):
            if name in KEEP:
                shutil.copyfile(os.path.join(src_dir, name), os.path.join(d, name))
                n += 1
        print(f"    {build}/  {n} config files")
    os.makedirs(os.path.join(REPO, "gguf"), exist_ok=True)

    # ---------------------------------------------------------- 4. assembler
    print("\n==> emitting assemble.sh")
    emit_assembler(whole, manifest)
    return 0


def emit_assembler(whole: dict, manifest: list[tuple[str, str]]) -> None:
    part_lines = []
    whole_lines = []
    for name, digest in manifest:
        if name.endswith("(reassembled)"):
            whole_lines.append((name.replace("  (reassembled)", ""), digest))
        else:
            part_lines.append((name, digest))

    rows = "\n".join(f'    "{n}:{d}"' for n, d in part_lines)
    wrows = "\n".join(f'    "{n}:{d}"' for n, d in whole_lines)

    body = f'''#!/usr/bin/env bash
#
# Fetch, verify and assemble every MiniCPM5-2B build this repository ships.
#
#   ./assemble.sh            fetch what is missing, verify, assemble
#   ./assemble.sh --check    verify what is here; download and write nothing
#
# Resumable: a part already present and correct is skipped, and a part that
# fails its checksum is deleted and re-fetched.
#
# The expected hashes below are generated by scripts/build_release.py from
# downloads that were themselves verified against the sha256 Hugging Face
# publishes. They are not hand-maintained.
#
# DO NOT edit an expected hash to match a download. A part that fails its
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

# name:sha256, one per line
PARTS_SHA="
{rows}
"
WHOLE_SHA="
{wrows}
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
  [ "$CHECK" -eq 1 ] && die "$name is missing or corrupt (--check makes no changes)."
  rm -f "$dest"
  note "fetch    $name"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$dest" "$BASE/$name" \\
    || die "download failed: $name
  A 404 here means the release asset is missing -- check $BASE"
  got="$(sha_of "$dest")"
  [ "$got" = "$want" ] || die "$name failed its checksum.
  expected $want
  actual   $got
  That is a bad download, not a wrong hash. Delete it and re-run; do not edit
  the expected value in this script."
  note "ok       $name"
}}

say "1/3  downloading parts"
mkdir -p "$PARTS"
printf '%s\\n' "$PARTS_SHA" | while IFS=: read -r n h; do
  [ -n "$n" ] || continue
  fetch "$n" "$h"
done

say "2/3  assembling"
printf '%s\\n' "$WHOLE_SHA" | while IFS=: read -r target h; do
  [ -n "$target" ] || continue
  dir="$ROOT/$(dirname "$target")"
  mkdir -p "$dir"
  out="$ROOT/$target"
  if [ -f "$out" ] && [ "$(sha_of "$out")" = "$h" ]; then
    note "skip     $target"
    continue
  fi
  [ "$CHECK" -eq 1 ] && die "$target is missing or corrupt (--check)."
  note "assemble $target"
  : > "$out"
  base="$(basename "$target")"
  for p in "$PARTS/$base.part-"*; do
    [ -e "$p" ] || continue
    cat "$p" >> "$out"
  done
  got="$(sha_of "$out")"
  [ "$got" = "$h" ] || die "$target failed its checksum after assembly.
  expected $h
  actual   $got
  The parts verified individually, so this means the concatenation order is
  wrong or a part is from a different build. Do not serve this file."
  note "ok       $target"
done

say "3/3  done"
for b in bf16 mlx-4bit mlx-8bit gguf; do
  [ -d "$ROOT/$b" ] && note "$b/"
done
cat <<'EOF'

  Next: SETUP.md section 1 for which build fits your memory, sections 2-4 for
  how to serve it. The parts under .parts/ are safe to delete once assembly
  has verified; assembled weights are not, and are gitignored.
EOF
'''

    path = os.path.join(REPO, "assemble.sh")
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o755)
    print(f"    {path}  ({len(body):,} B, {len(part_lines)} parts, "
          f"{len(whole_lines)} assembled files)")


if __name__ == "__main__":
    sys.exit(main())

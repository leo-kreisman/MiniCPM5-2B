#!/usr/bin/env bash
#
# Publish the MiniCPM5-2B release parts as GitHub Release assets.
#
# There is no `gh` on this machine, so this drives the REST API directly. The
# token comes from the configured git credential helper and is never echoed.
# Re-running is safe: an asset already uploaded at the right size is skipped, so
# an interrupted upload resumes instead of starting over.
#
# Usage:
#   scripts/upload_release.sh --parts DIR [--tag weights-v1] [--dry-run]
#
set -euo pipefail

# These defaults are this repository's. If you copy this script to another
# mirror, change REPO, NAME and the body together -- a copied script whose REPO
# still points at the previous model will cheerfully upload the wrong weights
# to the wrong release, and the size/skip logic will not catch it.
REPO="${REPO:-leo-kreisman/MiniCPM5-2B}"
TAG="${TAG:-weights-v1}"
NAME="${NAME:-MiniCPM5-2B builds: bf16, MLX 4-bit, MLX 8-bit, GGUF Q4_K_M}"
BODY_FILE=""
PARTS_DIR=""
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --parts)     PARTS_DIR="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    --name)      NAME="$2"; shift 2 ;;
    --body-file) BODY_FILE="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$PARTS_DIR" ] || { echo "need --parts DIR" >&2; exit 2; }
[ -d "$PARTS_DIR" ] || { echo "no such dir: $PARTS_DIR" >&2; exit 1; }

API="https://api.github.com/repos/${REPO}"
UPLOADS="https://uploads.github.com/repos/${REPO}"

size_of() { wc -c < "$1" | tr -d ' '; }

# ------------------------------------------------------------------ token
# Read the token from the credential helper without printing it, in whole or in
# part. Nothing below echoes it, and no failure message includes it.
TOKEN=""
if [ "$DRY_RUN" -eq 0 ]; then
  TOKEN="$(printf 'protocol=https\nhost=github.com\n\n' \
    | git-credential-manager get 2>/dev/null \
    | sed -n 's/^password=//p')"
  [ -n "$TOKEN" ] || { echo "no GitHub token from credential helper" >&2; exit 1; }
fi

api() {
  curl -sS -H "Authorization: Bearer ${TOKEN}" \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" "$@"
}

# ------------------------------------------------------------------ release
release_id=""
if [ "$DRY_RUN" -eq 0 ]; then
  release_id="$(api "${API}/releases/tags/${TAG}" \
    | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)"
fi

if [ -z "$release_id" ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "==> creating release ${TAG}"
  if [ -n "$BODY_FILE" ]; then
    [ -f "$BODY_FILE" ] || { echo "no such body file: $BODY_FILE" >&2; exit 1; }
    body="$(cat "$BODY_FILE")"
  else
    read -r -d '' body <<'BODYEOF' || true
MiniCPM5-2B builds for Apple Silicon, split so no single file exceeds GitHubs 2 GiB per-file cap.

| Asset | Bytes | Runtime |
| --- | ---: | --- |
| MiniCPM5-2B-bf16.safetensors.part-0/1/2 | 5,033,557,096 | Torch |
| MiniCPM5-2B-MLX-8bit.safetensors.part-0/1 | 2,674,327,290 | MLX |
| MiniCPM5-2B-MLX-4bit.safetensors | 1,416,035,216 | MLX |
| MiniCPM5-2B-Q4_K_M.gguf | 1,561,318,368 | llama.cpp |

Concatenate each set of parts in order to reproduce the original file. MANIFEST.sha256
lists the sha256 of every part and of every reassembled file. Run ./assemble.sh in the
repository to do this automatically, with verification.

Read SETUP.md section 1 before choosing a build: this model has 42 layers of full
attention, so its KV cache is 43,008 bytes per token, and that -- not raw model quality
-- is what decides which build fits.

Mirrors openbmb/MiniCPM5-2B, openbmb/MiniCPM5-2B-MLX, mlx-community/MiniCPM5-2B-8bit
and openbmb/MiniCPM5-2B-GGUF on Hugging Face. Licences are those of the upstream
repositories.
BODYEOF
  fi

  payload="$(python3 -c '
import json,sys
print(json.dumps({"tag_name":sys.argv[1],"name":sys.argv[2],"body":sys.argv[3]}))
' "$TAG" "$NAME" "$body")"

  resp="$(api -X POST "${API}/releases" -d "$payload")"
  release_id="$(printf '%s' "$resp" | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)"
  [ -n "$release_id" ] || { echo "failed to create release: $resp" >&2; exit 1; }
  echo "    release id ${release_id}"
fi

# ------------------------------------------------------------------ upload
upload_asset() {
  file="$1"
  base="$(basename "$file")"
  size="$(size_of "$file")"

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    [dry]    %-48s %14s bytes\n' "$base" "$size"
    return 0
  fi

  have="$(api "${API}/releases/${release_id}/assets?per_page=100" \
    | python3 -c '
import json,sys
name=sys.argv[1]
for a in json.load(sys.stdin):
    if a.get("name")==name:
        print(a.get("size",0)); break
' "$base")"
  if [ "$have" = "$size" ]; then
    printf '    [skip]   %-48s already uploaded\n' "$base"
    return 0
  fi

  printf '    [upload] %-48s %14s bytes\n' "$base" "$size"
  # --upload-file streams from disk. Do NOT use --data-binary @file: curl buffers
  # that form in memory, which fails outright on multi-GB parts.
  code="$(curl -sS -o /tmp/gh_asset_resp.$$ -w '%{http_code}' \
    -X POST "${UPLOADS}/releases/${release_id}/assets?name=${base}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --upload-file "${file}")"

  if [ "$code" != "201" ]; then
    echo "    [FAIL]   ${base} -> HTTP ${code}" >&2
    head -c 400 /tmp/gh_asset_resp.$$ >&2; echo >&2
    rm -f /tmp/gh_asset_resp.$$
    return 1
  fi
  rm -f /tmp/gh_asset_resp.$$
  printf '    [ok]     %s\n' "$base"
}

echo "==> uploading from ${PARTS_DIR}"
# Every pattern this release can produce. A file matches at most one, so nothing
# is uploaded twice. Do not narrow this list: an earlier version of this script
# in a sibling repository matched only *.safetensors.part-*, which silently
# skipped the entire GGUF half and still printed a valid-looking release URL.
# A dry run that enumerates fewer files than the parts directory contains is
# the signal that the patterns are wrong.
rc=0
for pat in '*.safetensors.part-*' '*.gguf.part-*' '*.gguf' '*.safetensors'; do
  for f in "${PARTS_DIR}"/$pat; do
    [ -e "$f" ] || continue
    upload_asset "$f" || rc=1
  done
done
if [ -f "${PARTS_DIR}/MANIFEST.sha256" ]; then
  upload_asset "${PARTS_DIR}/MANIFEST.sha256" || rc=1
fi

[ "$rc" -eq 0 ] && echo "==> done: https://github.com/${REPO}/releases/tag/${TAG}"
exit "$rc"

#!/usr/bin/env bash
#
# Publish the MiniCPM5-2B release parts as GitHub Release assets.
#
# There is no `gh` on this machine, so this drives the REST API directly. The
# token comes from the configured git credential helper and is never echoed.
# Re-running is safe. See the two hazards documented at `upload_asset` -- one of
# them silently ships a broken release, and this version exists because it was
# hit for real.
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
ATTEMPTS=4

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

# Scratch space for response bodies. Deliberately NOT /tmp: on this machine /tmp
# is on the 98 GB root partition while /home is 1.6 TB, and filling / took every
# tool call down with it, including the ability to read back an error body.
SCRATCH="${SCRATCH:-/home/scribe/tmp}"
mkdir -p "$SCRATCH"
[ -w "$SCRATCH" ] || { echo "scratch dir not writable: $SCRATCH" >&2; exit 1; }

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

# ------------------------------------------------------------------ secrets
# The token must never appear in this process's argv. `ps` shows every process's
# argument list to every user on the machine, so `curl -H "Authorization: Bearer
# $TOKEN"` publishes the credential to anyone who runs it -- and this was
# observed for real: an unrelated `ps --ppid` while debugging dumped the token
# straight out of the child curl's command line. curl reads its options from a
# config file given with -K, and `-K -` reads that from stdin, which is not
# visible in /proc. Every curl here goes through cf_curl, which does exactly
# that. Do not "simplify" these back into -H flags.
#
# Usage: cf_curl [extra config lines on stdin] -- <curl args that are NOT secret>
cf_curl() {
  curl -sS --config - "$@"
}

# Authenticated GitHub API call. Any extra args are passed through to curl.
api() {
  cf_curl "$@" <<CFGEOF
header = "Authorization: Bearer ${TOKEN}"
header = "Accept: application/vnd.github+json"
header = "X-GitHub-Api-Version: 2022-11-28"
CFGEOF
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

# Print "<size> <state> <id>" for an asset by name, or "<none>".
# `state` is the field that matters, and it is not optional detail -- see below.
asset_info() {
  api "${API}/releases/${release_id}/assets?per_page=100" | python3 -c '
import json,sys
want=sys.argv[1]
for a in json.load(sys.stdin):
    if a.get("name")==want:
        print(a.get("size",0), a.get("state","?"), a.get("id",0)); break
else:
    print("none", "none", 0)
' "$1"
}

# ------------------------------------------------------------------ upload
upload_asset() {
  file="$1"
  base="$(basename "$file")"
  size="$(size_of "$file")"

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    [dry]    %-48s %14s bytes\n' "$base" "$size"
    return 0
  fi

  read -r have state aid <<<"$(asset_info "$base")"

  # HAZARD 1 -- size alone is not evidence the asset is good.
  # GitHub creates the asset record when an upload BEGINS, and reports the full
  # intended size for it while it is still incomplete (`state: starter`). A
  # check of `size == local size` therefore treats an upload that died halfway
  # as already done, skips it forever, and ships a release that looks complete
  # but is missing a part. Observed for real: a dropped connection during
  # part-1 left exactly this, at the correct reported size. Only `uploaded`
  # means the bytes are actually there.
  if [ "$state" = "uploaded" ] && [ "$have" = "$size" ]; then
    printf '    [skip]   %-48s already uploaded\n' "$base"
    return 0
  fi

  # HAZARD 2 -- a stale record blocks the re-upload by name.
  # GitHub refuses a second asset with a name that already exists, so the
  # incomplete placeholder has to be deleted first. Without this, every retry
  # fails with 422 and the release can never be repaired.
  if [ "$aid" != "0" ]; then
    printf '    [stale]  %-48s state=%s size=%s -> deleting\n' "$base" "$state" "$have"
    api -X DELETE "${API}/releases/assets/${aid}" >/dev/null \
      || { echo "    [FAIL]   could not delete stale asset ${base}" >&2; return 1; }
  fi

  attempt=1
  while [ "$attempt" -le "$ATTEMPTS" ]; do
    printf '    [upload] %-48s %14s bytes%s\n' "$base" "$size" \
      "$([ "$attempt" -gt 1 ] && echo "  (attempt ${attempt}/${ATTEMPTS})")"

    # --upload-file streams from disk. Do NOT use --data-binary @file: curl
    # buffers that form in memory, which fails outright on multi-GB parts.
    # HTTP 000 is a connection-level failure (DNS, reset, "Network is
    # unreachable"), not a rejection, and it is worth retrying; a 4xx is not.
    #
    # Everything secret (the token) and everything long (the URL, the file path)
    # goes through the stdin config, so the argv stays clean for `ps`. The
    # scratch file lives on /home: /tmp is on the small root partition here, and
    # the previous version of this script wrote there and then failed to read
    # its own error body back when the filesystem filled.
    respfile="$(mktemp "${SCRATCH}/gh_asset_resp.XXXXXX")"
    code="$(cf_curl -o "${respfile}" -w '%{http_code}' <<CFGEOF || true
url = "${UPLOADS}/releases/${release_id}/assets?name=${base}"
request = "POST"
header = "Authorization: Bearer ${TOKEN}"
header = "Content-Type: application/octet-stream"
upload-file = "${file}"
CFGEOF
)"

    if [ "$code" = "201" ]; then
      rm -f "${respfile}"
      printf '    [ok]     %s\n' "$base"
      return 0
    fi

    # A half-finished attempt leaves another placeholder behind, which must go
    # before the next try for the same reason as above.
    read -r _h2 _s2 aid2 <<<"$(asset_info "$base")"
    if [ "$aid2" != "0" ]; then
      api -X DELETE "${API}/releases/assets/${aid2}" >/dev/null || true
    fi

    echo "    [warn]   ${base} -> HTTP ${code}" >&2
    [ -s "${respfile}" ] && head -c 300 "${respfile}" >&2 && echo >&2
    rm -f "${respfile}"

    case "$code" in
      000|408|429|500|502|503|504) ;;   # transient: retry
      *) echo "    [FAIL]   ${base}: HTTP ${code} is not retryable" >&2; return 1 ;;
    esac

    attempt=$((attempt + 1))
    [ "$attempt" -le "$ATTEMPTS" ] && sleep $((attempt * 5))
  done

  echo "    [FAIL]   ${base} gave up after ${ATTEMPTS} attempts" >&2
  return 1
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

if [ "$rc" -eq 0 ]; then
  echo "==> done: https://github.com/${REPO}/releases/tag/${TAG}"
else
  echo "==> INCOMPLETE: some assets failed. Re-run this script; it repairs"
  echo "    stale placeholders and skips only assets in state=uploaded."
fi
exit "$rc"

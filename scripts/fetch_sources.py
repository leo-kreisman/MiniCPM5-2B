#!/usr/bin/env python3
"""Download the pinned MiniCPM5-2B sources from Hugging Face, with verification.

Every file is fetched at an immutable 40-character commit revision and checked
against the sha256 Hugging Face itself reports for it (the `lfs.oid` field from
the tree API, recorded in SOURCES.json). Small config/tokenizer files are not
stored as LFS objects and carry no published hash; those are only size-checked.

Why the pinned revision matters: `main` moves. A mirror that downloads from
`main` cannot be reproduced, and the hash of a re-uploaded artifact would not
match the pin recorded here.

Why the *published* hash matters, and the resolve endpoint's ETag does not:
the ETag returned by /resolve/ is a CDN cache tag, not a content digest. Using
it as an expected hash rejects every correct download. The authoritative digest
is `lfs.oid` from /api/models/<repo>/tree/<rev>. This is the exact mistake that
was made and corrected once already in the sibling Bonsai mirrors -- see
DEVELOPMENT_LOG.md.

Resumable: a partial file is continued with a Range request, and a complete
file that already matches its hash is skipped. Safe to re-run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

SOURCES = os.environ.get("SOURCES", "/home/scribe/model-staging/MiniCPM5-2B/SOURCES.json")
DEST = os.environ.get("DEST", "/home/scribe/model-staging/MiniCPM5-2B")
RETRIES = 6
CHUNK = 1 << 20


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def url_for(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def fetch(repo: str, rev: str, path: str, size: int, expect: str | None, out: str) -> bool:
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if os.path.exists(out):
        have = os.path.getsize(out)
        if have == size:
            if expect is None or sha256_of(out) == expect:
                print(f"  [skip]   {os.path.basename(out)} already complete", flush=True)
                return True
            print(f"  [redo]   {os.path.basename(out)} present but hash differs", flush=True)
            os.remove(out)
        elif have > size:
            print(f"  [redo]   {os.path.basename(out)} larger than expected", flush=True)
            os.remove(out)
        else:
            print(f"  [resume] {os.path.basename(out)} at {human(have)} of {human(size)}", flush=True)

    for attempt in range(1, RETRIES + 1):
        done = os.path.getsize(out) if os.path.exists(out) else 0
        req = urllib.request.Request(url_for(repo, rev, path), headers={"User-Agent": "mirror/1"})
        if done:
            req.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # A server that ignores Range answers 200 and restarts from zero.
                if done and resp.status == 200:
                    print("  [note]   server ignored Range; restarting this file", flush=True)
                    done = 0
                mode = "ab" if done else "wb"
                last = time.time()
                with open(out, mode) as fh:
                    while True:
                        b = resp.read(CHUNK)
                        if not b:
                            break
                        fh.write(b)
                        done += len(b)
                        if time.time() - last > 15:
                            pct = 100.0 * done / size if size else 0
                            print(f"  ...      {os.path.basename(out)} {pct:5.1f}%  {human(done)}", flush=True)
                            last = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(2 ** attempt, 60)
            print(f"  [retry]  {os.path.basename(out)} attempt {attempt}: {e}; waiting {wait}s", flush=True)
            time.sleep(wait)
            continue

        got = os.path.getsize(out)
        if got != size:
            print(f"  [short]  {os.path.basename(out)} {human(got)} != {human(size)}; retrying", flush=True)
            continue
        if expect is not None:
            actual = sha256_of(out)
            if actual != expect:
                print(f"  [BAD]    {os.path.basename(out)} sha256 mismatch", flush=True)
                print(f"           expected {expect}", flush=True)
                print(f"           actual   {actual}", flush=True)
                if attempt < RETRIES:
                    os.remove(out)
                    continue
                return False
            print(f"  [ok]     {os.path.basename(out)}  {human(size)}  sha256 verified", flush=True)
        else:
            print(f"  [ok]     {os.path.basename(out)}  {human(size)}  size only (not LFS)", flush=True)
        return True

    return False


def main() -> int:
    with open(SOURCES) as fh:
        sources = json.load(fh)

    failures = []
    total_files = sum(len(v["files"]) for v in sources.values())
    n = 0

    for repo, info in sources.items():
        rev = info["revision"]
        local = os.path.join(DEST, repo.replace("/", "__"))
        print(f"\n==> {repo} @ {rev[:12]}", flush=True)
        for f in info["files"]:
            n += 1
            print(f"[{n}/{total_files}] {f['path']}", flush=True)
            ok = fetch(repo, rev, f["path"], f["size"], f.get("sha256"),
                       os.path.join(local, f["path"]))
            if not ok:
                failures.append(f"{repo}/{f['path']}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for x in failures:
            print("   " + x)
        return 1
    print("all files downloaded and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

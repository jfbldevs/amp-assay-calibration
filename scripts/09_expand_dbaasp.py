#!/usr/bin/env python3
"""Resume a versioned catalog expansion without changing the original sample."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
_local = threading.local()
_lock = threading.Lock()
_next_request = 0.0


def fetch(pid, output, rate):
    global _next_request
    path = output / f"{pid}.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            if obj.get("id") == pid:
                return {"id": pid, "status": "cached"}
        except (ValueError, AttributeError):
            pass
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "AMP-assay-context-reproducibility/0.2"
    url = f"https://dbaasp.org/peptides/{pid}"
    error = None
    for attempt in range(5):
        with _lock:
            wait = max(0, _next_request - time.monotonic())
            _next_request = max(time.monotonic(), _next_request) + 1 / rate
        time.sleep(wait)
        try:
            response = _local.session.get(url, timeout=(15, 45))
            if response.status_code == 404:
                return {"id": pid, "status": "not_found", "url": url}
            if response.status_code == 429:
                time.sleep(min(60, float(response.headers.get("Retry-After", 10))))
                continue
            response.raise_for_status()
            obj = response.json()
            if not isinstance(obj, dict) or obj.get("id") != pid:
                raise ValueError("Response does not match requested peptide ID")
            payload = response.content
            temporary = path.with_suffix(".json.partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
            return {
                "id": pid, "status": "downloaded", "url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
            time.sleep(min(30, 2 ** attempt))
    return {"id": pid, "status": "failed", "error": error, "url": url}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    base = ROOT / "data/raw/dbaasp_cards"
    output = ROOT / "data/raw/dbaasp_expansion_2026-10-20"
    output.mkdir(exist_ok=True)
    index = json.loads((ROOT / "data/raw/dbaasp_peptide_index.json").read_text())
    original = {int(p.stem) for p in base.glob("*.json")}
    ids = sorted({int(r["id"]) for r in index} - original)
    if args.limit:
        ids = ids[:args.limit]
    (output / "target_ids.json").write_text(json.dumps(ids))
    logpath = output / "download_events.jsonl"
    counts = {}
    started = time.monotonic()
    with logpath.open("a") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, pid, output, args.requests_per_second) for pid in ids]
        for n, future in enumerate(as_completed(futures), 1):
            event = future.result()
            counts[event["status"]] = counts.get(event["status"], 0) + 1
            log.write(json.dumps(event) + "\n")
            log.flush()
            if n % 100 == 0 or n == len(ids):
                state = {"completed": n, "requested": len(ids), "counts": counts,
                         "seconds": round(time.monotonic() - started, 1)}
                (output / "download_progress.json").write_text(json.dumps(state, indent=2))
                print(json.dumps(state), flush=True)
    if counts.get("failed"):
        raise SystemExit("Some downloads failed; rerun to resume.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Start independent posterior fits while the main controller runs its benchmark.

All callers of script14 share OS file locks. The main controller can later wait
for the same specification and reuse its verified result. Failed gates are left
for the main controller's documented longer-sampling retry policy.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample","expanded"], default="expanded")
args = parser.parse_args()
output = ROOT/"results/reconstruction_2026-10-20"/args.scope
source = ROOT/"data/curated"/args.scope/"activities.csv"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

def run(subset, prior):
    command = [sys.executable,"scripts/14_fit_censored_model.py","--scope",args.scope,
               "--subset",subset,"--prior-scale",str(prior),"--warmup","1500","--draws","3000"]
    log = output/f"prefetch_{subset}_prior{prior:g}.log"
    with log.open("a") as stream:
        process = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"subset":subset,
                          "prior":prior,"pid":process.pid,"log":str(log.relative_to(ROOT))}),flush=True)
        code = process.wait()
    result = {"subset":subset,"prior":prior,"returncode":code}
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(run,s,p) for s,p in [
        ("all",1.5),("multi_publication",1.5),("exact",1.5),
        ("no_reported_ranges",1.5),("all",3.)]]
    results = [future.result() for future in as_completed(futures)]
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash, "Input changed during prefetch."
(output/"posterior_prefetch.json").write_text(json.dumps({
    "created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":source_hash,
    "runs":results,"interpretation":"Scheduling only; results require normal computational, precision and posterior checks."},indent=2))
raise SystemExit(0 if all(r["returncode"] in [0,3] for r in results) else 1)

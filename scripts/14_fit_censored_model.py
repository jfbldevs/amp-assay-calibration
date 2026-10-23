#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

# Keep CmdStan's console progress current when controllers redirect this process
# to a log file. This affects monitoring only, including future queued fits.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.censored import fit_censored
from amp_rebuild.execution_locks import posterior_lease

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
parser.add_argument("--subset", choices=["all", "exact", "multi_publication", "no_reported_ranges"], default="all")
parser.add_argument("--prior-scale", type=float, default=1.5)
parser.add_argument("--warmup", type=int, default=750)
parser.add_argument("--draws", type=int, default=750)
parser.add_argument("--adapt-delta", type=float, default=.95)
parser.add_argument("--max-treedepth", type=int, default=12)
args = parser.parse_args()
lock_directory = Path(__file__).resolve().parents[1]/".runtime_locks/censored"
with posterior_lease(lock_directory, f"{args.scope}_{args.subset}_{args.prior_scale:g}"):
    result = fit_censored(args.scope, args.subset, args.prior_scale, args.warmup, args.draws,
                          args.adapt_delta, args.max_treedepth)
if not result["diagnostics"]["passed"]:
    raise SystemExit(3)

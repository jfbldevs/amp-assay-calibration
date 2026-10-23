#!/usr/bin/env python3
"""Run the corrected study in dependency order; optional acquisition is explicit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["sample", "expanded"], default="expanded")
    parser.add_argument("--download", action="store_true",
                        help="Resume the versioned full-catalog acquisition before analysis.")
    parser.add_argument("--skip-censored", action="store_true",
                        help="Leave Bayesian inference pending; never marks the full pipeline complete.")
    args = parser.parse_args()
    os.chdir(ROOT)
    py = sys.executable
    expanded = ["--expanded"] if args.scope == "expanded" else []
    tasks = []
    if args.download:
        tasks.append([py, "scripts/09_expand_dbaasp.py"])
    tasks += [
        [py, "scripts/17_verify_publications.py", *expanded],
        [py, "scripts/10_curate_assays.py", *expanded],
        [py, "scripts/11_prepare_features.py", "--embeddings"],
        [py, "scripts/11_prepare_features.py", "--scope", args.scope, "--chemical"],
        [py, "scripts/11_prepare_features.py", "--scope", args.scope, "--clusters"],
        [py, "-m", "pytest", "-q", "tests/test_reconstruction.py",
         "--junitxml", str(ROOT/"results/reconstruction_2026-10-20"/args.scope/"tests.xml")],
        [py, "scripts/13_corrected_statistics.py", "--scope", args.scope],
        [py, "scripts/29_audit_covariance_design.py", "--scope", args.scope],
        ["Rscript", "scripts/15_variance_sensitivity.R", args.scope],
        [py, "scripts/12_benchmark_context.py", "--scope", args.scope],
        [py, "scripts/19_evaluate_calibration.py", "--scope", args.scope],
    ]
    if not args.skip_censored:
        for subset, prior in [("all", "1.5"), ("exact", "1.5"), ("no_reported_ranges", "1.5"),
                              ("multi_publication", "1.5"), ("all", "3")]:
            tasks.append([py, "scripts/14_fit_censored_model.py", "--scope", args.scope,
                          "--subset", subset, "--prior-scale", prior,
                          "--warmup", "1500", "--draws", "3000"])
    tasks += [
        [py, "scripts/23_check_posteriors.py", "--scope", args.scope],
        [py, "scripts/21_generate_manuscript_inputs.py", "--scope", args.scope],
        [py, "scripts/22_build_documents.py", "--scope", args.scope, "--render"],
        [py, "scripts/24_build_execution_report.py", "--scope", args.scope],
    ]
    output = ROOT / "results/reconstruction_2026-10-20" / args.scope
    output.mkdir(parents=True, exist_ok=True)
    events = []
    for command in tasks:
        print("Running:", " ".join(command), flush=True)
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 3 and "scripts/14_fit_censored_model.py" in command:
            command = command[:-4] + ["--warmup", "2000", "--draws", "6000",
                                     "--adapt-delta", ".99", "--max-treedepth", "14"]
            result = subprocess.run(command, cwd=ROOT)
        events.append({"command": command, "started_utc": started,
                       "finished_utc": datetime.now(timezone.utc).isoformat(), "returncode": result.returncode})
        (output / "pipeline_execution.json").write_text(json.dumps(events, indent=2))
        if result.returncode:
            raise SystemExit(result.returncode)
    manifest = {}
    for folder in ["amp_rebuild", "scripts", "models", "tests"]:
        for path in (ROOT/folder).glob("*"):
            if path.suffix in {".py", ".R", ".stan"}:
                manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "code_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Computational commands finished. Inspect convergence and coverage gates before interpreting results.",
          flush=True)
    print("Scope: strictly in silico. Complete the document review and final audit to close the study.", flush=True)


if __name__ == "__main__":
    main()

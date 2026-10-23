#!/usr/bin/env python3
"""Continue the expanded reconstruction after the currently running download.

This controller waits on an explicitly supplied live acquisition PID. It never
restarts a live acquisition because an observation timeout expired.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reconstruction_2026-10-20"


def acquisition_live(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
    return result.returncode == 0 and "09_expand_dbaasp.py" in result.stdout


def missing_catalog_ids():
    expected = {int(row["id"]) for row in json.loads((ROOT/"data/raw/dbaasp_peptide_index.json").read_text())}
    present = set()
    for folder in ["dbaasp_cards", "dbaasp_expansion_2026-10-20"]:
        present.update(int(p.stem) for p in (ROOT/"data/raw"/folder).glob("*.json") if p.stem.isdigit())
    return expected-present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-pid", required=True, type=int)
    parser.add_argument("--existing-multi-publication-pid",type=int)
    parser.add_argument("--existing-multi-publication-result",type=Path)
    args = parser.parse_args()
    if bool(args.existing_multi_publication_pid) != bool(args.existing_multi_publication_result):
        parser.error("The existing posterior PID and result path must be supplied together.")
    os.chdir(ROOT)
    py = sys.executable
    previous = OUT/"expanded_controller_status.json"
    events = json.loads(previous.read_text()) if previous.exists() else []
    event_lock = threading.Lock()
    def record(stage, status, **extra):
        event = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, "status": status, **extra}
        with event_lock:
            events.append(event)
            temporary = OUT/"expanded_controller_status.json.partial"
            temporary.write_text(json.dumps(events, indent=2))
            temporary.replace(OUT/"expanded_controller_status.json")
            print(json.dumps(event), flush=True)
    def execute(stage, command):
        log = OUT / f"expanded_{stage}.log"
        record(stage, "running", command=command, log=str(log))
        with log.open("a") as stream:
            stream.write("\nCOMMAND: " + " ".join(command) + "\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record(stage, "process_started", pid=process.pid)
            code = process.wait()
        record(stage, "finished" if code == 0 else "failed", returncode=code)
        return code
    while missing_catalog_ids():
        missing = len(missing_catalog_ids())
        if acquisition_live(args.acquisition_pid):
            record("acquisition", "verified_wait", acquisition_pid=args.acquisition_pid, missing_ids=missing)
            time.sleep(30)
            continue
        record("acquisition", "previous_process_terminal", missing_ids=missing)
        code = execute("acquisition_retry", [py, "scripts/09_expand_dbaasp.py"])
        if code or missing_catalog_ids():
            raise SystemExit("Catalog remains incomplete; inspect acquisition failures.")
    record("acquisition", "coverage_complete")
    stages = [
        ("publications", [py, "scripts/17_verify_publications.py", "--expanded"]),
        ("curation", [py, "scripts/10_curate_assays.py", "--expanded"]),
        ("embeddings", [py, "scripts/11_prepare_features.py", "--embeddings"]),
        ("chemical_features", [py, "scripts/11_prepare_features.py", "--scope", "expanded", "--chemical"]),
        ("clusters", [py, "scripts/11_prepare_features.py", "--scope", "expanded", "--clusters"]),
        ("structural_audit", [py, "scripts/20_verify_reconstruction.py", "--scope", "expanded", "--structures"]),
        ("tests", [py, "-m", "pytest", "-q", "tests/test_reconstruction.py",
                   "--junitxml", str(OUT/"expanded/tests.xml")]),
        ("statistics", [py, "scripts/13_corrected_statistics.py", "--scope", "expanded"]),
        ("covariance_design", [py, "scripts/29_audit_covariance_design.py", "--scope", "expanded"]),
        ("reml", ["Rscript", "scripts/15_variance_sensitivity.R", "expanded"]),
        ("benchmark", [py, "scripts/12_benchmark_context.py", "--scope", "expanded"]),
        ("calibration", [py, "scripts/19_evaluate_calibration.py", "--scope", "expanded"]),
        ("preliminary_figures", [py, "scripts/21_generate_manuscript_inputs.py", "--scope", "expanded"]),
    ]
    for stage, command in stages:
        if execute(stage, command):
            raise SystemExit(f"Expanded stage failed: {stage}")
    def censored_variant(subset, prior):
        base = [py, "scripts/14_fit_censored_model.py", "--scope", "expanded",
                "--subset", subset, "--prior-scale", prior]
        stage = f"censored_{subset}_prior{prior}"
        if subset == "multi_publication" and args.existing_multi_publication_pid:
            while True:
                live=subprocess.run(["ps","-p",str(args.existing_multi_publication_pid),"-o","command="],
                                    capture_output=True,text=True)
                if live.returncode or "14_fit_censored_model.py" not in live.stdout:
                    break
                if "--subset multi_publication" not in live.stdout:
                    raise RuntimeError("Existing posterior PID belongs to a different specification.")
                record(stage,"verified_wait_on_existing_fit",pid=args.existing_multi_publication_pid)
                time.sleep(30)
            if args.existing_multi_publication_result.exists():
                sys.path.insert(0,str(ROOT))
                from amp_rebuild.posterior_reuse import reuse_identical_prepared_fit
                reused=reuse_identical_prepared_fit(args.existing_multi_publication_result.resolve())
                record(stage,"exact_input_equivalence_certified",result=str(reused))
            else:
                record(stage,"previous_process_terminal_without_result")
        code = execute(stage, base+["--warmup", "1500", "--draws", "3000"])
        if code == 3:
            fits = list((OUT/"expanded/censored"/f"{subset}_prior{float(prior):g}").glob("*/result.json"))
            latest = max((json.loads(path.read_text()) for path in fits), key=lambda r: r["created_utc"])
            computational = latest["diagnostics"]
            geometry_problem = computational["divergences"] or computational["max_treedepth_hits"]
            code = execute(stage+"_longer", base+[
                "--warmup", "2000", "--draws", "6000",
                "--adapt-delta", ".99" if geometry_problem else ".95",
                "--max-treedepth", "14" if geometry_problem else "12"])
        return stage, code
    failures = []
    # Independent posterior sensitivities share immutable input files, not outputs.
    # Three four-chain fits use at most twelve sampler processes on this 16-core host.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(censored_variant, subset, prior)
                for subset, prior in [("all", "1.5"), ("multi_publication", "1.5"),
                                      ("exact", "1.5"), ("no_reported_ranges", "1.5"), ("all", "3")]]
        for future in as_completed(jobs):
            stage, code = future.result()
            if code:
                failures.append({"stage": stage, "returncode": code})
    for stage, command in [
        ("posterior_checks", [py, "scripts/23_check_posteriors.py", "--scope", "expanded"]),
        ("final_figures", [py, "scripts/21_generate_manuscript_inputs.py", "--scope", "expanded"]),
        ("manuscripts", [py, "scripts/22_build_documents.py", "--scope", "expanded", "--render"]),
        ("execution_report", [py, "scripts/24_build_execution_report.py", "--scope", "expanded"]),
    ]:
        if execute(stage, command):
            raise SystemExit(f"Expanded reporting stage failed: {stage}")
    if failures:
        record("computational_stages", "inference_gates_incomplete", failures=failures)
        raise SystemExit(3)
    record("computational_stages", "finished_pending_independent_completion_audit",
           study_type="strictly_in_silico", experimental_validation_required=False,
           manuscript_and_figures="generated; visual inspection pending")


if __name__ == "__main__":
    main()

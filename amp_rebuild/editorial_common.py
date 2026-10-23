"""Versioned editorial extension; every closed expanded artifact is read-only."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from .curation import ROOT

PHASE = "editorial_phase_2026-10-22_v1"
BASE = ROOT/"results/reconstruction_2026-10-20/expanded"
OUT = BASE/PHASE
DATA = ROOT/"data/curated/expanded/benchmark_cells.csv"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default)+"\n")
    temp.replace(path)


def lock_json(path, payload):
    path = Path(path)
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Versioned configuration changed: {path}")
    else:
        write_json(path, payload)


def read_data():
    data = pd.read_csv(DATA, keep_default_na=False, na_values=[""], low_memory=False)
    assert len(data) == 66877 and data.assay_cell_id.is_unique
    return data


def frozen_split(data, path):
    saved = pd.read_csv(path, keep_default_na=False)
    assert saved.assay_cell_id.is_unique
    assert saved.assay_cell_id.tolist() == data.assay_cell_id.tolist()
    assert set(saved.role).issubset({"train", "test", "excluded"})
    tr = np.flatnonzero(saved.role.eq("train").to_numpy())
    te = np.flatnonzero(saved.role.eq("test").to_numpy())
    assert not set(tr) & set(te)
    return tr, te


def load_prediction(path, data, indices):
    pred = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    assert pred.assay_cell_id.is_unique
    assert set(pred.assay_cell_id) == set(data.iloc[indices].assay_cell_id)
    pred = pred.set_index("assay_cell_id").loc[data.iloc[indices].assay_cell_id].reset_index()
    assert np.allclose(pred.log2_mic, data.iloc[indices].log2_mic, rtol=0, atol=1e-12)
    assert np.isfinite(pred.prediction).all()
    return pred


def protected_paths():
    paths = set()
    for folder in [ROOT/"data/curated/expanded", BASE,
                   ROOT/"manuscript/generated/expanded", ROOT/"manuscript/reference_metadata"]:
        for path in folder.rglob("*"):
            if path.is_file() and OUT not in path.parents:
                paths.add(path)
    provenance = read_json(ROOT/"manuscript/generated/expanded/report_provenance.json")
    paths.update(ROOT/relative for relative in provenance["files"])
    paths.update(ROOT/relative for relative in [
        "data/study_scope_2026-10-20.json", "tests/test_reconstruction.py",
        "scripts/25_completion_audit.py", "output/pdf/Ejecucion_reconstruccion_AMP_expanded_2026-10-20.pdf",
        "results/reconstruction_2026-10-20/execution_status.json",
    ])
    return sorted(paths)


def snapshot_preservation():
    destination = OUT/"preservation_manifest.json"
    if destination.exists():
        return read_json(destination)
    paths = protected_paths()
    entries = {}
    for number, path in enumerate(paths):
        stat = path.stat()
        entries[str(path.relative_to(ROOT))] = {
            "sha256": sha(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
        if (number+1) % 500 == 0:
            print(f"Preservation: {number+1}/{len(paths)} files fingerprinted", flush=True)
    manifest = {"created_utc": utc(), "phase": PHASE, "files": entries,
                "rule": "No deletion, replacement or invalidation of the closed expanded analysis. New scope and outputs are separate versioned artifacts."}
    write_json(destination, manifest)
    return manifest


def verify_preservation(full=False):
    manifest = read_json(OUT/"preservation_manifest.json")
    bad, rehashed = [], 0
    for relative, expected in manifest["files"].items():
        path = ROOT/relative
        if not path.is_file():
            bad.append({"path": relative, "reason": "missing"})
            continue
        stat = path.stat()
        changed = stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]
        if full or changed:
            rehashed += 1
            if sha(path) != expected["sha256"]:
                bad.append({"path": relative, "reason": "content_changed"})
    report = {"checked_utc": utc(), "passed": not bad, "protected_files": len(manifest["files"]),
              "full_sha256_recheck": full, "files_rehashed": rehashed, "mismatches": bad,
              "manifest_sha256": sha(OUT/"preservation_manifest.json")}
    if bad:
        raise ValueError(json.dumps(report))
    return report


def rerun_original_audit(destination):
    """Run unchanged audit logic, redirecting only its three output writes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    old_write = Path.write_text
    outputs = {BASE/name: destination/name for name in
               ["source_amendment_audit.json", "code_manifest.json", "completion_audit.json"]}

    def routed_write(path, content, *args, **kwargs):
        path = Path(path)
        if path in outputs:
            return old_write(outputs[path], content, *args, **kwargs)
        if BASE in path.parents and OUT not in path.parents:
            raise PermissionError(f"Original audit attempted an unexpected protected write: {path}")
        return old_write(path, content, *args, **kwargs)

    previous_argv = sys.argv
    try:
        sys.argv = [str(ROOT/"scripts/25_completion_audit.py"), "--scope", "expanded"]
        with (destination/"stdout.log").open("w") as log, redirect_stdout(log), redirect_stderr(log), \
                patch.object(Path, "write_text", routed_write):
            try:
                runpy.run_path(str(ROOT/"scripts/25_completion_audit.py"), run_name="__main__")
            except SystemExit as result:
                if result.code not in [0, None]:
                    raise RuntimeError(f"Original audit failed with code {result.code}")
    finally:
        sys.argv = previous_argv
    result = read_json(destination/"completion_audit.json")
    assert result["study_complete"] and len(result["requirements"]) == 19
    assert all(row["passed"] for row in result["requirements"].values())
    write_json(destination/"write_routing.json", {
        "created_utc": utc(), "audit_script_sha256": sha(ROOT/"scripts/25_completion_audit.py"),
        "method": "Execute unchanged audit with only the three output Path.write_text destinations redirected. No gate, source, scope or scientific decision is altered.",
        "destinations": {str(k.relative_to(ROOT)): str(v.relative_to(ROOT)) for k,v in outputs.items()},
    })
    return result

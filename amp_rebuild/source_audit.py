"""Read-only verification of reviewed source amendments and their propagation."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from .curation import ROOT


def audit_source_amendments(scope):
    source = ROOT / "data/curated" / scope
    registry_path = ROOT / "data/source_amendments_2026-10-20.json"
    registry = json.loads(registry_path.read_text())
    manifest = json.loads((source / "curation_manifest.json").read_text())
    applied = json.loads((source / "source_amendments_applied.json").read_text())
    rows = pd.read_csv(source / "activities.csv", dtype=str, keep_default_na=False)
    cells = pd.read_csv(source / "all_exact_assay_cells.csv", dtype=str, keep_default_na=False)
    inputs = {r["path"] for r in manifest["input_files"]}
    expected = [r for r in registry["amendments"] if r["source_card"] in inputs]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checks = []
    for amendment in expected:
        found = rows[rows.record_id == amendment["record_id"]]
        card = json.loads((ROOT / amendment["source_card"]).read_text())
        activity_id = int(amendment["record_id"].split(":")[-1])
        acts = [r for r in card["targetActivities"] if r["id"] == activity_id]
        row = found.iloc[0] if len(found) == 1 else None
        raw = acts[0] if len(acts) == 1 else {}
        evidence_ok = all((ROOT / e["path"]).exists() and
                          digest(ROOT / e["path"]) == e["sha256"]
                          for e in amendment["evidence"])
        raw_ok = (digest(ROOT / amendment["source_card"]) == amendment["source_card_sha256"] and
                  raw.get("medium", {}).get("name") == amendment["original"] and
                  str(raw.get("concentration")) == str(amendment["expected"]["concentration_raw"]) and
                  raw.get("unit", {}).get("name") == amendment["expected"]["unit_raw"])
        values_ok = row is not None and row[amendment["field"]] == amendment["corrected"] and all(
            row[key] == str(value) for key, value in amendment["expected"].items()
            if key != amendment["field"])
        cell_ok = False
        if values_ok and row.eligible_exact_analysis == "True":
            cell = cells[cells.assay_cell_id == row.assay_cell_id]
            members = rows[(rows.assay_cell_id == row.assay_cell_id) &
                           (rows.eligible_exact_analysis == "True")]
            if len(cell) == 1:
                cell = cell.iloc[0]
                cell_ok = (cell.medium == amendment["corrected"] and
                           json.loads(cell.record_ids) == sorted(members.record_id.tolist()) and
                           int(cell.n_records) == len(members) and
                           np.isclose(float(cell.log2_mic), members.log2_mic.astype(float).mean(),
                                      rtol=0, atol=1e-12))
        elif values_ok:
            cell_ok = True
        checks.append({"id": amendment["id"], "record_id": amendment["record_id"],
                       "raw_card_preserved": raw_ok, "primary_evidence_hashes_match": evidence_ok,
                       "corrected_field_and_preserved_values_match": bool(values_ok),
                       "exact_cell_propagation_matches": bool(cell_ok),
                       "passed": bool(raw_ok and evidence_ok and values_ok and cell_ok)})
    registry_ok = (applied == expected and manifest["source_amendments_applied"] == len(expected) and
                   manifest["source_amendments_sha256"] == digest(registry_path))
    return {"scope": scope, "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": registry_ok and all(c["passed"] for c in checks),
            "registry_matches_applied_manifest": registry_ok, "amendments": checks,
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [registry_path,
                source / "source_amendments_applied.json", source / "activities.csv",
                source / "all_exact_assay_cells.csv", source / "curation_manifest.json"]},
            "limit": "Checks propagation and frozen evidence, not independent human adjudication or an exhaustive source audit."}

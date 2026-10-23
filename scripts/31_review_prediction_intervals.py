#!/usr/bin/env python3
"""Reconcile interval summaries, outer partitions and test errors from saved outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="expanded")
    args = parser.parse_args()
    out = ROOT/"results/reconstruction_2026-10-20"/args.scope
    folder, benchmark = out/"prediction_intervals", out/"benchmark"
    audit = json.loads((folder/"completion_audit.json").read_text())
    assert audit["passed"] and audit["independent_recalculation"] and audit["completed_fits"] == 24
    summary_audit = json.loads((benchmark/"summary_completion_audit.json").read_text())
    assert summary_audit["passed"]
    for relative, digest in summary_audit["files"].items():
        assert sha(ROOT/relative) == digest
    summary = pd.read_csv(folder/"coverage_by_split.csv")
    keys = ["protocol", "fold", "model", "score_kind", "nominal_coverage"]
    models = json.loads((folder/"config.json").read_text())["models"]
    sources = [Path(__file__), folder/"config.json", folder/"completion_audit.json",
               folder/"coverage_by_split.csv", folder/"split_audit.json",
               benchmark/"summary_completion_audit.json", ROOT/"amp_rebuild/verification.py"]
    rows, role_checks, mae_checks = [], [], 0
    for protocol, folds in [("publication", range(5)), ("temporal", range(1))]:
        for fold in folds:
            assignment_path = folder/f"split_{protocol}_{fold}.csv.gz"
            primary_path = benchmark/f"split_{protocol}_20261020_{fold}.csv.gz"
            assignment = pd.read_csv(assignment_path).set_index("assay_cell_id")
            primary = pd.read_csv(primary_path).set_index("assay_cell_id")
            assert set(assignment.index[assignment.role.eq("test")]) == set(primary.index[primary.role.eq("test")])
            assert set(assignment.index[assignment.role.isin(["fit", "calibration"])]) == set(primary.index[primary.role.eq("train")])
            role_checks.append({"protocol": protocol, "fold": fold,
                                "outer_test_matches_audited_benchmark": True,
                                "fit_and_calibration_union_matches_outer_train": True})
            sources.extend([assignment_path, primary_path])
            for model in models:
                result_path = folder/f"{protocol}_{fold}_{model}.json"
                test_path = folder/f"{protocol}_{fold}_{model}.csv.gz"
                values = json.loads(result_path.read_text())
                test = pd.read_csv(test_path)
                error = float(np.abs(test.log2_mic-test.prediction).mean())
                for row in values:
                    assert np.isclose(row["test_mae_log2"], error, rtol=1e-12, atol=1e-12)
                    assert np.isclose(row["width_log2"], 2*row["radius_log2"], rtol=1e-12, atol=1e-12)
                mae_checks += 1
                rows.extend(values)
                sources.extend([result_path, test_path, folder/f"{protocol}_{fold}_{model}_calibration.csv.gz"])
    raw = pd.DataFrame(rows).set_index(keys).sort_index()
    combined = summary.set_index(keys).sort_index()
    assert raw.index.equals(combined.index) and set(raw.columns) == set(combined.columns)
    np.testing.assert_allclose(raw[combined.columns], combined, rtol=1e-12, atol=1e-12)
    temporal = summary[summary.protocol.eq("temporal")]
    example = temporal[temporal.model.eq("sequence_hgb") & temporal.score_kind.eq("publication_maximum") &
                       temporal.nominal_coverage.eq(.9)].iloc[0]
    pooled = temporal[temporal.model.eq("sequence_hgb") & temporal.score_kind.eq("pooled_rows") &
                      temporal.nominal_coverage.eq(.9)].iloc[0]
    record = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope, "passed": True,
        "completed_interval_fits": 24, "summary_rows_reconciled": len(summary),
        "test_mae_recalculations": mae_checks, "outer_role_checks": role_checks,
        "temporal_descriptor_90pct": {
            "publication_maximum": example.to_dict(), "pooled_rows": pooled.to_dict(),
            "publication_maximum_multiplicative_half_width": float(2**example.radius_log2),
            "pooled_rows_multiplicative_half_width": float(2**pooled.radius_log2)},
        "temporal_all_comparisons": temporal.to_dict(orient="records"),
        "interpretation": [
            "Publication-maximum and pooled-row calibration target different coverage quantities. Individual-cell coverage is not simultaneous coverage of all tested cells in a publication.",
            "The temporal descriptor interval based on publication maxima has high observed whole-publication coverage but a broad multiplicative range, limiting resolution for closely spaced MIC values.",
            "A pooled-row interval with approximately nominal marginal coverage need not cover every tested cell in a publication. Lower whole-publication coverage does not itself establish failure of marginal calibration.",
            "These empirical coverages are conditional on the selected source data and the fitted/calibrated procedure, without a guarantee under arbitrary temporal or study shift.",
            "Interval models reserve training publications for calibration and differ from the primary point-prediction models. No performance difference is attributed only to the intervals.",
            "All 24 interval fits, their 96 score/coverage rows and both calibration approaches are retained. This exploratory extension does not establish laboratory utility.",
        ],
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    path = out/"prediction_interval_review.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)
    print(json.dumps({"passed": True, "scope": args.scope, "coverage_rows": len(summary),
                      "mae_checks": mae_checks, "outer_partitions": len(role_checks)}, indent=2))


if __name__ == "__main__":
    main()

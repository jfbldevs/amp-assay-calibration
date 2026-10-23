#!/usr/bin/env python3
"""Audit completion of the author's strictly in silico study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.reporting import ROOT, posterior_results, read_json, read_csv, load_covariance_design_audit
from amp_rebuild.source_audit import audit_source_amendments


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_document_review(scope, review):
    study_scope = read_json(ROOT/"data/study_scope_2026-10-20.json")
    expected = {f"output/pdf/Ejecucion_reconstruccion_AMP_{scope}_2026-10-20.pdf"}
    if study_scope.get("manuscript_authoring_required", True):
        expected.update({f"output/pdf/AMP_manuscript_{scope}_2026-10-20.pdf",
                         f"output/pdf/AMP_supplement_{scope}_2026-10-20.pdf"})
    documents = review.get("documents", [])
    checks = []
    for row in documents:
        path = ROOT/row.get("path", "")
        current = path.is_file() and sha(path) == row.get("sha256")
        page_count = len(PdfReader(path).pages) if current and row["path"] in expected else 0
        checks.append({
            "path": row.get("path"), "current_sha256_matches": current,
            "page_count": page_count,
            "all_pages_reviewed": bool(page_count) and
                                  row.get("reviewed_pages") == list(range(1, page_count+1)),
        })
    return {
        "passed": bool(review.get("passed")) and len(documents) == len(expected) and
                  {row.get("path") for row in documents} == expected and
                  all(row["current_sha256_matches"] and row["all_pages_reviewed"] for row in checks),
        "documents": checks,
        "expected_documents": sorted(expected),
        "note": "A structural receipt check; the reviewer must actually inspect every rendered page before recording it.",
    }


def audit_figure_review(scope, review, study_scope):
    expected = set(study_scope.get("required_research_figures", []))
    figures = review.get("figures", [])
    checks = []
    for row in figures:
        stem = row.get("stem", "")
        prefix = f"manuscript/generated/{scope}/{stem}"
        current = stem in expected and all(
            row.get(kind+"_path") == prefix+"."+kind and
            (ROOT/(prefix+"."+kind)).is_file() and
            sha(ROOT/(prefix+"."+kind)) == row.get(kind+"_sha256")
            for kind in ["png", "pdf"])
        checks.append({"stem": stem, "current_files_match": current,
                       "visually_reviewed": row.get("visually_reviewed") is True})
    return {
        "passed": bool(expected) and len(figures) == len(expected) and
                  {r.get("stem") for r in figures} == expected and
                  all(r["current_files_match"] and r["visually_reviewed"] for r in checks),
        "expected_figures": sorted(expected), "figures": checks,
        "note": "A receipt check only; every figure must actually be inspected before recording visual review.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="expanded")
    args = parser.parse_args()
    out = ROOT/"results/reconstruction_2026-10-20"/args.scope
    source = ROOT/"data/curated"/args.scope
    conditions = {}
    scope_path = ROOT/"data/study_scope_2026-10-20.json"
    study_scope = read_json(scope_path)
    conditions["authorized_study_scope"] = {
        "passed": study_scope.get("study_type") == "strictly_in_silico" and
                  study_scope.get("experimental_validation_required") is False,
        "scope_manifest": str(scope_path.relative_to(ROOT)),
        "sha256": sha(scope_path),
    }
    manifest = read_json(source/"curation_manifest.json")
    amendment_audit = audit_source_amendments(args.scope)
    (out/"source_amendment_audit.json").write_text(json.dumps(amendment_audit, indent=2))
    conditions["reviewed_source_amendments"] = amendment_audit
    exact_cells = read_csv(source/"all_exact_assay_cells.csv")
    prediction_cells = read_csv(source/"benchmark_cells.csv")
    actual_counts = {
        "exact_assay_cells": len(exact_cells),
        "benchmark_exact_assay_cells": len(prediction_cells),
        "exact_chemical_entities": int(exact_cells.chemical_id.nunique()),
        "exact_backbone_sequences": int(exact_cells.sequence_id.nunique()),
        "exact_cells_without_backbone_id": int(exact_cells.sequence_id.isna().sum()),
        "exact_publications": int(exact_cells.publication_id.nunique()),
    }
    conditions["cohort_counts"] = {
        "passed": all(manifest.get(k) == v for k,v in actual_counts.items()) and
                  not prediction_cells.sequence_id.isna().any(),
        "recalculated": actual_counts,
        "benchmark_chemical_entities": int(prediction_cells.chemical_id.nunique()),
        "benchmark_backbone_sequences": int(prediction_cells.sequence_id.nunique()),
        "benchmark_publications": int(prediction_cells.publication_id.nunique()),
        "note": "Missing backbone identifiers are not counted as sequences; inferential and predictive cohorts have distinct denominators.",
    }
    expected = {int(row["id"]) for row in read_json(ROOT/"data/raw/dbaasp_peptide_index.json")}
    present = {int(Path(row["path"]).stem) for row in manifest["input_files"]}
    conditions["catalog_coverage"] = {
        "passed": present == expected if args.scope == "expanded" else len(present) == 2500,
        "scope": args.scope, "cards": len(present), "indexed_cards": len(expected),
        "missing_indexed_cards": len(expected-present),
    }
    benchmark = read_json(out/"benchmark/completion_audit.json")
    conditions["benchmark"] = {
        "passed": benchmark["passed"] and benchmark["input_sha256"] == sha(source/"benchmark_cells.csv"),
        "expected_fits": benchmark["expected_fits"], "verified_fits": benchmark["verified_fits"],
        "test_predictions_verified": sum(p["rows"] for p in benchmark["checked_predictions"]),
    }
    summary_path = out/"benchmark/summary_completion_audit.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    conditions["benchmark_summaries"] = {
        "passed": bool(summary.get("passed")) and summary.get("input_sha256") == sha(source/"benchmark_cells.csv")
                  and all((ROOT/p).exists() and sha(ROOT/p) == digest for p,digest in summary.get("files",{}).items()),
        "audit": summary,
    }
    posterior = posterior_results(args.scope)
    variants = [("all",1.5),("exact",1.5),("multi_publication",1.5),
                ("no_reported_ranges",1.5),("all",3.0)]
    posterior_rows = []
    for key in variants:
        result, path = posterior.get(key, ({}, None))
        posterior_rows.append({
            "subset": key[0], "prior_scale": key[1], "available": path is not None,
            "passed": result.get("diagnostics",{}).get("passed",False),
            "result": str(path.relative_to(ROOT)) if path else None,
        })
    conditions["censored_inference"] = {"passed": all(r["passed"] for r in posterior_rows),
                                        "specifications": posterior_rows}
    precision_rows = []
    for key in variants:
        _, path = posterior.get(key, ({}, None))
        precision_path = path.parent/"sampling_precision.json" if path else None
        precision = read_json(precision_path) if precision_path and precision_path.exists() else {}
        precision_rows.append({
            "subset": key[0], "prior_scale": key[1],
            "passed": bool(precision.get("passed")) and all(
                (path.parent/name).exists() and sha(path.parent/name) == digest
                for name,digest in precision.get("source_sha256",{}).items()),
            "diagnostics": precision,
        })
    conditions["sampling_precision_review"] = {
        "passed": all(r["passed"] for r in precision_rows), "specifications": precision_rows}
    sensitivity = {}
    if all(key in posterior and posterior[key][0]["diagnostics"]["passed"] for key in [("all",1.5),("all",3.0)]):
        for name, small in posterior[("all",1.5)][0]["conditional_variance_fractions"].items():
            wide = posterior[("all",3.0)][0]["conditional_variance_fractions"][name]
            sensitivity[name] = {"prior_1_5": small, "prior_3": wide,
                                 "median_difference": wide["median"]-small["median"]}
    postchecks = read_json(out/"posterior_check_summary.json") if (out/"posterior_check_summary.json").exists() else []
    conditions["posterior_checks_performed"] = {
        "passed": len(postchecks) == len(variants), "checks": postchecks,
        "interpretation": "Model adequacy flags must be discussed; they are not erased by sampler convergence.",
    }
    test_file = out/"tests.xml"
    if test_file.exists():
        tree = ET.parse(test_file).getroot()
        suites = list(tree) if tree.tag == "testsuites" else [tree]
        totals = {name: sum(int(s.get(name, 0)) for s in suites)
                  for name in ["tests","failures","errors","skipped"]}
        conditions["tests"] = {"passed": totals["failures"] == totals["errors"] == 0 and
                              (totals["skipped"] == 0 if args.scope == "expanded" else True), **totals}
    else:
        conditions["tests"] = {"passed": False, "reason": "No final test receipt."}
    required_tables = ["benchmark/calibration_diagnostics.csv", "benchmark/screening_summary.csv",
                       "statistics/reml_sensitivity.json", "statistics/matched_rank_summary.json",
                       "statistics/design_support.json", "eligibility_exclusions.csv"]
    conditions["secondary_outputs"] = {"passed": all((out/p).exists() for p in required_tables)}
    secondary_path = out/"secondary_evaluation_review.json"
    secondary = read_json(secondary_path) if secondary_path.exists() else {}
    secondary_sources = secondary.get("source_sha256", {})
    conditions["secondary_evaluation_review"] = {
        "passed": bool(secondary.get("passed")) and bool(secondary_sources) and
                  all((ROOT/p).exists() and sha(ROOT/p) == digest
                      for p, digest in secondary_sources.items()),
        "review": str(secondary_path.relative_to(ROOT)),
        "screening_summary_rows_recomputed": secondary.get("screening_summary_rows_recomputed"),
        "temporal_selection_rows_reconstructed": secondary.get("temporal_selection_rows_reconstructed"),
        "temporal_calibration_strata_recomputed": secondary.get("temporal_calibration_strata_recomputed"),
        "note": "All screening aggregates and temporal selections/calibration are recomputed; individual non-temporal ranking coefficients are not all reconstructed.",
    }
    try:
        covariance = load_covariance_design_audit(args.scope)
        conditions["covariance_design_review"] = {
            "passed": True,
            "cohorts": {key: {name: row[name] for name in ["records", "rank", "component_count"]}
                        for key, row in covariance["cohorts"].items()},
            "note": "Completion requires a current recorded diagnostic, not full rank or a favorable scientific finding.",
        }
    except (OSError, ValueError, KeyError) as error:
        conditions["covariance_design_review"] = {"passed": False, "reason": str(error)}
    threshold_path = out/"statistics/rank_threshold_sensitivity.json"
    threshold_audit = read_json(threshold_path) if threshold_path.exists() else {}
    threshold_rows = threshold_audit.get("results", [])
    threshold_valid = False
    if threshold_rows:
        rank_pairs = read_csv(out/"statistics/matched_rank_pairs.csv")
        a, b = rank_pairs.difference_a_log2.to_numpy(), rank_pairs.difference_b_log2.to_numpy()
        threshold_valid = (
            [r["threshold_log2"] for r in threshold_rows] == [1.,1.01,1.05,1.1,1.25,1.5,2.] and
            all(r["pairs"] == len(rank_pairs) and
                r["inversions"] == int(((a*b < 0) & (np.minimum(abs(a),abs(b)) > r["threshold_log2"]+1e-12)).sum()) and
                np.isclose(r["rate"], r["inversions"]/len(rank_pairs)) and
                0 <= r["ci_low"] <= r["ci_high"] <= 1 for r in threshold_rows))
    conditions["rank_threshold_sensitivity"] = {
        "passed": threshold_valid and threshold_audit.get("numerical_comparison_atol_log2") == 1e-12 and
                  bool(threshold_audit.get("primary_result_reproduced")) and
                  threshold_audit.get("input_sha256") == sha(out/"statistics/matched_rank_pairs.csv"),
        "status": threshold_audit.get("status"),
        "thresholds": threshold_rows,
    }
    interval_path = out/"prediction_intervals/completion_audit.json"
    intervals = read_json(interval_path) if interval_path.exists() else {}
    conditions["prediction_intervals"] = {
        "passed": bool(intervals.get("passed")) and bool(intervals.get("independent_recalculation")) and
                  intervals.get("input_sha256") == sha(source/"benchmark_cells.csv"),
        "audit": intervals,
    }
    interval_review_path = out/"prediction_interval_review.json"
    interval_review = read_json(interval_review_path) if interval_review_path.exists() else {}
    interval_sources = interval_review.get("source_sha256", {})
    conditions["prediction_interval_review"] = {
        "passed": bool(interval_review.get("passed")) and bool(interval_sources) and
                  all((ROOT/p).exists() and sha(ROOT/p) == digest
                      for p, digest in interval_sources.items()),
        "review": str(interval_review_path.relative_to(ROOT)),
        "coverage_rows_reconciled": interval_review.get("summary_rows_reconciled"),
        "test_mae_recalculations": interval_review.get("test_mae_recalculations"),
    }
    provenance = ROOT/"manuscript/generated"/args.scope/"report_provenance.json"
    report = read_json(provenance) if provenance.exists() else {}
    conditions["generated_report_provenance"] = {
        "passed": bool(report) and all((ROOT/p).exists() and sha(ROOT/p) == digest
                                      for p,digest in report.get("files",{}).items())}
    visual_path = out/"visual_review.json"
    visual = read_json(visual_path) if visual_path.exists() else {}
    conditions["document_visual_review"] = {
        **audit_document_review(args.scope, visual),
        "receipt": str(visual_path.relative_to(ROOT)),
    }
    conditions["figure_visual_review"] = audit_figure_review(args.scope, visual, study_scope)
    code = {}
    for directory in ["amp_rebuild","scripts","models","tests"]:
        for p in (ROOT/directory).glob("*"):
            if p.suffix in {".py",".R",".stan"}:
                code[str(p.relative_to(ROOT))] = sha(p)
    (out/"code_manifest.json").write_text(json.dumps(code,indent=2))
    complete = all(r["passed"] for r in conditions.values())
    result = {
        "scope": args.scope, "computational_complete": complete,
        "study_complete": complete and args.scope == "expanded",
        "study_type": study_scope["study_type"],
        "experimental_validation_required": False,
        "prospective_measurements_performed": False,
        "requirements": conditions, "prior_sensitivity": sensitivity,
        "completion_interpretation": "Completion concerns the defined retrospective in silico study and its verified documentation. It does not certify experimental utility or journal acceptance.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out/"completion_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"prior_sensitivity"}},indent=2))
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()

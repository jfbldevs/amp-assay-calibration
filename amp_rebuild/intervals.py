"""Exploratory prediction intervals calibrated without using outer-test outcomes."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .benchmark import feature_table, make_estimator
from .curation import ROOT
from .splits import make_splits

INTERVAL_MODELS = ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb"]


def conformal_quantile(scores, coverage):
    """Finite-sample upper order statistic, including the necessary infinity case."""
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a nonempty finite array.")
    rank = math.ceil((len(values)+1)*coverage)
    return float(values[rank-1]) if rank <= len(values) else math.inf


def calibration_partition(data, outer_train, seed, fraction=.2):
    publications = np.sort(data.iloc[outer_train].publication_id.unique())
    if len(publications) < 10:
        raise ValueError("Insufficient publications for a separated calibration cohort.")
    shuffled = np.random.default_rng(seed).permutation(publications)
    count = max(1, math.ceil(len(shuffled)*fraction))
    is_calibration = data.iloc[outer_train].publication_id.isin(shuffled[:count]).to_numpy()
    return outer_train[~is_calibration], outer_train[is_calibration]


def evaluate_intervals(scope):
    source = ROOT/"data/curated"/scope/"benchmark_cells.csv"
    out = ROOT/"results/reconstruction_2026-10-20"/scope/"prediction_intervals"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "scope": scope, "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "models": INTERVAL_MODELS, "protocols": ["publication", "temporal"],
        "outer_seed": 20261020, "calibration_fraction_of_training_publications": .2,
        "nominal_coverages": [.9, .95],
        "calibration_scores": ["publication_maximum", "pooled_rows"],
        "status": "Exploratory extension specified after inspecting the initial point-prediction benchmark.",
        "assumptions": "Publication-maximum calibration requires exchangeability of entire study groups for a group-level coverage statement. Shared peptides, selective reporting and time shift can violate it. Pooled-row calibration additionally ignores within-paper dependence.",
        "interpretation": "Report empirical outer-test coverage and width; no unconditional validity guarantee is asserted for this literature dataset.",
        "estimator_sha256": hashlib.sha256(inspect.getsource(make_estimator).encode()).hexdigest(),
        "calibration_algorithm_sha256": hashlib.sha256((inspect.getsource(conformal_quantile)+
            inspect.getsource(calibration_partition)).encode()).hexdigest(),
    }
    lock = out/"config.json"
    if lock.exists():
        if json.loads(lock.read_text()) != config:
            raise ValueError("Prediction interval inputs or configuration changed; version the output.")
    else:
        lock.write_text(json.dumps(config, indent=2))
        (out/"config_lock.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat()}))
    data = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False)
    features, embedding, _ = feature_table(data, True, False)
    summaries, audits = [], []
    for protocol in config["protocols"]:
        for fold, (outer_train, test) in enumerate(make_splits(data, protocol, 20261020, 5)):
            fitting, calibration = calibration_partition(
                data, outer_train, 20271020+fold+(100 if protocol == "temporal" else 0))
            fit_pubs = set(data.iloc[fitting].publication_id)
            cal_pubs = set(data.iloc[calibration].publication_id)
            test_pubs = set(data.iloc[test].publication_id)
            assert not fit_pubs & cal_pubs and not fit_pubs & test_pubs and not cal_pubs & test_pubs
            label = f"{protocol}_{fold}"
            assignment = np.full(len(data), "excluded", dtype=object)
            assignment[fitting], assignment[calibration], assignment[test] = "fit", "calibration", "test"
            pd.DataFrame({"assay_cell_id": data.assay_cell_id, "role": assignment}).to_csv(
                out/f"split_{label}.csv.gz", index=False)
            audits.append({
                "protocol": protocol, "fold": fold, "fit_rows": len(fitting),
                "calibration_rows": len(calibration), "test_rows": len(test),
                "fit_publications": len(fit_pubs), "calibration_publications": len(cal_pubs),
                "test_publications": len(test_pubs), "publication_overlap": 0,
                "test_sequence_seen_in_fitting_fraction": float(
                    data.iloc[test].sequence_id.isin(set(data.iloc[fitting].sequence_id)).mean()),
            })
            for name in INTERVAL_MODELS:
                stem = out/f"{label}_{name}"
                result_path = stem.with_suffix(".json")
                prediction_path = stem.with_suffix(".csv.gz")
                calibration_path = out/f"{label}_{name}_calibration.csv.gz"
                if result_path.exists() and prediction_path.exists() and calibration_path.exists():
                    summaries.extend(json.loads(result_path.read_text()))
                    continue
                model = make_estimator(name, embedding, len(fitting), 20261020)
                with threadpool_limits(limits=4):
                    model.fit(features.iloc[fitting], data.iloc[fitting].log2_mic)
                    cal_pred = model.predict(features.iloc[calibration])
                    test_pred = model.predict(features.iloc[test])
                cal = data.iloc[calibration][["assay_cell_id","publication_id","log2_mic"]].copy()
                cal["prediction"] = cal_pred
                cal["absolute_error"] = abs(cal.log2_mic-cal.prediction)
                cal.to_csv(calibration_path, index=False)
                saved = data.iloc[test][["assay_cell_id","publication_id","sequence_id","log2_mic"]].copy()
                saved["prediction"] = test_pred
                results = []
                for score_kind in config["calibration_scores"]:
                    scores = (cal.groupby("publication_id").absolute_error.max().to_numpy()
                              if score_kind == "publication_maximum" else cal.absolute_error.to_numpy())
                    for level in config["nominal_coverages"]:
                        radius = conformal_quantile(scores, level)
                        suffix = f"{score_kind}_{int(100*level)}"
                        saved["lower_"+suffix], saved["upper_"+suffix] = test_pred-radius, test_pred+radius
                        covered = pd.DataFrame({
                            "publication_id": saved.publication_id,
                            "covered": abs(saved.log2_mic-test_pred) <= radius,
                        })
                        per_pub = covered.groupby("publication_id").covered
                        results.append({
                            "protocol": protocol, "fold": fold, "model": name,
                            "score_kind": score_kind, "nominal_coverage": level,
                            "radius_log2": radius, "width_log2": 2*radius,
                            "test_cells": len(test), "calibration_scores": len(scores),
                            "cell_weighted_coverage": float(covered.covered.mean()),
                            "equal_publication_mean_cell_coverage": float(per_pub.mean().mean()),
                            "whole_publication_coverage": float(per_pub.all().mean()),
                            "test_mae_log2": float(np.mean(abs(saved.log2_mic-test_pred))),
                            "fit_rows": len(fitting), "calibration_rows": len(calibration),
                        })
                saved.to_csv(prediction_path, index=False)
                result_path.write_text(json.dumps(results, indent=2))
                summaries.extend(results)
                print(f"Intervals {scope} {label} {name}: completed", flush=True)
    pd.DataFrame(summaries).to_csv(out/"coverage_by_split.csv", index=False)
    (out/"split_audit.json").write_text(json.dumps(audits, indent=2))
    receipt = {
        "completed_fits": len(summaries)//4, "expected_fits": 24,
        "input_sha256": config["input_sha256"],
        "passed": len(summaries) == 96 and all(a["publication_overlap"] == 0 for a in audits),
        "scope": scope, "test_outcomes_used_for_calibration": False,
        "interpretation": config["interpretation"],
    }
    (out/"completion_audit.json").write_text(json.dumps(receipt, indent=2))
    from .verification import audit_prediction_intervals
    return audit_prediction_intervals(scope)

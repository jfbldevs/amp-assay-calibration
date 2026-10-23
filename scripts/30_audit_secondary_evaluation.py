#!/usr/bin/env python3
"""Review saved screening summaries and independently reconstruct temporal selection.

The tie-averaging sensitivity is exploratory, after inspecting the saved screening
results. It preserves the primary deterministic ranking and all trained models.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLOCK_FIELDS = ["publication_id", "target_exact", "medium", "cfu_raw", "salt_type_raw",
                "ionic_strength_raw", "ph_raw", "note_raw"]
SCORES = ["spearman", "kendall_tau_b", "gain_over_random_log2",
          "regret_to_observed_oracle_log2"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True):
        raise AssertionError(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="expanded")
    args = parser.parse_args()
    out = ROOT/"results/reconstruction_2026-10-20"/args.scope
    benchmark = out/"benchmark"
    data_path = ROOT/"data/curated"/args.scope/"benchmark_cells.csv"
    sources = [data_path, Path(__file__), ROOT/"amp_rebuild/evaluation.py"]
    for name in ["within_assay_screening.csv", "screening_by_publication.csv",
                 "screening_summary.csv", "calibration_diagnostics.csv",
                 "input_manifest.json", "analysis_config.json"]:
        sources.append(benchmark/name)
    manifest = json.loads((benchmark/"input_manifest.json").read_text())
    assert sha(data_path) == manifest["sha256"]
    config = json.loads((benchmark/"analysis_config.json").read_text())
    rank = pd.read_csv(benchmark/"within_assay_screening.csv")
    block_key = ["protocol", "model", "seed", "fold", "ranking_block_id"]
    assert not rank.duplicated(block_key).any()
    assert (rank.n_candidates >= 5).all()
    assert np.array_equal(rank.n_selected, np.ceil(.2*rank.n_candidates).astype(int))
    assert_close(rank.gain_over_random_log2,
                 rank.unselected_population_mean_log2_mic-rank.selected_mean_log2_mic,
                 "Saved screening gain arithmetic")
    assert (rank.regret_to_observed_oracle_log2 >= -1e-10).all()
    support = rank.groupby(["protocol", "seed", "fold", "ranking_block_id"])
    assert (support.model.nunique() == len(config["models"])).all()
    assert (support.n_candidates.nunique() == 1).all()
    assert (support.publication_id.nunique() == 1).all()

    pub_key = ["protocol", "model", "publication_id"]
    recomputed_pub = rank.groupby(pub_key)[SCORES].mean().sort_index()
    saved_pub = pd.read_csv(benchmark/"screening_by_publication.csv").set_index(pub_key).sort_index()
    assert recomputed_pub.index.equals(saved_pub.index)
    assert_close(recomputed_pub, saved_pub[SCORES], "Publication-level screening scores")
    summary = pd.read_csv(benchmark/"screening_summary.csv").set_index(["protocol", "model"]).sort_index()
    recalculated = recomputed_pub.groupby(["protocol", "model"]).mean().sort_index()
    assert summary.index.equals(recalculated.index)
    for source, target in [("spearman", "mean_within_assay_spearman"),
                           ("gain_over_random_log2", "mean_screening_gain_log2"),
                           ("regret_to_observed_oracle_log2", "mean_regret_log2")]:
        assert_close(recalculated[source], summary[target], target)
    assert np.array_equal(recomputed_pub.groupby(["protocol", "model"]).size(), summary.publications)

    data = pd.read_csv(data_path, keep_default_na=False, na_values=[""], low_memory=False)
    data["ranking_block_id"] = [
        "SCREEN:"+hashlib.sha256(json.dumps(list(row), sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode()).hexdigest()[:24]
        for row in data[BLOCK_FIELDS].fillna("").itertuples(index=False, name=None)
    ]
    meta = data[["assay_cell_id", "ranking_block_id", "publication_id", "log2_mic"]]
    calibration = pd.read_csv(benchmark/"calibration_diagnostics.csv")
    temporal_rank = rank[rank.protocol.eq("temporal")].set_index(["model", "ranking_block_id"])
    ties, temporal_files, calibration_checks, selection_checks = [], [], 0, 0
    for path in sorted((benchmark/"predictions").glob("temporal_*.csv.gz")):
        temporal_files.append(path)
        frame = pd.read_csv(path)
        model = frame.model.iloc[0]
        joined = frame.merge(meta, on="assay_cell_id", how="left", validate="one_to_one",
                             suffixes=("", "_source"))
        assert joined.ranking_block_id.notna().all()
        assert joined.publication_id.equals(joined.publication_id_source)
        assert_close(joined.log2_mic, joined.log2_mic_source, "Temporal labels")
        for novelty, part in joined.groupby("novelty"):
            if len(part) < 10 or part.prediction.std() == 0:
                continue
            actual = calibration[(calibration.protocol=="temporal") &
                                 (calibration.model==model) & (calibration.novelty==novelty)]
            assert len(actual) == 1
            actual = actual.iloc[0]
            pred, y = part.prediction.to_numpy(), part.log2_mic.to_numpy()
            centered = pred-pred.mean()
            slope = np.dot(centered, y-y.mean())/np.dot(centered, centered)
            expected = [y.mean()-slope*pred.mean(), slope, (pred-y).mean()]
            assert_close(actual[["intercept", "slope", "bias"]].to_numpy(float), expected,
                         "Temporal calibration coefficients")
            assert actual.n == len(part)
            calibration_checks += 1
        for block, part in joined.groupby("ranking_block_id"):
            if len(part) < 5:
                continue
            y, pred = part.log2_mic.to_numpy(), part.prediction.to_numpy()
            n, k = len(part), math.ceil(.2*len(part))
            order = np.lexsort((part.assay_cell_id.to_numpy(), pred))
            selected = y[order[:k]].mean()
            oracle = np.sort(y)[:k].mean()
            actual = temporal_rank.loc[(model, block)]
            assert actual.n_candidates == n and actual.n_selected == k
            assert_close(actual[["selected_mean_log2_mic", "unselected_population_mean_log2_mic",
                                 "gain_over_random_log2", "regret_to_observed_oracle_log2"]].to_numpy(float),
                         [selected, y.mean(), y.mean()-selected, selected-oracle],
                         "Temporal candidate selection reconstructed from predictions")
            selection_checks += 1
            cutoff = pred[order[k-1]]
            for tolerance in [0., 1e-12]:
                tied = np.abs(pred-cutoff) <= tolerance
                lower = pred < cutoff-tolerance
                remaining = k-int(lower.sum())
                assert 0 < remaining <= int(tied.sum())
                expected_selected = (y[lower].sum()+remaining*y[tied].mean())/k
                ties.append({"model": model, "publication_id": part.publication_id.iloc[0],
                             "ranking_block_id": block, "tie_tolerance_log2": tolerance,
                             "cutoff_tie_changes_selection": int(tied.sum()) > remaining,
                             "deterministic_gain_log2": float(y.mean()-selected),
                             "expected_tie_gain_log2": float(y.mean()-expected_selected),
                             "expected_tie_regret_log2": float(expected_selected-oracle)})
    assert len(temporal_files) == len(config["models"])
    assert selection_checks == len(temporal_rank)
    tie_frame = pd.DataFrame(ties)
    tie_pub = tie_frame.groupby(["model", "tie_tolerance_log2", "publication_id"])[
        ["deterministic_gain_log2", "expected_tie_gain_log2", "expected_tie_regret_log2"]].mean()
    tie_summary = tie_pub.groupby(["model", "tie_tolerance_log2"]).mean()
    tie_summary["publications"] = tie_pub.groupby(["model", "tie_tolerance_log2"]).size()
    grouped_ties = tie_frame.groupby(["model", "tie_tolerance_log2"])
    tie_summary["block_evaluations"] = grouped_ties.size()
    tie_summary["blocks_with_cutoff_ties"] = grouped_ties.cutoff_tie_changes_selection.sum()
    tie_path = out/"temporal_screening_tie_sensitivity.csv"
    tie_summary.reset_index().to_csv(tie_path, index=False)
    sources.extend(temporal_files)
    sources.append(tie_path)
    receipt = {
        "reviewed_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope, "passed": True,
        "screening_block_rows_checked": len(rank), "screening_publication_rows_recomputed": len(saved_pub),
        "screening_summary_rows_recomputed": len(summary),
        "all_models_have_same_eligible_blocks_and_candidate_counts": True,
        "temporal_prediction_files_checked": len(temporal_files),
        "temporal_selection_rows_reconstructed": selection_checks,
        "temporal_calibration_strata_recomputed": calibration_checks,
        "tie_sensitivity": tie_summary.reset_index().to_dict(orient="records"),
        "interpretation": [
            "Primary screening uses the saved deterministic order: prediction, then assay-cell identifier.",
            "The exploratory tie sensitivity integrates uniformly over candidate choices at the cutoff; it changes neither fitted models nor primary results.",
            "The 1e-12 absolute log2 tolerance is a numerical tie sensitivity, not a biological equivalence threshold.",
            "The saved unselected_population_mean_log2_mic column is the mean of all candidates, including selected candidates; it is the expected mean under uniformly random selection.",
            "Counts of eligible publications differ across protocols; their screening summaries are not matched estimates of a protocol effect.",
            "Joint screening pools available candidates within each held-out block; it is not restricted to the new-sequence/new-publication stratum.",
            "Differences in screening gain are descriptive without a sampling confidence interval. Equal-publication averaging differs from cell-weighted MAE.",
            "This review recomputes all saved screening aggregates and temporal selection/calibration. It does not recompute every non-temporal individual ranking or its rank-correlation coefficient.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
    }
    (out/"secondary_evaluation_review.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ["tie_sensitivity", "source_sha256", "interpretation"]}, indent=2))
    print(tie_summary.loc[["sequence_hgb", "sequence_context_hgb", "esm_hgb",
                           "esm_context_hgb", "target_mean", "context_ridge"]].to_string())


if __name__ == "__main__":
    main()

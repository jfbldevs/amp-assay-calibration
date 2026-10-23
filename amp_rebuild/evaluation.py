"""Calibration and within-assay screening diagnostics from saved test predictions."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .curation import ROOT, stable_hash


def screening_metrics(frame, fraction=.2):
    frame = frame.sort_values("assay_cell_id")
    k = max(1, math.ceil(len(frame)*fraction))
    selected = frame.sort_values("prediction", kind="stable").head(k)
    oracle = frame.sort_values("log2_mic", kind="stable").head(k)
    y, pred = frame.log2_mic.to_numpy(), frame.prediction.to_numpy()
    return {
        "n_candidates": len(frame), "n_selected": k,
        "spearman": float(spearmanr(y, pred).statistic) if np.std(y)>0 and np.std(pred)>0 else np.nan,
        "kendall_tau_b": float(kendalltau(y, pred).statistic) if np.std(y)>0 and np.std(pred)>0 else np.nan,
        "selected_mean_log2_mic": float(selected.log2_mic.mean()),
        "unselected_population_mean_log2_mic": float(frame.log2_mic.mean()),
        "gain_over_random_log2": float(frame.log2_mic.mean()-selected.log2_mic.mean()),
        "regret_to_observed_oracle_log2": float(selected.log2_mic.mean()-oracle.log2_mic.mean()),
    }


def evaluate_saved(scope):
    output = ROOT / "results/reconstruction_2026-10-20" / scope / "benchmark"
    data_path = ROOT / "data/curated" / scope / "benchmark_cells.csv"
    manifest = json.loads((output / "input_manifest.json").read_text())
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Prediction provenance no longer matches the assay data.")
    data = pd.read_csv(data_path, keep_default_na=False, na_values=[""], low_memory=False)
    match = ["publication_id", "target_exact", "medium", "cfu_raw", "salt_type_raw",
             "ionic_strength_raw", "ph_raw", "note_raw"]
    data["ranking_block_id"] = [
        stable_hash(list(row), "SCREEN:") for row in data[match].fillna("").itertuples(index=False, name=None)
    ]
    metadata = data[["assay_cell_id", "ranking_block_id"]]
    ranking, calibration = [], []
    for path in sorted((output / "predictions").glob("*.csv.gz")):
        predictions = pd.read_csv(path, keep_default_na=False, na_values=[""])
        joined = predictions.merge(metadata, on="assay_cell_id", how="left", validate="many_to_one")
        if joined.ranking_block_id.isna().any():
            raise ValueError("A saved prediction could not be mapped to its original assay cell.")
        header = {k: joined.iloc[0][k] for k in ["protocol", "model", "seed", "fold"]}
        for stratum, frame in joined.groupby("novelty"):
            if len(frame) < 10 or frame.prediction.std() == 0:
                continue
            y, prediction = frame.log2_mic.to_numpy(), frame.prediction.to_numpy()
            slope = np.cov(prediction, y, ddof=0)[0,1]/np.var(prediction)
            calibration.append({**header, "novelty": stratum, "n": len(frame),
                                "intercept": float(y.mean()-slope*prediction.mean()), "slope": float(slope),
                                "bias": float((prediction-y).mean()),
                                "note": "Test-set diagnostic only; predictions are not recalibrated on test labels."})
        for block, frame in joined.groupby("ranking_block_id"):
            if len(frame) >= 5:
                ranking.append({**header, "ranking_block_id": block,
                                "publication_id": frame.publication_id.iloc[0],
                                **screening_metrics(frame)})
    pd.DataFrame(calibration).to_csv(output / "calibration_diagnostics.csv", index=False)
    rank = pd.DataFrame(ranking)
    rank.to_csv(output / "within_assay_screening.csv", index=False)
    if len(rank):
        numeric = ["spearman", "kendall_tau_b", "gain_over_random_log2", "regret_to_observed_oracle_log2"]
        # Collapse repeated folds, seeds and blocks within each paper first.
        study = rank.groupby(["protocol", "model", "publication_id"])[numeric].mean().reset_index()
        study.to_csv(output / "screening_by_publication.csv", index=False)
        summary = study.groupby(["protocol", "model"]).agg(
            publications=("publication_id", "nunique"),
            mean_within_assay_spearman=("spearman", "mean"),
            mean_screening_gain_log2=("gain_over_random_log2", "mean"),
            mean_regret_log2=("regret_to_observed_oracle_log2", "mean")).reset_index()
        summary.to_csv(output / "screening_summary.csv", index=False)
    print(f"Calibration strata: {len(calibration)}; screening blocks across fits: {len(rank)}", flush=True)

"""Independent numerical and provenance review of the additive editorial phase."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .editorial_common import ROOT, BASE, OUT, sha, utc, read_json, write_json, read_data, verify_preservation


def plans(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def check_calibration():
    from .assay_calibration import load_split, MODELS, METHODS, KS, STAT_NAMES, SEED
    data = read_data()
    destination = OUT/"assay_calibration"
    reports, support = [], []
    maximum_point_difference, maximum_radius_difference = 0., 0.
    for protocol, fold in [("temporal", 0)]+[("publication", f) for f in range(5)]:
        folder = destination/f"{protocol}_{fold}"
        receipt = read_json(folder/"completion.json")
        for name, expected in receipt["files_sha256"].items():
            assert sha(folder/name) == expected
        key, fit, variance_idx, conformal_idx, test, full_r, proper_r, sources = load_split(data, protocol, fold)
        variance_table = pd.read_csv(folder/"training_variance_components.csv")
        expected_stats = pd.read_csv(folder/"publication_statistics.csv.gz")
        radii = pd.read_csv(folder/"conformal_radii.csv.gz")
        frequency = data.iloc[fit].chemical_id.value_counts().to_dict()
        records_checked, radii_checked, candidate_plans = 0, 0, 0
        for grouping in ["publication", "publication_target"]:
            columns = ["publication_id"]+(["target_exact"] if grouping == "publication_target" else [])
            frame = data.iloc[variance_idx][columns].copy()
            independently_estimated = []
            for m, name in enumerate(MODELS):
                frame["r"] = proper_r[variance_idx,m]
                grouped = frame.groupby(columns).r.agg(["size", "mean", "var"])
                n, g = len(frame), len(grouped)
                within = float((grouped["var"].fillna(0)*(grouped["size"]-1)).sum()/(n-g))
                between = float((grouped["size"]*(grouped["mean"]-frame.r.mean())**2).sum()/(g-1))
                n0 = (n-(grouped["size"]**2).sum()/n)/(g-1)
                b = max(0., (between-within)/n0)
                recorded = variance_table.query("grouping == @grouping and model == @name").iloc[0]
                assert np.allclose([b, within], recorded[["sigma_b2","sigma_e2"]].to_numpy(float), atol=1e-10, rtol=0)
                independently_estimated.append((b, within))
            for strategy in ["random", "standard"]:
                cal = plans(folder/f"anchors_{grouping}_{strategy}_conformal.jsonl.gz")
                test_plans = plans(folder/f"anchors_{grouping}_{strategy}_test.jsonl.gz")
                expected_draws = 200 if strategy == "random" else 1
                for cohort, blocks, allowed in [("test", test_plans, test), ("conformal", cal, conformal_idx)]:
                    covered_rows = set()
                    eligible_rows = set()
                    for _, group in data.iloc[allowed].groupby(columns):
                        if group.chemical_id.nunique() >= 6:
                            eligible_rows.update(group.index)
                    for block in blocks:
                        rows = np.array(block["cell_indices"])
                        anchors = np.array(block["anchor_indices_by_draw"])
                        assert anchors.shape == (expected_draws, 5)
                        assert set(rows) <= set(allowed) and not covered_rows & set(rows)
                        covered_rows.update(rows)
                        assert set(anchors.ravel()) <= set(rows)
                        assert data.loc[rows, "chemical_id"].nunique() >= 6
                        assert all(data.loc[a, "chemical_id"].nunique() == 5 for a in anchors)
                        assert all(data.loc[rows, c].nunique() == 1 for c in columns)
                        if strategy == "standard":
                            counts = [frequency.get(c, 0) for c in data.loc[anchors[0], "chemical_id"]]
                            eligible_counts = sorted([frequency.get(c,0) for c in data.loc[rows,"chemical_id"].unique()], reverse=True)
                            assert counts == eligible_counts[:5]
                        candidate_plans += expected_draws
                    assert covered_rows == eligible_rows
                    if cohort == "test":
                        for k in KS:
                            historical = sum(sum(frequency.get(c,0)>0 for c in data.loc[np.array(b["anchor_indices_by_draw"])[0,:k],"chemical_id"]) for b in blocks) if strategy == "standard" else None
                            support.append({"protocol": protocol, "fold": fold, "grouping": grouping, "strategy": strategy,
                                "k": k, "all_test_cells": len(test), "all_test_publications": data.iloc[test].publication_id.nunique(),
                                "eligible_cells_before_anchors": len(eligible_rows), "eligible_publications": len({b["publication_id"] for b in blocks}),
                                "eligible_blocks": len(blocks), "anchor_measurements": k*len(blocks),
                                "standard_anchors_with_training_frequency_above_zero": historical})
                # Independently re-evaluate selected publications, including the largest.
                pub_sizes = {p:sum(len(b["cell_indices"]) for b in test_plans if b["publication_id"] == p)
                             for p in {b["publication_id"] for b in test_plans}}
                selected_pubs = sorted({min(pub_sizes), max(pub_sizes), max(pub_sizes, key=pub_sizes.get)})
                for k in KS:
                    for publication in selected_pubs:
                        selected = [b for b in test_plans if b["publication_id"] == publication]
                        for m, model in enumerate(MODELS):
                            bvar, evar = independently_estimated[m]
                            w = k*bvar/(k*bvar+evar) if k*bvar+evar else 0.
                            for method, factor in zip(METHODS, [0.,1.,w]):
                                observed_sums = np.zeros(7)
                                for block in selected:
                                    rows = np.array(block["cell_indices"])
                                    draws = block["anchor_indices_by_draw"]
                                    for anchors in draws:
                                        anchors = np.array(anchors[:k])
                                        queried = rows[~data.loc[rows, "chemical_id"].isin(set(data.loc[anchors,"chemical_id"])).to_numpy()]
                                        assert len(queried) and not set(queried) & set(anchors)
                                        offset = factor*np.mean(full_r[anchors,m])
                                        error = offset-full_r[queried,m]
                                        y = data.loc[queried,"log2_mic"].to_numpy()
                                        observed_sums += np.array([len(queried), y.sum(), (y*y).sum(), abs(error).sum(),
                                                                  (error*error).sum(), error.sum(), np.sum(abs(error)<=1)])/len(draws)
                                recorded = expected_stats[(expected_stats.grouping == grouping) & (expected_stats.strategy == strategy) &
                                    (expected_stats.k == k) & (expected_stats.branch == "point_frozen") & (expected_stats.model == model) &
                                    (expected_stats.method == method) & (expected_stats.publication_id == publication)]
                                assert len(recorded) == 1
                                difference = np.max(abs(recorded.iloc[0][STAT_NAMES[:7]].to_numpy(float)-observed_sums))
                                maximum_point_difference = max(maximum_point_difference, float(difference))
                                assert difference < 2e-7
                                records_checked += 1
                    # Recompute conformal order statistics directly from remaining query residuals.
                    for draw in ([0,99,199] if strategy == "random" else [0]):
                        for m, model in enumerate(MODELS):
                            bvar, evar = independently_estimated[m]
                            w = k*bvar/(k*bvar+evar) if k*bvar+evar else 0.
                            for method, factor in zip(METHODS, [0.,1.,w]):
                                by_pub = {}
                                for block in cal:
                                    rows = np.array(block["cell_indices"])
                                    anchors = np.array(block["anchor_indices_by_draw"][draw][:k])
                                    queried = rows[~data.loc[rows,"chemical_id"].isin(set(data.loc[anchors,"chemical_id"])).to_numpy()]
                                    error = abs(proper_r[queried,m]-factor*np.mean(proper_r[anchors,m]))
                                    by_pub[block["publication_id"]] = max(by_pub.get(block["publication_id"], 0.), float(error.max()))
                                scores = sorted(by_pub.values())
                                found = radii[(radii.grouping == grouping) & (radii.strategy == strategy) & (radii.k == k) &
                                    (radii.draw == draw) & (radii.model == model) & (radii.method == method)].iloc[0]
                                for coverage, column in [(.9,"radius90"),(.95,"radius95")]:
                                    rank = int(np.ceil((len(scores)+1)*coverage))
                                    value = scores[rank-1] if rank <= len(scores) else np.inf
                                    difference = abs(value-found[column]) if np.isfinite(value) else 0.
                                    assert np.isfinite(value) == np.isfinite(found[column]) and difference < 1e-10
                                    maximum_radius_difference = max(maximum_radius_difference, float(difference))
                                    radii_checked += 1
        reports.append({"split": key, "all_anchor_draw_plans_validated": candidate_plans,
                        "independent_publication_statistic_rows_recomputed": records_checked,
                        "independent_conformal_radii_recomputed": radii_checked, "passed": True})
        print("Few-shot independent audit: "+key, flush=True)
    pd.DataFrame(support).to_csv(destination/"support_and_reference_availability.csv", index=False)
    report = {"created_utc": utc(), "passed": True, "splits": reports,
        "maximum_sufficient_statistic_difference": maximum_point_difference,
        "maximum_conformal_radius_difference": maximum_radius_difference,
        "scope": "All anchor plans, eligibility sets, training-only variance estimates and source hashes checked; independent losses for three publications per setting, including the largest; conformal radii independently recomputed at first, middle and last random draws.",
        "source_sha256": {"metrics.csv":sha(destination/"metrics.csv"), "paired_improvements.csv":sha(destination/"paired_improvements.csv")}}
    write_json(destination/"independent_audit.json", report)
    return report


def check_tail():
    folder = OUT/"posterior_tail_gaps"
    result = read_json(folder/"completion.json")
    manifest = read_json(OUT/"preservation_manifest.json")["files"]
    assert result["fits"] == 5 and result["causal_fraction_identified"] is False
    chains, metrics = 0, 0
    for receipt_path in folder.glob("*/extraction.json"):
        receipt = read_json(receipt_path)
        for source in receipt["source_chains"]:
            assert source["sha256"] == manifest[source["source"]]["sha256"]
            chains += 1
        completion = read_json(receipt_path.parent/"completion.json")
        assert sha(receipt_path.parent/"summary.csv") == completion["summary_sha256"]
        assert all(v["passed"] for v in completion["original_predictive_checks_reconciliation"].values())
        metrics += len(completion["original_predictive_checks_reconciliation"])
    assert chains == 20 and metrics == 25
    table = pd.read_csv(folder/"tail_gap_summary.csv")
    assert not table.query("fit == 'exact_prior1.5'").positive_gap_fraction_defined.any()
    baseline = table[table.scenario.eq("gaussian_unselected")]
    assert len(baseline) == 5
    report = {"created_utc": utc(), "passed": True, "original_chain_sha256_reconciled": chains,
              "original_predictive_summaries_reconciled": metrics,
              "negative_exact_only_gap_explicit": True, "causal_fraction_not_identified": True,
              "summary_sha256": sha(folder/"tail_gap_summary.csv")}
    write_json(folder/"independent_audit.json", report)
    return report


def check_deep():
    from .deep_contextual import DEST, MODELS, CONFIG, task_list, fit_preprocessor, transform, predict, MultimodalRegressor, seed_everything
    data = read_data()
    completion = read_json(DEST/"completion.json")
    assert completion["splits"] == 91 and completion["prediction_files"] == 182
    checked = []
    for protocol, seed, fold in task_list():
        key = f"{protocol}_{seed}_{fold}"
        assignment = read_json(DEST/"fits"/key/"assignment.json")
        for model in MODELS:
            folder = DEST/"fits"/key
            result = read_json(folder/f"{model}.json")
            history = read_json(folder/f"{model}_early_stopping.json")
            best, selected, stale = float("inf"), 0, 0
            for row in history:
                if row["validation_mae_log2"] < best-CONFIG["minimum_mae_improvement_log2"]:
                    best, selected, stale = row["validation_mae_log2"], row["epoch"], 0
                else:
                    stale += 1
            assert selected == result["selected_epochs"]
            assert stale >= CONFIG["patience"] or len(history) == CONFIG["max_epochs"]
            refit = read_json(folder/f"{model}_refit_history.json")
            assert len(refit) == selected and all("validation_mae_log2" not in r for r in refit)
            assert sha(folder/f"{model}.pt") == result["state_dict_sha256"]
            checked.append({"split":key,"model":model,"selected_epochs":selected})
    # Replay both complete temporal prediction files from saved state dictionaries.
    raw = np.load(DEST/"features/matrix.npy", mmap_mode="r")
    role = pd.read_csv(BASE/"benchmark/split_temporal_20261020_0.csv.gz").role
    train, test = np.flatnonzero(role.eq("train")), np.flatnonzero(role.eq("test"))
    replay = []
    for name in MODELS:
        folder = DEST/"fits/temporal_20261020_0"
        prep = read_json(folder/f"{name}_preprocessor.json")
        assert prep == fit_preprocessor(raw, data, train)
        seed_everything(20261020)
        network = MultimodalRegressor(raw.shape[1]-3328,len(prep["vocab"]["target_exact"]),len(prep["vocab"]["medium"]),name.endswith("context"))
        network.load_state_dict(torch.load(folder/f"{name}.pt", map_location="cpu", weights_only=True))
        tensors = transform(raw,data,test,prep)
        actual = predict(network,tensors)*prep["y_scale"]+prep["y_mean"]
        saved = pd.read_csv(DEST/"predictions"/f"temporal_20261020_0_{name}.csv.gz")
        assert saved.assay_cell_id.tolist() == data.iloc[test].assay_cell_id.tolist()
        difference = float(np.max(abs(actual-saved.prediction.to_numpy())))
        assert difference < 1e-5
        replay.append({"model":name,"predictions_replayed":len(test),"maximum_difference":difference})
    report = {"created_utc":utc(),"passed":True,"fits_checked":len(checked),"epoch_selection_checks":checked,
              "full_temporal_checkpoint_replays":replay,"summary_sha256":sha(DEST/"metrics.csv"),
              "comparison_sha256":sha(DEST/"paired_improvements.csv")}
    write_json(DEST/"independent_audit.json",report)
    return report

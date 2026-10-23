"""Training-only empirical Bayes assay offsets with strictly held-out queries.

Two predictors are kept distinct: frozen full-training OOF predictions for point
accuracy, and the frozen proper-fit predictor for split-conformal evaluation.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .editorial_common import (ROOT, BASE, OUT, DATA, sha, utc, read_json, write_json,
                               lock_json, read_data, frozen_split, load_prediction)

DEST = OUT/"assay_calibration"
MODELS = ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb"]
METHODS = ["uncalibrated", "unshrunk_anchor_mean", "empirical_bayes"]
KS = [1, 2, 3, 5]
SEED, DRAWS, BOOTSTRAPS = 20261020, 200, 2000
STAT_NAMES = ["query_n", "sum_y", "sum_y2", "sum_absolute_error", "sum_squared_error",
              "sum_signed_error", "sum_within_one", "sum_covered90", "sum_covered95",
              "sum_width90", "sum_width95"]


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()[:8], "little")


def variance_components(residual, group):
    """Unbalanced one-way ANOVA moments; intercept mean is not a test correction."""
    residual = np.asarray(residual, float)
    labels, groups = pd.factorize(group)
    n, g = len(residual), len(groups)
    counts = np.bincount(labels).astype(float)
    means = np.bincount(labels, weights=residual)/counts
    grand = float(residual.mean())
    if g < 2 or n <= g:
        return {"sigma_b2": 0., "sigma_e2": float(np.var(residual)) if n else 0.,
                "n": n, "groups": g, "identified": False, "grand_residual_mean": grand}
    within = np.sum((residual-means[labels])**2)/(n-g)
    between = np.sum(counts*(means-grand)**2)/(g-1)
    effective = (n-np.sum(counts**2)/n)/(g-1)
    return {"sigma_b2": float(max(0., (between-within)/effective)), "sigma_e2": float(within),
            "n": n, "groups": g, "identified": True, "grand_residual_mean": grand,
            "between_ms": float(between), "effective_group_size": float(effective),
            "nonnegative_boundary": bool(between <= within)}


def shrinkage(k, sigma_b2, sigma_e2):
    denominator = k*sigma_b2+sigma_e2
    return k*sigma_b2/denominator if denominator > 0 else 0.


def group_keys(frame, grouping):
    if grouping == "publication":
        return frame.publication_id.astype(str)
    return pd.Series(list(zip(frame.publication_id.astype(str), frame.target_exact.astype(str))), index=frame.index)


def calibration_halves(data, calibration, split_key):
    pubs = np.array(sorted(data.iloc[calibration].publication_id.astype(str).unique()))
    selected = np.random.default_rng(seed_for(split_key, "variance_partition")).permutation(pubs)
    variance_pubs = set(selected[:len(selected)//2])
    mask = data.iloc[calibration].publication_id.astype(str).isin(variance_pubs).to_numpy()
    return calibration[mask], calibration[~mask]


def build_plan(data, indices, grouping, strategy, frequency, split_key, cohort):
    frame = data.iloc[indices]
    blocks, eligibility = [], []
    for label, local_rows in frame.groupby(group_keys(frame, grouping), sort=True).groups.items():
        rows = np.array(sorted(local_rows), dtype=int)
        chemical_groups = {str(c): np.array(v, int) for c,v in data.loc[rows].groupby("chemical_id").groups.items()}
        entities = sorted(chemical_groups)
        eligible = len(entities) >= max(KS)+1
        label_string = json.dumps(label)
        block_id = hashlib.sha256(label_string.encode()).hexdigest()[:20]
        eligibility.append({"split": split_key, "cohort": cohort, "grouping": grouping, "block_id": block_id,
                            "publication_id": str(data.loc[rows[0], "publication_id"]), "target_exact": str(label[1]) if isinstance(label, tuple) else "ALL",
                            "cells": len(rows), "chemical_entities": len(entities), "eligible": eligible,
                            "entities_previously_measured": sum(frequency.get(c, 0)>0 for c in entities)})
        if not eligible:
            continue
        rng = np.random.default_rng(seed_for(split_key, grouping, cohort, block_id, strategy))
        if strategy == "random":
            anchor_rows = []
            for _ in range(DRAWS):
                order = rng.choice(len(entities), max(KS), replace=False)
                anchor_rows.append([rng.choice(chemical_groups[entities[j]]) for j in order])
            anchors = np.asarray(anchor_rows, int)
        else:
            standard = sorted(entities, key=lambda c: (-frequency.get(c, 0), seed_for("frequency_tie", c)))[:max(KS)]
            anchors = np.asarray([[min(chemical_groups[c], key=lambda j: seed_for("standard_cell", data.loc[j, "assay_cell_id"])) for c in standard]], int)
        blocks.append({"block_id": block_id, "publication_id": str(data.loc[rows[0], "publication_id"]),
                       "rows": rows, "anchors": anchors})
    return blocks, eligibility


def query_mask(data, block, k):
    selected = data.chemical_id.to_numpy()[block["anchors"][:, :k]]
    chemicals = data.chemical_id.to_numpy()[block["rows"]]
    # All cells of each revealed chemical entity are excluded, including different contexts.
    return ~np.any(chemicals[None, :, None] == selected[:, None, :], axis=2)


def save_plan(path, blocks, data):
    with gzip.open(path, "wt") as handle:
        for block in blocks:
            handle.write(json.dumps({"block_id": block["block_id"], "publication_id": block["publication_id"],
                "cell_indices": block["rows"].tolist(), "anchor_indices_by_draw": block["anchors"].tolist()})+"\n")


def weights_for(k, variance):
    return np.array([[0., 1., shrinkage(k, v["sigma_b2"], v["sigma_e2"])] for v in variance])


def calibration_scores(data, blocks, residual, k, weight):
    draws = len(blocks[0]["anchors"]) if blocks else DRAWS
    pubs = sorted({b["publication_id"] for b in blocks})
    pub_index = {p:i for i,p in enumerate(pubs)}
    scores = np.full((draws, len(pubs), len(MODELS), len(METHODS)), -np.inf)
    for block in blocks:
        q = query_mask(data, block, k)
        r = residual[block["rows"]]
        anchor_mean = residual[block["anchors"][:, :k]].mean(axis=1)
        for method in range(len(METHODS)):
            error = np.abs(r[None, :, :]-anchor_mean[:, None, :]*weight[None, None, :, method])
            maxima = np.where(q[:, :, None], error, -np.inf).max(axis=1)
            pi = pub_index[block["publication_id"]]
            scores[:, pi, :, method] = np.maximum(scores[:, pi, :, method], maxima)
    assert np.isfinite(scores).all() if len(pubs) else True
    return pubs, scores


def score_radii(scores, coverage):
    n = scores.shape[1]
    rank = int(np.ceil((n+1)*coverage))
    if n == 0 or rank > n:
        return np.full((scores.shape[0], scores.shape[2], scores.shape[3]), np.inf)
    return np.sort(scores, axis=1)[:, rank-1]


def evaluate_queries(data, blocks, residual, k, weight, radii=None):
    pubs = sorted({b["publication_id"] for b in blocks})
    pub_index = {p:i for i,p in enumerate(pubs)}
    draws = len(blocks[0]["anchors"]) if blocks else DRAWS
    stats = np.zeros((len(pubs), len(MODELS), len(METHODS), len(STAT_NAMES)))
    pub_coverage = np.ones((draws, len(pubs), len(MODELS), len(METHODS), 2))
    budget = np.zeros((len(pubs), 3))  # blocks, revealed cells, previously measured reference cells
    y_all = data.log2_mic.to_numpy(float)
    for block in blocks:
        q = query_mask(data, block, k)
        assert q.any(axis=1).all()
        r = residual[block["rows"]]
        y = y_all[block["rows"]]
        mean = residual[block["anchors"][:, :k]].mean(axis=1)
        pi = pub_index[block["publication_id"]]
        budget[pi, 0] += 1
        budget[pi, 1] += k
        for method in range(len(METHODS)):
            # prediction minus truth; offset = shrinkage * (truth - prediction) at anchors.
            error = mean[:, None, :]*weight[None, None, :, method]-r[None, :, :]
            absolute = np.abs(error)
            masks = q[:, :, None]
            values = np.stack([
                np.broadcast_to(q.sum(axis=1)[:, None], (draws, len(MODELS))),
                np.broadcast_to((q*y).sum(axis=1)[:, None], (draws, len(MODELS))),
                np.broadcast_to((q*y**2).sum(axis=1)[:, None], (draws, len(MODELS))),
                (absolute*masks).sum(axis=1), (error**2*masks).sum(axis=1),
                (error*masks).sum(axis=1), ((absolute <= 1)*masks).sum(axis=1),
                np.zeros((draws, len(MODELS))), np.zeros((draws, len(MODELS))),
                np.zeros((draws, len(MODELS))), np.zeros((draws, len(MODELS))),
            ], axis=-1)
            if radii is not None:
                for ci in range(2):
                    radius = radii[ci][:, :, method]
                    covered = absolute <= radius[:, None, :]
                    values[:, :, 7+ci] = (covered*masks).sum(axis=1)
                    values[:, :, 9+ci] = 2*radius*q.sum(axis=1)[:, None]
                    pub_coverage[:, pi, :, method, ci] *= np.all(covered | ~masks, axis=1)
            stats[pi, :, method] += values.mean(axis=0)
    return pubs, stats, pub_coverage.mean(axis=0), budget


def stats_metrics(stats):
    n, y, y2, absolute, square, bias, one, c90, c95, w90, w95 = np.moveaxis(stats, -1, 0)
    variance = y2/n-(y/n)**2
    return {"mae_log2": absolute/n, "rmse_log2": np.sqrt(square/n),
            "r2": np.divide(square/n, variance, out=np.full_like(square, np.nan), where=variance>0)*-1+1,
            "calibration_bias_log2": bias/n, "within_one_dilution": one/n,
            "coverage90": c90/n, "coverage95": c95/n, "width90_log2": w90/n, "width95_log2": w95/n}


def load_split(data, protocol, fold):
    key = f"{protocol}_{SEED}_{fold}"
    train, test = frozen_split(data, BASE/"benchmark"/f"split_{key}.csv.gz")
    proper_path = BASE/"prediction_intervals"/f"split_{protocol}_{fold}.csv.gz"
    roles = pd.read_csv(proper_path, keep_default_na=False)
    assert roles.assay_cell_id.tolist() == data.assay_cell_id.tolist()
    proper = np.flatnonzero(roles.role.eq("fit"))
    calibration = np.flatnonzero(roles.role.eq("calibration"))
    assert set(proper) | set(calibration) == set(train)
    assert set(np.flatnonzero(roles.role.eq("test"))) == set(test)
    variance, conformal = calibration_halves(data, calibration, key)
    partitions = [proper, variance, conformal, test]
    for i in range(4):
        for j in range(i):
            assert not set(data.iloc[partitions[i]].publication_id) & set(data.iloc[partitions[j]].publication_id)
    full_residual = np.full((len(data), len(MODELS)), np.nan)
    proper_residual = np.full_like(full_residual, np.nan)
    sources = {str(proper_path.relative_to(ROOT)): sha(proper_path)}
    for m, name in enumerate(MODELS):
        for branch, path, indices in [
            ("full", BASE/"benchmark/predictions"/f"{key}_{name}.csv.gz", test),
            ("proper", BASE/"prediction_intervals"/f"{protocol}_{fold}_{name}.csv.gz", test),
            ("proper", BASE/"prediction_intervals"/f"{protocol}_{fold}_{name}_calibration.csv.gz", calibration),
        ]:
            pred = load_prediction(path, data, indices)
            target = full_residual if branch == "full" else proper_residual
            target[indices, m] = pred.log2_mic.to_numpy()-pred.prediction.to_numpy()
            sources[str(path.relative_to(ROOT))] = sha(path)
    return key, proper, variance, conformal, test, full_residual, proper_residual, sources


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    data = read_data()
    lock_json(DEST/"configuration.json", {"protocol_sha256": sha(OUT/"locked_editorial_protocol.json"),
        "benchmark_data_sha256": sha(DATA), "algorithm_sha256": sha(Path(__file__)),
        "models": MODELS, "methods": METHODS, "k": KS, "draws": DRAWS, "seed": SEED, "bootstrap_repeats": BOOTSTRAPS})
    for protocol, fold in [("temporal", 0)]+[("publication", i) for i in range(5)]:
        folder = DEST/f"{protocol}_{fold}"
        if (folder/"completion.json").exists():
            print(f"Cached {protocol}_{fold}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        key, fit, variance_idx, conformal_idx, test, full_r, proper_r, sources = load_split(data, protocol, fold)
        frequency = data.iloc[fit].chemical_id.value_counts().to_dict()
        write_json(folder/"partitions.json", {"split": key, "proper_fit": data.iloc[fit].assay_cell_id.tolist(),
            "variance": data.iloc[variance_idx].assay_cell_id.tolist(), "conformal": data.iloc[conformal_idx].assay_cell_id.tolist(),
            "test": data.iloc[test].assay_cell_id.tolist(), "source_sha256": sources,
            "frequency_counts": frequency, "frequency_source": "proper-fit rows only"})
        records, eligibility, variance_rows, radius_rows = [], [], [], []
        for grouping in ["publication", "publication_target"]:
            groups = group_keys(data.iloc[variance_idx], grouping)
            variance = [variance_components(proper_r[variance_idx, m], groups) for m in range(len(MODELS))]
            variance_rows.extend([{"grouping": grouping, "model": model, **v} for model,v in zip(MODELS, variance)])
            for strategy in ["random", "standard"]:
                cal_plan, cal_elig = build_plan(data, conformal_idx, grouping, strategy, frequency, key, "conformal")
                test_plan, test_elig = build_plan(data, test, grouping, strategy, frequency, key, "test")
                if strategy == "random":
                    eligibility.extend(cal_elig+test_elig)
                save_plan(folder/f"anchors_{grouping}_{strategy}_conformal.jsonl.gz", cal_plan, data)
                save_plan(folder/f"anchors_{grouping}_{strategy}_test.jsonl.gz", test_plan, data)
                for k in KS:
                    weights = weights_for(k, variance)
                    cal_pubs, scores = calibration_scores(data, cal_plan, proper_r, k, weights)
                    radii = [score_radii(scores, c) for c in [.90, .95]]
                    for draw in range(scores.shape[0]):
                        for m, model in enumerate(MODELS):
                            for a, method in enumerate(METHODS):
                                radius_rows.append({"grouping": grouping, "strategy": strategy, "k": k, "draw": draw,
                                    "model": model, "method": method, "calibration_publications": len(cal_pubs),
                                    "radius90": radii[0][draw,m,a], "radius95": radii[1][draw,m,a]})
                    for branch, residual, bounds in [("point_frozen", full_r, None), ("interval_proper_fit", proper_r, radii)]:
                        pubs, stats, pub_cover, budget = evaluate_queries(data, test_plan, residual, k, weights, bounds)
                        for p, pub in enumerate(pubs):
                            for m, model in enumerate(MODELS):
                                for a, method in enumerate(METHODS):
                                    records.append({"protocol": protocol, "fold": fold, "grouping": grouping,
                                        "strategy": strategy, "k": k, "branch": branch, "publication_id": pub,
                                        "model": model, "method": method, "draws": DRAWS if strategy == "random" else 1,
                                        "eligible_blocks": int(budget[p, 0]), "anchor_measurements": int(budget[p, 1]),
                                        "publication_coverage90": float(pub_cover[p,m,a,0]) if bounds is not None else np.nan,
                                        "publication_coverage95": float(pub_cover[p,m,a,1]) if bounds is not None else np.nan,
                                        **dict(zip(STAT_NAMES, stats[p,m,a]))})
                    print(f"{protocol} {fold} {grouping} {strategy} k={k}; test publications={len(pubs)}, conformal publications={len(cal_pubs)}", flush=True)
        pd.DataFrame(records).to_csv(folder/"publication_statistics.csv.gz", index=False)
        pd.DataFrame(eligibility).to_csv(folder/"block_eligibility.csv", index=False)
        pd.DataFrame(variance_rows).to_csv(folder/"training_variance_components.csv", index=False)
        pd.DataFrame(radius_rows).to_csv(folder/"conformal_radii.csv.gz", index=False)
        write_json(folder/"completion.json", {"created_utc": utc(), "passed": True,
            "statistics_rows": len(records), "source_sha256": sources,
            "files_sha256": {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "completion.json"}})
    summarize()


def summarize():
    pieces = []
    for protocol, fold in [("temporal", 0)]+[("publication", i) for i in range(5)]:
        folder = DEST/f"{protocol}_{fold}"
        receipt = read_json(folder/"completion.json")
        for name, expected in receipt["files_sha256"].items():
            assert sha(folder/name) == expected
        pieces.append(pd.read_csv(folder/"publication_statistics.csv.gz", low_memory=False))
    all_stats = pd.concat(pieces, ignore_index=True)
    summaries, differences = [], []
    grouping_keys = ["protocol", "grouping", "strategy", "k", "branch"]
    with threadpool_limits(limits=4):
        for key, frame in all_stats.groupby(grouping_keys, sort=True):
            common = dict(zip(grouping_keys, key))
            pubs = sorted(frame.publication_id.unique())
            rng = np.random.default_rng(seed_for("bootstrap", key[0], key[1]))
            bootstrap_weights = rng.multinomial(len(pubs), np.full(len(pubs), 1/len(pubs)), size=BOOTSTRAPS)
            point, boot, arrays = {}, {}, {}
            for (model, method), block in frame.groupby(["model", "method"]):
                assert block.publication_id.is_unique
                ordered = block.set_index("publication_id").loc[pubs]
                values = ordered[STAT_NAMES].to_numpy(float)
                arrays[(model, method)] = values
                point[(model, method)] = stats_metrics(values.sum(axis=0))
                # Infinity can be a legitimate conformal radius; avoid 0 * inf -> NaN.
                finite_values = np.where(np.isfinite(values), values, 0)
                resampled = bootstrap_weights@finite_values
                for col in np.flatnonzero(np.isinf(values).any(axis=0)):
                    resampled[(bootstrap_weights[:, np.isinf(values[:, col])].sum(axis=1)>0), col] = np.inf
                boot[(model, method)] = stats_metrics(resampled)
                row = {**common, "model": model, "method": method, "publications": len(pubs),
                       "expected_query_evaluations_per_draw": float(values[:,0].sum()),
                       "anchor_measurements": int(ordered.anchor_measurements.sum()),
                       "eligible_blocks": int(ordered.eligible_blocks.sum()),
                       "limited_independent_support": len(pubs) < 10}
                for name, value in point[(model, method)].items():
                    if key[-1] == "point_frozen" and name.startswith(("coverage", "width")):
                        continue
                    row[name] = float(value)
                    # Quantiles of infinity are infinity, not an interpolated NaN.
                    ordered_boot = np.sort(boot[(model, method)][name])
                    row[name+"_ci_low"] = float(ordered_boot[int(np.floor(.025*(BOOTSTRAPS-1)))])
                    row[name+"_ci_high"] = float(ordered_boot[int(np.ceil(.975*(BOOTSTRAPS-1)))])
                if key[-1] == "interval_proper_fit":
                    row["publication_coverage90"] = float(ordered.publication_coverage90.mean())
                    row["publication_coverage95"] = float(ordered.publication_coverage95.mean())
                row["equal_publication_mae_log2"] = float(np.mean(values[:,3]/values[:,0]))
                summaries.append(row)
            for base, context in [("sequence_hgb", "sequence_context_hgb"), ("esm_hgb", "esm_context_hgb")]:
                for alternative in ["unshrunk_anchor_mean", "empirical_bayes"]:
                    pairs = {
                        "calibration_gain": ((base, "uncalibrated"), (base, alternative)),
                        "metadata_gain_same_queries": ((base, "uncalibrated"), (context, "uncalibrated")),
                        "calibration_minus_metadata_gain": ((context, "uncalibrated"), (base, alternative)),
                        "shrinkage_over_raw_gain": ((base, "unshrunk_anchor_mean"), (base, "empirical_bayes")),
                    }
                    row = {**common, "base_model": base, "context_model": context, "calibration_method": alternative,
                           "publications": len(pubs), "bootstrap_repeats": BOOTSTRAPS}
                    for label, (a,b) in pairs.items():
                        assert np.array_equal(arrays[a][:,:3], arrays[b][:,:3])
                        gain = point[a]["mae_log2"]-point[b]["mae_log2"]
                        sampled = boot[a]["mae_log2"]-boot[b]["mae_log2"]
                        row[label+"_log2"] = float(gain)
                        row[label+"_ci_low"], row[label+"_ci_high"] = np.quantile(sampled, [.025,.975])
                    differences.append(row)
    pd.DataFrame(summaries).to_csv(DEST/"metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(DEST/"paired_improvements.csv", index=False)
    all_stats.to_csv(DEST/"publication_statistics.csv.gz", index=False)
    write_json(DEST/"completion.json", {"created_utc": utc(), "passed": True,
        "metrics_rows": len(summaries), "paired_comparisons": len(differences), "bootstrap_repeats": BOOTSTRAPS,
        "branch_separation": "Point accuracy uses original full-training benchmark predictions. Conformal results use the separate saved proper-fit predictor and disjoint variance/conformity publications.",
        "anchor_scope": "Random anchors are distinct chemical entities; all their remaining cells are excluded from queries. Standard anchors may have zero training frequency if no historical candidate exists in a block.",
        "uncertainty": "Publication bootstrap conditional on fitted predictions, variance estimates and sampled anchor plans. It does not retrain models or include uncertainty in source curation.",
        "metrics_sha256": sha(DEST/"metrics.csv"), "comparisons_sha256": sha(DEST/"paired_improvements.csv")})

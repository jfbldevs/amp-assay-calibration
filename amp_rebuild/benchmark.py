"""Matched, leakage-audited predictive comparisons with saved held-out predictions."""
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .curation import ROOT
from .features import DESCRIPTOR_NAMES, descriptors
from .splits import audit_split, make_splits, novelty

MODEL_NAMES = [
    "target_mean", "context_ridge", "sequence_ridge", "sequence_context_ridge",
    "sequence_hgb", "sequence_context_hgb", "esm_ridge", "esm_context_ridge",
    "esm_hgb", "esm_context_hgb",
    "chem_ridge", "chem_context_ridge", "chem_hgb", "chem_context_hgb",
]
CONTEXT_NUM = ["log10_cfu", "ph", "nacl_reported_concentration"]
CONTEXT_CAT = ["medium", "supplement_category", "cfu_kind"]
CHEM_NUM = ["n_intrachain_bonds", "n_unusual_residues", "n_d_residues"]
CHEM_CAT = ["n_terminus", "c_terminus"]
TARGET_CAT = ["species", "target_exact"]
CONFIG = {
    "protocols": ["random", "sequence", "homology80", "homology50", "publication", "joint", "temporal"],
    "seeds": [20261020, 20261021, 20261022],
    "folds": 5, "models": MODEL_NAMES,
    "primary_metric": "paired reduction in mean absolute log2 MIC error",
    "secondary_metrics": ["R2", "RMSE_log2", "within_one_dilution", "Spearman"],
    "ridge_alpha": 100.0,
    "hgb": {"max_iter": 200, "learning_rate": .06, "max_leaf_nodes": 15,
            "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 10.0,
            "early_stopping": False},
    "esm_pca_components_for_hgb": 32,
    "hyperparameters": "fixed before fitting; no outer-test optimization; no claim of exhaustive SOTA optimization",
    "encoding": "train-fitted one-hot categories, numerical imputation/scaling and ESM PCA",
    "publication_identifier_as_feature": False,
    "uncertainty": "crossed publication-by-sequence Poisson/pigeonhole bootstrap of paired cell errors",
    "endpoint": "exact MIC; mean log2 MIC within identical reported assay cells",
    "censoring": "primary prediction metric restricted to exact labels; separate censored hierarchical sensitivity",
}


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def lock_config(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "analysis_config.json"
    payload = json.dumps(CONFIG, sort_keys=True, indent=2)
    if path.exists() and json.loads(path.read_text()) != CONFIG:
        raise ValueError("Analysis configuration changed. Use a new explicitly versioned run directory.")
    if not path.exists():
        path.write_text(payload)
        (output / "config_lock.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "status": "fixed before benchmark fitting; this is a local analysis lock, not public preregistration",
        }, indent=2))


def feature_table(data, use_embeddings, use_chemistry=False):
    features = data.copy()
    desc = np.stack([descriptors(s) for s in data.backbone_sequence])
    for i, name in enumerate(DESCRIPTOR_NAMES):
        features[name] = desc[:, i]
    embedding_columns = []
    if use_embeddings:
        path = ROOT / "data/features/esm2_t33_650M_UR50D"
        unique = {}
        for key in data.sequence_id.unique():
            file = path / (key.replace(":", "_") + ".npy")
            if not file.exists():
                raise FileNotFoundError(f"Embedding missing: {file}")
            unique[key] = np.load(file)
        matrix = np.stack([unique[key] for key in data.sequence_id])
        embedding_columns = [f"esm_{i}" for i in range(matrix.shape[1])]
        features = pd.concat([features, pd.DataFrame(matrix, columns=embedding_columns)], axis=1)
    chemical_columns = []
    if use_chemistry:
        path = ROOT / "data/features/morgan_radius2"
        unique = {}
        for key in data.chemical_id.unique():
            file = path / (key.replace(":", "_") + ".npy")
            if not file.exists():
                raise FileNotFoundError(f"Chemical representation missing: {file}")
            unique[key] = np.load(file)
        matrix = np.stack([unique[key] for key in data.chemical_id])
        chemical_columns = [f"morgan_{i}" for i in range(matrix.shape[1])]
        features = pd.concat([features, pd.DataFrame(matrix, columns=chemical_columns)], axis=1)
    for col in CONTEXT_NUM + CHEM_NUM:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    for col in CONTEXT_CAT + CHEM_CAT + TARGET_CAT:
        features[col] = features[col].fillna("Unreported").astype(str)
    return features, embedding_columns, chemical_columns


def make_estimator(name, embedding_columns, train_size, seed, chemical_columns=None):
    context = "context" in name
    molecular = name != "context_ridge"
    numeric = []
    categorical = list(TARGET_CAT)
    transformers = []
    if molecular:
        numeric += CHEM_NUM
        categorical += CHEM_CAT
        if name.startswith(("esm", "chem")):
            vector_columns = chemical_columns if name.startswith("chem") else embedding_columns
            if name.endswith("hgb"):
                projection_steps = ([("input_scale", StandardScaler())] if name.startswith("chem") else [])
                projection_steps += [
                    ("pca", PCA(n_components=min(32, train_size-1, len(vector_columns)),
                                svd_solver="randomized", random_state=seed)),
                    ("scale", StandardScaler()),
                ]
                transformers.append(("molecular_vector", Pipeline(projection_steps), vector_columns))
            else:
                transformers.append(("molecular_vector", StandardScaler(), vector_columns))
        else:
            numeric += DESCRIPTOR_NAMES
    if context:
        numeric += CONTEXT_NUM
        categorical += CONTEXT_CAT
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    transformers.append(("categories", OneHotEncoder(
        handle_unknown="ignore", min_frequency=3, max_categories=100, sparse_output=False,
        dtype=np.float32), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    learner = (HistGradientBoostingRegressor(**CONFIG["hgb"], random_state=seed)
               if name.endswith("hgb") else Ridge(alpha=CONFIG["ridge_alpha"], solver="lsqr"))
    return Pipeline([("features", preprocessor), ("model", learner)])


def metrics(y, pred, weight=None):
    y, pred = np.asarray(y), np.asarray(pred)
    w = np.ones(len(y)) if weight is None else np.asarray(weight)
    if len(y) == 0 or w.sum() <= 0:
        return {}
    err = y - pred
    mse = np.average(err**2, weights=w)
    variance = np.average((y - np.average(y, weights=w))**2, weights=w)
    correlation = float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else np.nan
    return {"n": len(y), "r2": 1 - mse / variance if variance > 0 else np.nan,
            "mae_log2": np.average(abs(err), weights=w), "rmse_log2": np.sqrt(mse),
            "within_one_dilution": np.average(abs(err) <= 1, weights=w),
            "spearman": correlation,
            "calibration_bias_log2": np.average(pred-y, weights=w)}


def fit_fold(data, features, embedding_columns, chemical_columns, prediction_dir,
             protocol, seed, fold, tr, te, models):
    """Fit one immutable split; every output name belongs to this split only."""
    split_id = f"{protocol}_{seed}_{fold}"
    y_train, y_test = data.iloc[tr].log2_mic.to_numpy(), data.iloc[te].log2_mic.to_numpy()
    completed = 0
    for name in models:
        target = prediction_dir / f"{split_id}_{name}.csv.gz"
        metric_path = prediction_dir / f"{split_id}_{name}.json"
        if target.exists() and metric_path.exists():
            continue
        started = time.monotonic()
        if name == "target_mean":
            group = data.iloc[tr].groupby("species").log2_mic.agg(["mean", "count"])
            global_mean = y_train.mean()
            average = (group["mean"]*group["count"] + 10*global_mean)/(group["count"]+10)
            predicted = data.iloc[te].species.map(average).fillna(global_mean).to_numpy()
        else:
            model = make_estimator(name, embedding_columns, len(tr), seed, chemical_columns)
            with threadpool_limits(limits=4):
                model.fit(features.iloc[tr], y_train)
                predicted = model.predict(features.iloc[te])
        if not np.isfinite(predicted).all():
            raise ValueError(f"Nonfinite prediction: {split_id} {name}")
        saved = data.iloc[te][["assay_cell_id", "sequence_id", "chemical_id", "publication_id",
                              "cluster_80", "target_exact", "medium", "log2_mic"]].copy()
        saved["prediction"] = predicted
        saved["novelty"] = novelty(data.iloc[tr], data.iloc[te])
        saved["homology80_seen"] = data.iloc[te].cluster_80.isin(set(data.iloc[tr].cluster_80)).to_numpy()
        if "sequence_leakage_group" in data:
            saved["resampling_sequence_id"] = data.iloc[te].sequence_leakage_group.to_numpy()
            saved["structural_identity_group_seen"] = data.iloc[te].sequence_leakage_group.isin(
                set(data.iloc[tr].sequence_leakage_group)).to_numpy()
        saved["protocol"], saved["seed"], saved["fold"], saved["model"] = protocol, seed, fold, name
        temporary = target.with_name(target.name + ".partial")
        saved.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(target)
        result = metrics(y_test, predicted)
        result.update({"protocol": protocol, "seed": seed, "fold": fold, "model": name,
                       "seconds": time.monotonic() - started})
        metric_path.write_text(json.dumps(result, indent=2, default=json_default))
        print(json.dumps(result, default=json_default), flush=True)
        completed += 1
    return {"split": split_id, "new_fits": completed}


_WORKER_INPUTS = None


def initialize_fold_worker(data, features, embedding_columns, chemical_columns, prediction_dir):
    global _WORKER_INPUTS
    _WORKER_INPUTS = (data, features, embedding_columns, chemical_columns, prediction_dir)


def fit_fold_worker(task):
    return fit_fold(*_WORKER_INPUTS, *task)


def run_benchmark(scope="sample", protocols=None, seeds=None, models=None, workers=1):
    output = ROOT / "results/reconstruction_2026-10-20" / scope / "benchmark"
    lock_config(output)
    protocols = protocols or CONFIG["protocols"]
    seeds = seeds or CONFIG["seeds"]
    models = models or MODEL_NAMES
    source = ROOT / "data/curated" / scope / "benchmark_cells.csv"
    data = pd.read_csv(source, low_memory=False, keep_default_na=False, na_values=[""])
    if data.assay_cell_id.duplicated().any():
        raise ValueError("Benchmark input contains repeated assay cell IDs")
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    input_manifest = output / "input_manifest.json"
    algorithm_hash = hashlib.sha256("".join(inspect.getsource(f) for f in [
        make_estimator, feature_table, descriptors, make_splits, audit_split, novelty,
    ]).encode()).hexdigest()
    previous_features = {}
    if input_manifest.exists():
        old = json.loads(input_manifest.read_text())
        if old["sha256"] != input_hash:
            raise ValueError("Benchmark input changed. Archive stale results before refitting.")
        if (old.get("algorithm_sha256") != algorithm_hash and
                algorithm_hash not in old.get("verified_compatible_algorithm_sha256", [])):
            raise ValueError("Fitting algorithm changed. Version the results before refitting.")
        previous_features = old.get("feature_sha256", {})
    features, embedding_columns, chemical_columns = feature_table(
        data, any(n.startswith("esm") for n in models), any(n.startswith("chem") for n in models))
    feature_hashes = {}
    for representation, columns in [("esm", embedding_columns), ("chem", chemical_columns)]:
        if columns:
            digest = hashlib.sha256(features[columns].to_numpy(dtype=np.float32).tobytes()).hexdigest()
            if representation in previous_features and digest != previous_features[representation]:
                raise ValueError(f"{representation} feature cache changed. Version the analysis.")
            feature_hashes[representation] = digest
    input_manifest.write_text(json.dumps({
        "path": str(source.relative_to(ROOT)), "sha256": input_hash, "n_cells": len(data),
        "algorithm_sha256": algorithm_hash, "feature_sha256": {**previous_features, **feature_hashes},
        "resumed_from_algorithm_sha256": old.get("algorithm_sha256") if input_manifest.exists() else None,
    }, indent=2))
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    audits, tasks = [], []
    for protocol in protocols:
        for seed in seeds:
            if protocol == "temporal" and seed != seeds[0]:
                continue  # The chronological partition is deterministic.
            for fold, (tr, te) in enumerate(make_splits(data, protocol, seed, CONFIG["folds"])):
                split_audit = audit_split(data, tr, te, protocol)
                split_audit.update({"protocol": protocol, "seed": seed, "fold": fold})
                audits.append(split_audit)
                split_id = f"{protocol}_{seed}_{fold}"
                pd.DataFrame({
                    "assay_cell_id": data.assay_cell_id,
                    "role": np.where(np.isin(np.arange(len(data)), tr), "train",
                                     np.where(np.isin(np.arange(len(data)), te), "test", "excluded")),
                }).to_csv(output / f"split_{split_id}.csv.gz", index=False)
                tasks.append((protocol, seed, fold, tr, te, models))
    (output / "split_audits.json").write_text(json.dumps(audits, indent=2, default=json_default))
    (output / "fitting_execution.json").write_text(json.dumps({
        "workers": workers, "threads_per_worker": 4,
        "worker_source_sha256": hashlib.sha256(inspect.getsource(fit_fold).encode()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shared_inputs": "Immutable copies; each process writes only its own split predictions.",
    }, indent=2))
    inputs = (data, features, embedding_columns, chemical_columns, prediction_dir)
    if workers == 1:
        for task in tasks:
            fit_fold(*inputs, *task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_fold_worker, initargs=inputs) as pool:
            futures = [pool.submit(fit_fold_worker, task) for task in tasks]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
    # A partial CLI run is useful for development, but cannot emit a comparison
    # table until every locked fit is available on exactly the same test cases.
    from .verification import audit_benchmark
    completion = audit_benchmark(scope, require_complete=False)
    if completion["passed"]:
        summarize_benchmark(output)


def paired_crossed_bootstrap(frame, base, alternative, repeats=2000, seed=20261020, losses=False):
    """Publication x sequence multiplier bootstrap; paired losses on identical cells."""
    difference = (frame[base]-frame[alternative] if losses else
                  abs(frame.log2_mic-frame[base]) - abs(frame.log2_mic-frame[alternative]))
    pubs, unique_pubs = pd.factorize(frame.publication_id)
    seqs, unique_seqs = pd.factorize(frame.get("resampling_sequence_id", frame.sequence_id))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        weights = rng.poisson(1, len(unique_pubs))[pubs] * rng.poisson(1, len(unique_seqs))[seqs]
        if weights.sum():
            samples.append(np.average(difference, weights=weights))
    return {
        "n_paired_cells": len(frame), "publications": len(unique_pubs), "sequences": len(unique_seqs),
        "mae_reduction_log2": float(difference.mean()),
        "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975)),
        "bootstrap_repeats": len(samples),
        "limited_independent_support": len(unique_pubs) < 10 or len(unique_seqs) < 10,
    }


def aggregate_prediction_losses(frame, keys):
    """Average held-out losses, not predictions, across repeated evaluations."""
    work = frame.copy()
    error = work.prediction-work.log2_mic
    work["absolute_error"] = abs(error)
    work["squared_error"] = error**2
    work["within_one"] = (abs(error) <= 1).astype(float)
    return work.groupby(keys, dropna=False).agg(
        log2_mic=("log2_mic", "first"), prediction=("prediction", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
        within_one_rate=("within_one", "mean"),
        n_predictions=("prediction", "size")).reset_index()


def metrics_from_losses(frame):
    y = frame.log2_mic.to_numpy()
    variance = np.var(y)
    mse = float(frame.mean_squared_error.mean())
    return {
        "n": len(frame), "r2": 1-mse/variance if variance > 0 else np.nan,
        "mae_log2": float(frame.mean_absolute_error.mean()), "rmse_log2": np.sqrt(mse),
        "within_one_dilution": float(frame.within_one_rate.mean()),
        "spearman_mean_prediction": (float(spearmanr(y, frame.prediction).statistic)
                                     if len(frame) > 2 and frame.prediction.std() > 0 else np.nan),
        "calibration_bias_log2": float((frame.prediction-frame.log2_mic).mean()),
    }


def summarize_benchmark(output):
    files = list((output / "predictions").glob("*.csv.gz"))
    if not files:
        return
    all_predictions = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""])
                                 for p in files], ignore_index=True)
    # Repeated CV predictions do not create new independent measurements.
    keys = ["protocol", "model", "assay_cell_id", "sequence_id", "publication_id",
            "target_exact", "medium", "novelty"]
    if "resampling_sequence_id" in all_predictions:
        keys.append("resampling_sequence_id")
    unique = aggregate_prediction_losses(all_predictions, keys)
    unique.to_csv(output / "cell_predictions.csv.gz", index=False)
    summaries = []
    for (protocol, model), group in unique.groupby(["protocol", "model"]):
        # For the factorial joint design each stratum has a distinct training
        # exposure target; avoid pooling duplicates across strata.
        if protocol != "joint":
            raw = all_predictions[(all_predictions.protocol == protocol) & (all_predictions.model == model)]
            pooled = aggregate_prediction_losses(raw, ["assay_cell_id"])
            summaries.append({"protocol": protocol, "model": model, "stratum": "all",
                              **metrics_from_losses(pooled)})
        for stratum, block in group.groupby("novelty"):
            summaries.append({"protocol": protocol, "model": model, "stratum": stratum,
                              **metrics_from_losses(block)})
    pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    comparisons = []
    pairs = [("sequence_ridge", "sequence_context_ridge"), ("sequence_hgb", "sequence_context_hgb"),
             ("esm_ridge", "esm_context_ridge"), ("esm_hgb", "esm_context_hgb"),
             ("chem_ridge", "chem_context_ridge"), ("chem_hgb", "chem_context_hgb"),
             ("target_mean", "context_ridge")]
    for protocol, group in unique.groupby("protocol"):
        strata = sorted(group.novelty.unique())
        if protocol != "joint":
            strata += ["all"]
        for stratum in strata:
            if stratum == "all":
                raw = all_predictions[all_predictions.protocol == protocol]
                block = aggregate_prediction_losses(raw, [k for k in keys if k != "novelty"])
            else:
                block = group[group.novelty == stratum]
            idx = ["assay_cell_id", "sequence_id", "publication_id", "log2_mic"]
            if "resampling_sequence_id" in block:
                idx.append("resampling_sequence_id")
            wide = block.pivot_table(index=idx, columns="model", values="mean_absolute_error",
                                    aggfunc="mean").reset_index()
            for base, alternative in pairs:
                if base not in wide or alternative not in wide:
                    continue
                matched = wide.dropna(subset=[base, alternative])
                if len(matched) < 3:
                    continue
                comparisons.append({
                    "protocol": protocol, "stratum": stratum, "base": base, "alternative": alternative,
                    **paired_crossed_bootstrap(matched, base, alternative, losses=True),
                })
    pd.DataFrame(comparisons).to_csv(output / "paired_context_improvements.csv", index=False)
    (output / "summary_manifest.json").write_text(json.dumps({
        "prediction_files": len(files), "raw_test_predictions": len(all_predictions),
        "averaging": "mean held-out losses per assay cell, model and novelty stratum; no error-of-mean-prediction shortcut",
        "primary_comparison": "paired MAE reduction; positive means context improves prediction",
        "uncertainty_limit": "crossed bootstrap is approximate; small numbers of clusters are flagged",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

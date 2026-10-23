"""Deterministic multimodal regressors evaluated on immutable expanded splits.

Early stopping is publication-grouped INSIDE the outer training set. The chosen
epoch count is then refitted from initialization on the complete outer training
set, with preprocessing fitted again on that training set only.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import torch
from torch import nn

from .editorial_common import (ROOT, BASE, OUT, DATA, utc, sha, read_json, write_json,
                               lock_json, read_data, frozen_split, load_prediction)
from .features import descriptors, DESCRIPTOR_NAMES
from .splits import audit_split, novelty
from .benchmark import metrics, aggregate_prediction_losses, metrics_from_losses, paired_crossed_bootstrap

DEST = OUT/"deep_contextual"
MODELS = ["multimodal_mlp", "multimodal_film_context"]
CONFIG = {
    "version": 1, "models": MODELS, "device": "cpu", "threads_per_worker": 4,
    "deterministic_algorithms": True, "validation_publication_fraction": 0.15,
    "max_epochs": 100, "patience": 12, "minimum_mae_improvement_log2": 0.0001,
    "refit_all_outer_training": True, "batch_size": 512, "learning_rate": 0.001,
    "weight_decay": 0.0001, "dropout": 0.1, "loss": "MSE of training-standardized log2 MIC",
    "esm_projection": 64, "morgan_projection": 64, "descriptor_projection": 32,
    "target_embedding": 16, "medium_embedding": 8, "hidden": 128, "hidden_layers": 2,
    "context": ["medium", "log10_cfu", "ph", "nacl_reported_concentration"],
    "context_missingness": "Separate flags for all four fields; missing medium and unknown medium have distinct IDs.",
    "salt_unit_limit": "The curated NaCl reported-concentration field is used as recorded, not interpreted as molar ionic strength.",
    "initialization": "Same split seed for paired model backbones; FiLM residual affine parameters start at zero.",
    "protocols": ["random", "sequence", "homology80", "homology50", "publication", "joint", "temporal"],
    "seeds": [20261020, 20261021, 20261022], "folds": 5, "bootstrap_repeats": 2000,
    "pretraining_limit": "ESM-2 is frozen; sequence overlap with its pretraining corpus is unknown.",
    "comparison_scope": "Reproducible architecture comparator, not a reproduction of all specialized contemporary AMP tools.",
}
ESM_END, MORGAN_END = 1280, 3328


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = True


def prepare_features(data):
    """Read original caches only; hash every vector actually used."""
    folder = DEST/"features"
    manifest_path = folder/"manifest.json"
    if manifest_path.exists():
        m = read_json(manifest_path)
        assert m["benchmark_data_sha256"] == sha(DATA)
        assert sha(folder/"matrix.npy") == m["matrix_sha256"]
        return m
    folder.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    seq_vectors, chem_vectors, desc_vectors = {}, {}, {}
    for row in data[["sequence_id", "backbone_sequence"]].drop_duplicates().itertuples(index=False):
        path = ROOT/"data/features/esm2_t33_650M_UR50D"/(row.sequence_id.replace(":", "_")+".npy")
        vector = np.load(path)
        assert vector.shape == (ESM_END,) and np.isfinite(vector).all()
        seq_vectors[row.sequence_id] = vector
        desc_vectors[row.sequence_id] = descriptors(row.backbone_sequence)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    for key in sorted(set(data.chemical_id)):
        path = ROOT/"data/features/morgan_radius2"/(key.replace(":", "_")+".npy")
        vector = np.load(path)
        assert vector.shape == (2054,) and np.isfinite(vector).all()
        chem_vectors[key] = vector
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    desc_names = [*DESCRIPTOR_NAMES, "MW", "HBD", "HBA", "TPSA", "formal_charge", "structure_missing",
                  "n_intrachain_bonds", "n_unusual_residues", "n_d_residues"]
    p = MORGAN_END+len(desc_names)
    matrix = np.lib.format.open_memmap(folder/"matrix.partial.npy", mode="w+", dtype="float32", shape=(len(data), p))
    annotations = data[["n_intrachain_bonds", "n_unusual_residues", "n_d_residues"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    for i, row in enumerate(data[["sequence_id", "chemical_id"]].itertuples(index=False)):
        chem = chem_vectors[row.chemical_id]
        matrix[i, :ESM_END] = seq_vectors[row.sequence_id]
        matrix[i, ESM_END:MORGAN_END] = chem[:2048]
        matrix[i, MORGAN_END:] = np.r_[desc_vectors[row.sequence_id], chem[2048:], annotations[i]]
    matrix.flush()
    del matrix
    (folder/"matrix.partial.npy").replace(folder/"matrix.npy")
    write_json(folder/"source_vectors_sha256.json", source_hashes)
    m = {"created_utc": utc(), "shape": [len(data), p], "dtype": "float32",
         "descriptor_names": desc_names, "benchmark_data_sha256": sha(DATA),
         "matrix_sha256": sha(folder/"matrix.npy"), "source_vectors": len(source_hashes),
         "source_vectors_manifest_sha256": sha(folder/"source_vectors_sha256.json"),
         "esm": "Frozen ESM-2 650M residue-mean vectors, without PCA", "morgan": "Frozen chiral radius-2 fingerprints averaged over source tautomers; no PCA"}
    write_json(manifest_path, m)
    return m


def inner_partition(data, train, seed):
    publications = np.array(sorted(data.iloc[train].publication_id.astype(str).unique()))
    assert len(publications) >= 3
    rng = np.random.default_rng(seed)
    held = set(rng.choice(publications, max(1, int(np.ceil(len(publications)*CONFIG["validation_publication_fraction"]))), replace=False))
    mask = data.iloc[train].publication_id.astype(str).isin(held).to_numpy()
    fit, validation = train[~mask], train[mask]
    assert len(fit) and len(validation) and not set(data.iloc[fit].publication_id) & set(data.iloc[validation].publication_id)
    return fit, validation


def category_values(series):
    return series.fillna("__MISSING__").astype(str).to_numpy()


def fit_preprocessor(raw, data, train):
    # Morgan bits retain their bounded scale; all continuous features use only training moments.
    continuous = np.r_[np.arange(ESM_END), np.arange(MORGAN_END, raw.shape[1])]
    values = np.asarray(raw[np.ix_(train, continuous)], dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(np.where(finite, values, 0).sum(axis=0), count, out=np.zeros(len(continuous)), where=count>0)
    variance = np.divide(np.where(finite, values-mean, 0).__pow__(2).sum(axis=0), count, out=np.ones(len(continuous)), where=count>0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1
    ctx = data[["log10_cfu", "ph", "nacl_reported_concentration"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    context_fit = ctx[train]
    med = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in context_fit.T])
    imp = np.where(np.isfinite(context_fit), context_fit, med)
    ctxmean, ctxscale = imp.mean(axis=0), imp.std(axis=0)
    ctxscale[ctxscale < 1e-8] = 1
    # ID zero is exclusively unknown. Missing context is an explicit training vocabulary token.
    vocab = {}
    for key in ["target_exact", "medium"]:
        items = set(category_values(data.iloc[train][key]))
        if key == "medium":
            items.add("__MISSING__")
        vocab[key] = {v: i+1 for i,v in enumerate(sorted(items))}
    y = data.iloc[train].log2_mic.to_numpy(float)
    return {"continuous": continuous.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "context_median": med.tolist(), "context_mean": ctxmean.tolist(), "context_scale": ctxscale.tolist(),
            "vocab": vocab, "y_mean": float(y.mean()), "y_scale": max(float(y.std()), 1e-8),
            "fit_rows": len(train), "fit_id_sha256": hashlib.sha256("\n".join(data.iloc[train].assay_cell_id).encode()).hexdigest()}


def transform(raw, data, indices, prep):
    x = np.array(raw[indices], dtype=np.float32, copy=True)
    continuous = prep["continuous"]
    mean, scale = np.asarray(prep["mean"], np.float32), np.asarray(prep["scale"], np.float32)
    v = x[:, continuous]
    x[:, continuous] = (np.where(np.isfinite(v), v, mean)-mean)/scale
    assert np.isfinite(x).all()
    frame = data.iloc[indices]
    cats = [np.array([prep["vocab"][key].get(v, 0) for v in category_values(frame[key])], np.int64)
            for key in ["target_exact", "medium"]]
    context = frame[["log10_cfu", "ph", "nacl_reported_concentration"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    missing = ~np.isfinite(context)
    context = (np.where(missing, prep["context_median"], context)-prep["context_mean"])/prep["context_scale"]
    ctx = np.c_[context, missing.astype(float), frame.medium.isna().to_numpy(float)].astype(np.float32)
    y = ((frame.log2_mic.to_numpy(np.float32)-prep["y_mean"])/prep["y_scale"]).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in [x, cats[0], cats[1], ctx, y])


class MultimodalRegressor(nn.Module):
    def __init__(self, descriptors_size, n_targets, n_media, context=False):
        super().__init__()
        self.esm = nn.Sequential(nn.Linear(ESM_END, 64), nn.LayerNorm(64), nn.SiLU())
        self.morgan = nn.Sequential(nn.Linear(2048, 64), nn.LayerNorm(64), nn.SiLU())
        self.descriptors = nn.Sequential(nn.Linear(descriptors_size, 32), nn.LayerNorm(32), nn.SiLU())
        self.target = nn.Embedding(n_targets+1, 16, padding_idx=0)
        self.hidden = nn.ModuleList([nn.Linear(176, 128), nn.Linear(128, 128)])
        self.norms = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.output = nn.Linear(128, 1)
        self.with_context = context
        if context:
            self.medium = nn.Embedding(n_media+1, 8, padding_idx=0)
            self.condition = nn.Sequential(nn.Linear(15, 32), nn.SiLU())
            self.affine = nn.ModuleList([nn.Linear(32, 256), nn.Linear(32, 256)])
            for layer in self.affine:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, target, medium, context):
        h = torch.cat([self.esm(x[:, :ESM_END]), self.morgan(x[:, ESM_END:MORGAN_END]),
                       self.descriptors(x[:, MORGAN_END:]), self.target(target)], dim=1)
        cond = self.condition(torch.cat([self.medium(medium), context], dim=1)) if self.with_context else None
        for i, (linear, norm) in enumerate(zip(self.hidden, self.norms)):
            h = norm(linear(h))
            if self.with_context:
                gamma, beta = self.affine[i](cond).chunk(2, dim=1)
                h = (1+gamma)*h+beta
            h = self.dropout(torch.nn.functional.silu(h))
        return self.output(h).squeeze(1)


def predict(model, tensors):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for first in range(0, len(tensors[0]), 2048):
            predictions.append(model(*(v[first:first+2048] for v in tensors[:4])).numpy())
    return np.concatenate(predictions)


def train_model(train_tensors, validation_tensors, prep, model_name, seed, fixed_epochs=None):
    seed_everything(seed, CONFIG["threads_per_worker"])
    model = MultimodalRegressor(train_tensors[0].shape[1]-MORGAN_END,
                               len(prep["vocab"]["target_exact"]), len(prep["vocab"]["medium"]),
                               model_name == "multimodal_film_context")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    order_rng = torch.Generator().manual_seed(seed)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, (fixed_epochs or CONFIG["max_epochs"])+1):
        model.train()
        order = torch.randperm(len(train_tensors[0]), generator=order_rng)
        loss_total = 0.
        for indices in order.split(CONFIG["batch_size"]):
            batch = tuple(t[indices] for t in train_tensors)
            optimizer.zero_grad(set_to_none=True)
            output = model(*batch[:4])
            loss = torch.mean((output-batch[4])**2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite neural training loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())*len(indices)
        row = {"epoch": epoch, "training_mse_standardized": loss_total/len(order)}
        if validation_tensors is not None:
            predicted = predict(model, validation_tensors)
            mae = float(np.mean(abs(predicted-validation_tensors[4].numpy()))*prep["y_scale"])
            row["validation_mae_log2"] = mae
            if mae < best-CONFIG["minimum_mae_improvement_log2"]:
                best, best_epoch, stale = mae, epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation_tensors is not None and stale >= CONFIG["patience"]:
            break
    return model, history, best_epoch if validation_tensors is not None else fixed_epochs


def task_list():
    tasks = []
    for protocol in CONFIG["protocols"]:
        for seed in CONFIG["seeds"]:
            if protocol == "temporal" and seed != CONFIG["seeds"][0]:
                continue
            for fold in range(1 if protocol == "temporal" else CONFIG["folds"]):
                tasks.append((protocol, seed, fold))
    assert len(tasks) == 91
    # Start transfer protocols early, without changing any assigned split or seed.
    return sorted(tasks, key=lambda t: (0 if t[0] == "temporal" else 1 if t[0] == "publication" else 2, t))


def fit_task(task):
    protocol, seed, fold = task
    data = read_data()
    raw = np.load(DEST/"features/matrix.npy", mmap_mode="r")
    key = f"{protocol}_{seed}_{fold}"
    split_path = BASE/"benchmark"/f"split_{key}.csv.gz"
    train, test = frozen_split(data, split_path)
    audit = audit_split(data, train, test, protocol)
    fit, validation = inner_partition(data, train, stable_seed(seed, fold, protocol, "early_stop"))
    folder = DEST/"fits"/key
    folder.mkdir(parents=True, exist_ok=True)
    lock_json(folder/"assignment.json", {
        "protocol": protocol, "seed": seed, "fold": fold, "outer_split_sha256": sha(split_path),
        "audit": audit, "inner_fit_ids": data.iloc[fit].assay_cell_id.tolist(),
        "inner_validation_ids": data.iloc[validation].assay_cell_id.tolist(),
        "inner_fit_publications": sorted(data.iloc[fit].publication_id.astype(str).unique()),
        "inner_validation_publications": sorted(data.iloc[validation].publication_id.astype(str).unique()),
    })
    code_hash = sha(Path(__file__))
    config_hash = sha(DEST/"locked_configuration.json")
    for model_name in MODELS:
        receipt = folder/f"{model_name}.json"
        prediction_path = DEST/"predictions"/f"{key}_{model_name}.csv.gz"
        if receipt.exists():
            done = read_json(receipt)
            assert done["algorithm_sha256"] == code_hash and done["configuration_sha256"] == config_hash
            assert sha(prediction_path) == done["prediction_sha256"]
            print(json.dumps({"cached": f"{key}_{model_name}"}), flush=True)
            continue
        started = time.monotonic()
        model_seed = stable_seed(seed, fold, protocol, "network")
        prep = fit_preprocessor(raw, data, fit)
        train_tensors = transform(raw, data, fit, prep)
        validation_tensors = transform(raw, data, validation, prep)
        model, history, selected = train_model(train_tensors, validation_tensors, prep, model_name, model_seed)
        del model, train_tensors, validation_tensors
        write_json(folder/f"{model_name}_inner_preprocessor.json", prep)
        write_json(folder/f"{model_name}_early_stopping.json", history)
        prep = fit_preprocessor(raw, data, train)
        train_tensors = transform(raw, data, train, prep)
        model, refit_history, _ = train_model(train_tensors, None, prep, model_name, model_seed, fixed_epochs=selected)
        del train_tensors
        test_tensors = transform(raw, data, test, prep)
        predictions = predict(model, test_tensors)*prep["y_scale"]+prep["y_mean"]
        assert len(predictions) == len(test) and np.isfinite(predictions).all()
        saved = data.iloc[test][["assay_cell_id", "sequence_id", "chemical_id", "publication_id", "cluster_80", "target_exact", "medium", "log2_mic"]].copy()
        saved["prediction"] = predictions
        saved["novelty"] = novelty(data.iloc[train], data.iloc[test])
        saved["homology80_seen"] = data.iloc[test].cluster_80.isin(set(data.iloc[train].cluster_80)).to_numpy()
        saved["resampling_sequence_id"] = data.iloc[test].sequence_leakage_group.to_numpy()
        saved["structural_identity_group_seen"] = data.iloc[test].sequence_leakage_group.isin(set(data.iloc[train].sequence_leakage_group)).to_numpy()
        saved["protocol"], saved["seed"], saved["fold"], saved["model"] = protocol, seed, fold, model_name
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temp = prediction_path.with_name(prediction_path.name+".partial")
        saved.to_csv(temp, index=False, compression="gzip")
        temp.replace(prediction_path)
        write_json(folder/f"{model_name}_preprocessor.json", prep)
        write_json(folder/f"{model_name}_refit_history.json", refit_history)
        checkpoint = folder/f"{model_name}.pt"
        torch.save(model.state_dict(), checkpoint)
        result = {"created_utc": utc(), "protocol": protocol, "seed": seed, "fold": fold, "model": model_name,
                  "selected_epochs": selected, "early_stopping_epochs_run": len(history),
                  "early_stop_hit_maximum": len(history) == CONFIG["max_epochs"],
                  "minimum_inner_validation_mae_log2": min(r["validation_mae_log2"] for r in history),
                  "parameter_count": sum(p.numel() for p in model.parameters()), "network_seed": model_seed,
                  "seconds": time.monotonic()-started, "algorithm_sha256": code_hash,
                  "configuration_sha256": config_hash, "prediction_sha256": sha(prediction_path),
                  "state_dict_sha256": sha(checkpoint), "metrics": metrics(saved.log2_mic, saved.prediction),
                  "unknown_target_test_fraction": float((test_tensors[1] == 0).float().mean()),
                  "unknown_medium_test_fraction": float((test_tensors[2] == 0).float().mean())}
        write_json(receipt, result)
        print(json.dumps({k:result[k] for k in ["protocol", "seed", "fold", "model", "selected_epochs", "seconds", "metrics"]}), flush=True)
        del test_tensors, model
    return key


def run(workers=2, limit=None):
    DEST.mkdir(parents=True, exist_ok=True)
    lock_json(DEST/"locked_configuration.json", CONFIG)
    data = read_data()
    manifest = prepare_features(data)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "torch": torch.__version__, "platform": platform.platform(), "machine": platform.machine(),
                   "device": "cpu", "workers": workers, "threads_per_worker": CONFIG["threads_per_worker"],
                   "deterministic_algorithms": True, "limitations": "Bitwise reproducibility is scoped to this software and hardware configuration.",
                   "feature_manifest_sha256": sha(DEST/"features/manifest.json")}
    lock_json(DEST/"environment.json", environment)
    tasks = task_list()[:limit] if limit else task_list()
    if workers == 1:
        for task in tasks:
            fit_task(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(fit_task, task) for task in tasks]
            for future in as_completed(futures):
                print("Completed split: "+future.result(), flush=True)


def summarize():
    """Independent split/label checks and paired loss bootstrap, never test tuning."""
    data = read_data()
    all_predictions, audits = [], []
    for protocol, seed, fold in task_list():
        key = f"{protocol}_{seed}_{fold}"
        tr, te = frozen_split(data, BASE/"benchmark"/f"split_{key}.csv.gz")
        audit = audit_split(data, tr, te, protocol)
        assignment = read_json(DEST/"fits"/key/"assignment.json")
        assert not set(assignment["inner_fit_publications"]) & set(assignment["inner_validation_publications"])
        assert set(assignment["inner_fit_ids"]) | set(assignment["inner_validation_ids"]) == set(data.iloc[tr].assay_cell_id)
        for name in MODELS:
            p = DEST/"predictions"/f"{key}_{name}.csv.gz"
            pred = load_prediction(p, data, te)
            receipt = read_json(DEST/"fits"/key/f"{name}.json")
            assert sha(p) == receipt["prediction_sha256"]
            assert np.array_equal(pred.novelty.to_numpy(), novelty(data.iloc[tr], data.iloc[te]))
            all_predictions.append(pred)
        audits.append({"protocol": protocol, "seed": seed, "fold": fold, **audit})
    raw = pd.concat(all_predictions, ignore_index=True)
    keys = ["protocol", "model", "assay_cell_id", "sequence_id", "publication_id", "target_exact", "medium", "novelty", "resampling_sequence_id"]
    unique = aggregate_prediction_losses(raw, keys)
    unique.to_csv(DEST/"cell_predictions.csv.gz", index=False)
    summaries, effects = [], []
    base = pd.read_csv(BASE/"benchmark/cell_predictions.csv.gz", low_memory=False, keep_default_na=False, na_values=[""])
    for protocol in CONFIG["protocols"]:
        group = unique[unique.protocol == protocol]
        strata = sorted(group.novelty.unique()) + ([] if protocol == "joint" else ["all"])
        for stratum in strata:
            if stratum == "all":
                block = aggregate_prediction_losses(raw[raw.protocol == protocol], [k for k in keys if k != "novelty"])
                # Base predictions have one occurrence per cell/model/stratum; nonjoint has deterministic stratum per split only for publication.
                # Reaggregate original raw files to avoid equal-weighting differently sized stratum subsets across seeds.
                classic = []
                for _, seed, fold in [t for t in task_list() if t[0] == protocol]:
                    for name in ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb", "chem_hgb", "chem_context_hgb"]:
                        classic.append(pd.read_csv(BASE/"benchmark/predictions"/f"{protocol}_{seed}_{fold}_{name}.csv.gz", keep_default_na=False, na_values=[""], low_memory=False))
                classical = aggregate_prediction_losses(pd.concat(classic, ignore_index=True), [k for k in keys if k != "novelty"])
            else:
                block = group[group.novelty == stratum]
                classical = base[(base.protocol == protocol) & (base.novelty == stratum)]
            for name, rows in block.groupby("model"):
                summaries.append({"protocol": protocol, "model": name, "stratum": stratum, **metrics_from_losses(rows)})
            both = pd.concat([block, classical], ignore_index=True)
            index = ["assay_cell_id", "sequence_id", "publication_id", "log2_mic", "resampling_sequence_id"]
            wide = both.pivot_table(index=index, columns="model", values="mean_absolute_error").reset_index()
            pairs = [(MODELS[0], MODELS[1])] + [(name, nn_name) for name in ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb", "chem_hgb", "chem_context_hgb"] for nn_name in MODELS]
            for comparator, alternative in pairs:
                matched = wide.dropna(subset=[comparator, alternative])
                assert len(matched) == len(wide)
                effects.append({"protocol": protocol, "stratum": stratum, "base": comparator, "alternative": alternative,
                                **paired_crossed_bootstrap(matched, comparator, alternative, losses=True)})
    pd.DataFrame(summaries).to_csv(DEST/"metrics.csv", index=False)
    pd.DataFrame(effects).to_csv(DEST/"paired_improvements.csv", index=False)
    write_json(DEST/"split_audits.json", audits)
    write_json(DEST/"completion.json", {"created_utc": utc(), "passed": True, "splits": len(audits), "prediction_files": len(all_predictions),
               "raw_predictions": len(raw), "unique_benchmark_cells": raw.assay_cell_id.nunique(),
               "temporal_cells_per_model": raw[raw.protocol == "temporal"].groupby("model").size().to_dict(),
               "bootstrap_repeats": CONFIG["bootstrap_repeats"], "bootstrap": "Paired publication x sequence-alias Poisson multiplier weights; losses averaged within cell and stratum before resampling.",
               "metrics_sha256": sha(DEST/"metrics.csv"), "comparisons_sha256": sha(DEST/"paired_improvements.csv")})

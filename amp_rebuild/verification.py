"""Completion and molecular-alias audits independent of model fitting."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from .curation import ROOT, card_paths, stable_hash
from .splits import audit_split, make_splits, novelty


def audit_benchmark(scope, require_complete=True):
    from .benchmark import CONFIG, metrics
    directory = ROOT / "results/reconstruction_2026-10-20" / scope / "benchmark"
    source = ROOT / "data/curated" / scope / "benchmark_cells.csv"
    data = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False)
    config = json.loads((directory / "analysis_config.json").read_text())
    assert config == CONFIG, "The saved design differs from the current locked specification."
    manifest = json.loads((directory / "input_manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    expected, missing, missing_splits, checked, splits = set(), [], [], [], []
    for protocol in config["protocols"]:
        seeds = config["seeds"][:1] if protocol == "temporal" else config["seeds"]
        for seed in seeds:
            partitions = make_splits(data, protocol, seed, config["folds"])
            assert partitions, f"No evaluable partition: {protocol}"
            for fold, (train, test) in enumerate(partitions):
                split_id = f"{protocol}_{seed}_{fold}"
                split = audit_split(data, train, test, protocol)
                split.update(protocol=protocol, seed=seed, fold=fold)
                splits.append(split)
                expected_roles = np.full(len(data), "excluded", dtype=object)
                expected_roles[train], expected_roles[test] = "train", "test"
                split_path = directory / f"split_{split_id}.csv.gz"
                if split_path.exists():
                    actual = pd.read_csv(split_path)
                    assert actual.assay_cell_id.tolist() == data.assay_cell_id.tolist()
                    assert np.array_equal(actual.role, expected_roles)
                else:
                    missing_splits.append(split_id)
                truth = data.iloc[test]
                exposure = novelty(data.iloc[train], truth)
                for model in config["models"]:
                    name = f"{split_id}_{model}"
                    expected.add(name + ".csv.gz")
                    path = directory / "predictions" / (name + ".csv.gz")
                    metric_path = path.with_name(name + ".json")
                    if not path.exists() or not metric_path.exists():
                        missing.append(name)
                        continue
                    pred = pd.read_csv(path, keep_default_na=False, na_values=[""])
                    assert pred.assay_cell_id.tolist() == truth.assay_cell_id.tolist(), name
                    assert not pred.assay_cell_id.duplicated().any(), name
                    for key in ["sequence_id", "chemical_id", "publication_id", "target_exact", "medium"]:
                        assert pred[key].tolist() == truth[key].tolist(), (name, key)
                    assert np.allclose(pred.log2_mic, truth.log2_mic, rtol=0, atol=1e-12), name
                    assert np.isfinite(pred.prediction).all(), name
                    assert np.array_equal(pred.novelty, exposure), name
                    if "sequence_leakage_group" in data:
                        assert pred.resampling_sequence_id.tolist() == truth.sequence_leakage_group.tolist(), name
                    for key, value in [("protocol", protocol), ("seed", seed), ("fold", fold), ("model", model)]:
                        assert pred[key].eq(value).all(), (name, key)
                    saved_metrics = json.loads(metric_path.read_text())
                    recomputed = metrics(pred.log2_mic, pred.prediction)
                    for key in ["mae_log2", "rmse_log2", "r2", "within_one_dilution"]:
                        assert np.isclose(saved_metrics[key], recomputed[key], rtol=1e-10, atol=1e-12), (name, key)
                    checked.append({"file": str(path.relative_to(ROOT)),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "rows": len(pred)})
    present = {p.name for p in (directory / "predictions").glob("*.csv.gz")}
    unexpected = sorted(present - expected)
    result = {
        "scope": scope, "input_sha256": manifest["sha256"],
        "expected_fits": len(expected), "verified_fits": len(checked),
        "split_count": len(splits), "missing_fits": missing, "missing_splits": missing_splits,
        "unexpected_files": unexpected,
        "identical_test_cases_and_exposure_across_models": not missing and not unexpected and not missing_splits,
        "passed": not missing and not unexpected and not missing_splits, "checked_predictions": checked,
        "checks": ["exact saved split reconstruction", "forbidden group overlaps",
                   "publication-exclusive temporal partition", "identical cases across all models",
                   "saved labels and metadata match curated source", "finite predictions",
                   "novelty derives from actual training exposure", "independent metric recomputation"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "completion_audit.json").write_text(json.dumps(result, indent=2))
    (directory / "verified_split_audits.json").write_text(json.dumps(splits, indent=2))
    if require_complete and not result["passed"]:
        raise ValueError(f"Benchmark incomplete: {len(missing)} missing fits; {len(unexpected)} unexpected files.")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_predictions"}, indent=2), flush=True)
    return result


def audit_benchmark_summaries(scope):
    """Rebuild primary summary metrics with streaming per-cell NumPy accumulators.

    This deliberately does not use the benchmark's pandas aggregation or metric
    helper. It verifies loss averaging and paired point effects, not bootstrap
    validity, model adequacy, or rank statistics sensitive to floating-point ties.
    """
    folder = ROOT/"results/reconstruction_2026-10-20"/scope/"benchmark"
    source = ROOT/"data/curated"/scope/"benchmark_cells.csv"
    data = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False)
    index = pd.Index(data.assay_cell_id)
    truth = data.log2_mic.to_numpy()
    receipt = json.loads((folder/"completion_audit.json").read_text())
    assert receipt["passed"] and receipt["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    metrics_path, pairs_path = folder/"metrics.csv", folder/"paired_context_improvements.csv"
    metrics = pd.read_csv(metrics_path)
    comparisons = pd.read_csv(pairs_path)
    files = {}
    expected_hashes = {r["file"]: r["sha256"] for r in receipt["checked_predictions"]}
    for path in sorted((folder/"predictions").glob("*.csv.gz")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[str(path.relative_to(ROOT))]
        meta = json.loads(path.with_name(path.name.removesuffix(".csv.gz")+".json").read_text())
        files.setdefault(meta["protocol"], []).append((path, meta["model"]))
    metric_checks, pair_checks, max_difference = 0, 0, 0.0
    checked_fields = ["r2", "mae_log2", "rmse_log2", "within_one_dilution", "calibration_bias_log2"]
    for protocol, paths in files.items():
        totals = {}
        for path, model in paths:
            frame = pd.read_csv(path, usecols=["assay_cell_id", "log2_mic", "prediction", "novelty"])
            blocks = list(frame.groupby("novelty"))
            if protocol != "joint":
                blocks.append(("all", frame))
            for stratum, block in blocks:
                indices = index.get_indexer(block.assay_cell_id)
                assert (indices >= 0).all() and len(np.unique(indices)) == len(indices)
                error = block.prediction.to_numpy()-block.log2_mic.to_numpy()
                values = np.column_stack([np.ones(len(block)), np.abs(error), error**2,
                                          (np.abs(error) <= 1).astype(float), block.prediction])
                accumulator = totals.setdefault((model, stratum), np.zeros((len(data), 5)))
                accumulator[indices] += values
        expected_rows = metrics[metrics.protocol.eq(protocol)]
        assert set(totals) == set(zip(expected_rows.model, expected_rows.stratum))
        for row in expected_rows.itertuples():
            accumulator = totals[(row.model, row.stratum)]
            used = accumulator[:, 0] > 0
            average = accumulator[used, 1:]/accumulator[used, :1]
            variance = np.var(truth[used])
            recomputed = {
                "r2": 1-average[:, 1].mean()/variance if variance > 0 else np.nan,
                "mae_log2": average[:, 0].mean(),
                "rmse_log2": np.sqrt(average[:, 1].mean()),
                "within_one_dilution": average[:, 2].mean(),
                "calibration_bias_log2": (average[:, 3]-truth[used]).mean(),
            }
            assert int(used.sum()) == row.n, (protocol, row.model, row.stratum, "n")
            for field, value in recomputed.items():
                saved = getattr(row, field)
                assert np.isclose(saved, value, rtol=1e-10, atol=1e-12, equal_nan=True), (protocol, row.model, row.stratum, field, saved, value)
                if np.isfinite(value):
                    max_difference = max(max_difference, abs(float(saved-value)))
            metric_checks += 1
        for row in comparisons[comparisons.protocol.eq(protocol)].itertuples():
            base, alternative = totals[(row.base, row.stratum)], totals[(row.alternative, row.stratum)]
            used = (base[:, 0] > 0) & (alternative[:, 0] > 0)
            reduction = np.mean(base[used, 1]/base[used, 0]-alternative[used, 1]/alternative[used, 0])
            assert np.isclose(reduction, row.mae_reduction_log2, rtol=1e-10, atol=1e-12)
            assert int(used.sum()) == row.n_paired_cells
            assert data.loc[used, "publication_id"].nunique() == row.publications
            sequence_key = "sequence_leakage_group" if "sequence_leakage_group" in data else "sequence_id"
            assert data.loc[used, sequence_key].nunique() == row.sequences
            max_difference = max(max_difference, abs(float(reduction-row.mae_reduction_log2)))
            pair_checks += 1
        print(f"Summary audit {scope}: {protocol} verified", flush=True)
    assert metric_checks == len(metrics) and pair_checks == len(comparisons)
    result = {
        "passed": True, "scope": scope, "input_sha256": receipt["input_sha256"],
        "prediction_files": sum(map(len, files.values())), "verified_metric_rows": metric_checks,
        "verified_paired_comparisons": pair_checks, "verified_metric_fields": checked_fields,
        "maximum_absolute_difference": max_difference,
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [metrics_path, pairs_path]},
        "method": "Streaming NumPy sums per original assay cell, separate from fitting and pandas summary aggregation.",
        "scope_limit": "Checks loss metrics, paired point effects and support counts; does not refit models, repeat the bootstrap or verify rank-of-mean-prediction statistics.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (folder/"summary_completion_audit.json").write_text(json.dumps(result, indent=2))
    return result


def audit_prediction_intervals(scope):
    """Recompute interval endpoints and coverage from saved calibration/test labels."""
    import math
    from .intervals import INTERVAL_MODELS
    folder = ROOT/"results/reconstruction_2026-10-20"/scope/"prediction_intervals"
    source = ROOT/"data/curated"/scope/"benchmark_cells.csv"
    data = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False).set_index("assay_cell_id")
    config = json.loads((folder/"config.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["input_sha256"]
    checked = []
    for protocol, folds in [("publication", range(5)), ("temporal", range(1))]:
        for fold in folds:
            stem = f"{protocol}_{fold}"
            assignment = pd.read_csv(folder/f"split_{stem}.csv.gz").set_index("assay_cell_id")
            assert set(assignment.index) == set(data.index)
            groups = {
                role: set(data.loc[assignment.index[assignment.role.eq(role)], "publication_id"])
                for role in ["fit","calibration","test"]
            }
            assert not groups["fit"] & groups["calibration"]
            assert not groups["fit"] & groups["test"]
            assert not groups["calibration"] & groups["test"]
            for model in INTERVAL_MODELS:
                name = f"{stem}_{model}"
                cal = pd.read_csv(folder/f"{name}_calibration.csv.gz").set_index("assay_cell_id")
                test = pd.read_csv(folder/f"{name}.csv.gz").set_index("assay_cell_id")
                for frame, role in [(cal,"calibration"),(test,"test")]:
                    assert not frame.index.duplicated().any()
                    assert set(frame.index) == set(assignment.index[assignment.role.eq(role)])
                    np.testing.assert_allclose(frame.log2_mic, data.loc[frame.index,"log2_mic"], atol=1e-12)
                    assert (frame.publication_id == data.loc[frame.index,"publication_id"]).all()
                    assert np.isfinite(frame.prediction).all()
                errors = (cal.log2_mic-cal.prediction).abs()
                np.testing.assert_allclose(errors, cal.absolute_error, atol=1e-12)
                rows = json.loads((folder/f"{name}.json").read_text())
                assert len(rows) == 4
                for row in rows:
                    scores = (errors.groupby(cal.publication_id).max().to_numpy()
                              if row["score_kind"] == "publication_maximum" else errors.to_numpy())
                    rank = math.ceil((len(scores)+1)*row["nominal_coverage"])
                    radius = np.sort(scores)[rank-1] if rank <= len(scores) else np.inf
                    np.testing.assert_allclose(radius,row["radius_log2"],atol=1e-10)
                    suffix = f"{row['score_kind']}_{int(100*row['nominal_coverage'])}"
                    np.testing.assert_allclose(test["lower_"+suffix],test.prediction-radius,atol=1e-10)
                    np.testing.assert_allclose(test["upper_"+suffix],test.prediction+radius,atol=1e-10)
                    covered = (test.log2_mic-test.prediction).abs() <= radius
                    values = [covered.mean(),covered.groupby(test.publication_id).mean().mean(),
                              covered.groupby(test.publication_id).all().mean()]
                    for key,value in zip(["cell_weighted_coverage","equal_publication_mean_cell_coverage",
                                          "whole_publication_coverage"], values):
                        np.testing.assert_allclose(row[key],value,atol=1e-12)
                checked.append({"fit":name,"calibration_records":len(cal),"test_records":len(test)})
    receipt = {
        "passed": len(checked) == 24, "scope": scope,
        "input_sha256": config["input_sha256"], "completed_fits": len(checked), "expected_fits": 24,
        "independent_recalculation": True, "checked_fits": checked,
        "test_outcomes_used_for_calibration": False,
        "interpretation": "Saved roles, source labels, nonoverlap, finite predictions, quantiles, interval endpoints and three coverage estimands independently checked.",
    }
    (folder/"completion_audit.json").write_text(json.dumps(receipt,indent=2))
    return receipt


def structural_alias_audit(scope):
    """Detect identical source molecular graphs with distinct canonical backbones.

    Molecular graphs are hashed for the report; equality is checked on the full
    canonical isomeric SMILES first. Tautomer equivalence is not inferred.
    """
    from rdkit import Chem
    folder = ROOT / "data/curated" / scope
    peptides = pd.read_csv(folder / "peptides.csv", keep_default_na=False)
    peptides = peptides.set_index("peptide_id", verify_integrity=True)
    # Only the frozen curation inputs can contribute to an audit of that scope.
    frozen = json.loads((folder / "curation_manifest.json").read_text())["input_files"]
    structures = {}
    valid_cards = set()
    for record in frozen:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        card = json.loads(path.read_text())
        peptide_id = card["id"]
        row = peptides.loc[peptide_id]
        for entry in card.get("smiles") or []:
            raw = entry.get("smiles") if isinstance(entry, dict) else entry
            molecule = Chem.MolFromSmiles(raw) if raw else None
            if molecule is None or any(a.GetAtomicNum() == 0 for a in molecule.GetAtoms()):
                continue
            canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
            structures.setdefault(canonical, set()).add(
                (peptide_id, row.chemical_id, row.backbone_sequence, row.sequence_id))
            valid_cards.add(peptide_id)
    aliases, sequence_edges = [], set()
    for structure, records in structures.items():
        backbones = {r[2] for r in records if r[2]}
        if len(backbones) < 2:
            continue
        structure_id = stable_hash(structure, "STRUCTURE:")
        for record in sorted(records):
            aliases.append({"structure_id": structure_id, "peptide_id": record[0],
                            "chemical_id": record[1], "backbone_sequence": record[2],
                            "sequence_id": record[3]})
        ordered = sorted(backbones)
        for other in ordered[1:]:
            sequence_edges.add((ordered[0], other))
    frame = pd.DataFrame(aliases, columns=["structure_id", "peptide_id", "chemical_id",
                                          "backbone_sequence", "sequence_id"])
    frame.to_csv(folder / "structural_aliases.csv", index=False)
    pd.DataFrame(sorted(sequence_edges), columns=["sequence_a", "sequence_b"]).to_csv(
        folder / "structural_alias_edges.csv", index=False)
    result = {
        "scope": scope, "curated_cards": len(peptides), "cards_with_valid_graph": len(valid_cards),
        "unique_molecular_graphs": len(structures),
        "graphs_with_different_backbones": int(frame.structure_id.nunique()),
        "distinct_backbone_alias_edges": len(sequence_edges),
        "limitation": "Exact isomeric molecular-graph matches only; missing structures and unrecorded modifications remain unresolved.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (folder / "structural_alias_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result

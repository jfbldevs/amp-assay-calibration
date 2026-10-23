#!/usr/bin/env python3
"""Check archived neural predictions against frozen splits, labels and metrics.

Run from the extracted code repository with data archives 01, 03 and 04.
This checks retained outputs without fitting a model or importing its estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.repository.resolve()
    base = root/'results/reconstruction_2026-10-20/expanded'
    folder = base/'editorial_phase_2026-10-22_v1/deep_contextual'
    data = pd.read_csv(root/'data/curated/expanded/benchmark_cells.csv', keep_default_na=False, low_memory=False)
    assert data.assay_cell_id.is_unique
    indexed = data.set_index('assay_cell_id')
    models = ['multimodal_mlp', 'multimodal_film_context']
    raw_count, max_difference = 0, 0.
    checks, losses, hashes = [], [], {}
    for assignment_path in sorted((folder/'fits').glob('*/assignment.json')):
        key = assignment_path.parent.name
        protocol = key.rsplit('_', 2)[0]
        split_path = base/'benchmark'/('split_'+key+'.csv.gz')
        split = pd.read_csv(split_path, keep_default_na=False)
        assert split.assay_cell_id.tolist() == data.assay_cell_id.tolist()
        train = data[split.role.eq('train')]
        test = data[split.role.eq('test')]
        train_ids, test_ids = set(train.assay_cell_id), set(test.assay_cell_id)
        assert not train_ids & test_ids
        assignment = json.loads(assignment_path.read_text())
        inner_fit, inner_val = set(assignment['inner_fit_ids']), set(assignment['inner_validation_ids'])
        assert not inner_fit & inner_val and inner_fit | inner_val == train_ids
        assert not set(indexed.loc[list(inner_fit), 'publication_id']) & set(indexed.loc[list(inner_val), 'publication_id'])
        grouped = {'sequence': 'sequence_leakage_group', 'homology80': 'cluster_80',
                   'homology50': 'cluster_50', 'publication': 'publication_id', 'temporal': 'publication_id'}
        if protocol in grouped:
            field = grouped[protocol]
            assert not set(train[field]) & set(test[field]), key
        if protocol in ['sequence', 'homology80', 'homology50']:
            assert not set(train.sequence_leakage_group) & set(test.sequence_leakage_group)
        if protocol == 'temporal':
            assert pd.to_numeric(train.publication_year).max() < pd.to_numeric(test.publication_year).min()
        for model in models:
            p = folder/'predictions'/(key+'_'+model+'.csv.gz')
            pred = pd.read_csv(p, keep_default_na=False)
            expected = json.loads((assignment_path.parent/(model+'.json')).read_text())
            digest = sha(p)
            assert digest == expected['prediction_sha256']
            assert pred.assay_cell_id.is_unique and set(pred.assay_cell_id) == test_ids
            source = indexed.loc[pred.assay_cell_id]
            assert np.allclose(source.log2_mic, pred.log2_mic, atol=1e-12, rtol=0)
            assert np.isfinite(pred.prediction).all()
            y = pred.log2_mic.to_numpy(float)
            error = pred.prediction.to_numpy(float)-y
            # Per-fit receipts used the original float32 network outputs. CSV
            # decimal strings recover these exactly when read back as float32;
            # pooled summary files instead used the parsed float64 CSV values.
            fit_error = pred.prediction.to_numpy(np.float32).astype(float)-y
            values = {'mae_log2': float(np.mean(abs(fit_error))), 'rmse_log2': float(np.sqrt(np.mean(fit_error**2))),
                      'r2': float(1-np.sum(fit_error**2)/np.sum((y-y.mean())**2)),
                      'within_one_dilution': float(np.mean(abs(fit_error)<=1))}
            for name, actual in values.items():
                difference = abs(actual-expected['metrics'][name])
                max_difference = max(max_difference, difference)
                assert difference < 1e-10, (key, model, name, difference)
            loss = pd.DataFrame({'protocol': protocol, 'model': model, 'assay_cell_id': pred.assay_cell_id,
                                 'y': y, 'absolute': abs(error), 'square': error**2, 'signed': error,
                                 'one': (abs(error)<=1).astype(float), 'novelty': pred.novelty})
            losses.append(loss)
            raw_count += len(pred)
            hashes[str(p.relative_to(root))] = digest
            checks.append({'split': key, 'model': model, 'n_test': len(pred)})
    # Average losses for repeated evaluations; averaging predictions first would
    # artificially improve errors. Include every saved novelty-specific row.
    raw = pd.concat(losses, ignore_index=True)
    metrics = pd.read_csv(folder/'metrics.csv')
    for _, row in metrics.iterrows():
        subset = raw[(raw.protocol == row.protocol) & (raw.model == row.model)]
        if row.stratum != 'all':
            subset = subset[subset.novelty == row.stratum]
        unique = subset.groupby('assay_cell_id')[['y','absolute','square','signed','one']].mean()
        assert len(unique) == row.n
        v = {'mae_log2':unique.absolute.mean(), 'rmse_log2':np.sqrt(unique.square.mean()),
             'r2':1-unique.square.mean()/unique.y.var(ddof=0), 'calibration_bias_log2':unique.signed.mean(),
             'within_one_dilution':unique.one.mean()}
        for name, actual in v.items():
            difference = abs(actual-row[name])
            max_difference = max(max_difference, float(difference))
            assert difference < 1e-10, (row.protocol, row.model, row.stratum, name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'passed': True, 'saved_splits_checked': len(checks)//2, 'prediction_files_checked': len(checks),
               'individual_predictions_checked': raw_count, 'aggregate_metric_rows_checked': len(metrics),
               'maximum_absolute_metric_difference':max_difference,
               'source_sha256':hashes, 'split_checks':checks,
               'precision':'Per-fit metrics use recovered float32 network outputs; aggregate metrics use float64 CSV parsing, matching their documented computational stages. Tolerance 1e-10.',
               'scope':'Saved split alignment, inner publication separation, protocol-specific leakage constraints, all individual labels and losses; no model refit.'}
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','split_checks']}))


if __name__ == '__main__':
    main()

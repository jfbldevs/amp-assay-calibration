"""Outcome-independent splits with explicit novelty labels and leakage checks."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def novelty(train, test):
    seq = test.sequence_id.isin(set(train.sequence_id))
    pub = test.publication_id.isin(set(train.publication_id))
    return np.where(seq, "known_sequence", "new_sequence") + np.where(pub, "__known_publication", "__new_publication")


def make_splits(data, protocol, seed=20261020, folds=5):
    indices = np.arange(len(data))
    if protocol == "random":
        return list(KFold(folds, shuffle=True, random_state=seed).split(indices))
    identity_group = "sequence_leakage_group" if "sequence_leakage_group" in data else "sequence_id"
    grouped = {"sequence": identity_group, "publication": "publication_id",
               "homology80": "cluster_80", "homology50": "cluster_50"}
    if protocol in grouped:
        groups = data[grouped[protocol]]
        k = min(folds, groups.nunique())
        if k < 2:
            return []
        return list(GroupKFold(k, shuffle=True, random_state=seed).split(indices, groups=groups))
    if protocol == "temporal":
        # One locked chronological split; publications from the same year never
        # cross the boundary. Cutoff is determined using counts, never outcomes.
        years = pd.to_numeric(data.publication_year, errors="coerce")
        valid = years.notna()
        unique = sorted(years[valid].unique())
        if len(unique) < 2:
            return []
        cutoff = min(unique[:-1], key=lambda yr: abs(float((years[valid] <= yr).mean()) - .8))
        tr = indices[valid & (years <= cutoff)]
        te = indices[valid & (years > cutoff)]
        assert not set(data.iloc[tr].publication_id) & set(data.iloc[te].publication_id)
        return [(tr, te)]
    if protocol == "joint":
        # Factorial holdouts. Test strata are assigned from ACTUAL training
        # exposure, not desired category labels. A row can be evaluated in more
        # than one fold; downstream summaries average repeated predictions
        # within cell and novelty stratum before cluster resampling.
        rng = np.random.default_rng(seed)
        publications = np.array(sorted(data.publication_id.unique()))
        clusters = np.array(sorted(data.cluster_80.unique()))
        rng.shuffle(publications)
        rng.shuffle(clusters)
        pub_folds = {g: i % folds for i, g in enumerate(publications)}
        seq_folds = {g: i % folds for i, g in enumerate(clusters)}
        result = []
        for fold in range(folds):
            held_pub = data.publication_id.map(pub_folds).eq(fold).to_numpy()
            held_seq = data.cluster_80.map(seq_folds).eq(fold).to_numpy()
            candidates = indices[~held_pub & ~held_seq]
            rng.shuffle(candidates)
            held_known = candidates[:max(1, len(candidates)//5)]
            te = np.unique(np.concatenate([indices[held_pub | held_seq], held_known]))
            tr = np.setdiff1d(indices, te)
            result.append((tr, te))
        return result
    raise ValueError(protocol)


def audit_split(data, train_indices, test_indices, protocol):
    train, test = data.iloc[train_indices], data.iloc[test_indices]
    assert not set(train_indices) & set(test_indices)
    assert not set(train.assay_cell_id) & set(test.assay_cell_id)
    fields = ["sequence_id", "chemical_id", "publication_id", "cluster_80", "cluster_50"]
    if "sequence_leakage_group" in data:
        fields.append("sequence_leakage_group")
    overlap = {
        key: float(test[key].isin(set(train[key])).mean())
        for key in fields
    }
    forbidden = {"sequence": "sequence_id", "publication": "publication_id",
                 "homology80": "cluster_80", "homology50": "cluster_50", "temporal": "publication_id"}
    if protocol in forbidden:
        assert overlap[forbidden[protocol]] == 0
    if protocol in {"sequence", "homology80", "homology50"} and "sequence_leakage_group" in data:
        assert overlap["sequence_leakage_group"] == 0
    return {
        "n_train": len(train), "n_test": len(test),
        "overlap_fraction": overlap,
        "novelty_counts": pd.Series(novelty(train, test)).value_counts().to_dict(),
        "train_year_max": pd.to_numeric(train.publication_year, errors="coerce").max(),
        "test_year_min": pd.to_numeric(test.publication_year, errors="coerce").min(),
    }

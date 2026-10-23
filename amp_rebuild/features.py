"""Outcome-independent sequence features and deterministic similarity components."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import edlib
import numpy as np
import pandas as pd

from .curation import AA, ROOT, card_paths, chemistry, stable_hash

KD = dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,
                                           3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
DESCRIPTOR_NAMES = ["length", *["fraction_" + a for a in sorted(AA)], "charge_proxy",
                    "hydrophobicity", "hydrophobic_moment_100deg", "aromatic_fraction"]


def descriptors(sequence):
    counts = Counter(sequence)
    length = len(sequence)
    hyd = np.array([KD[a] for a in sequence])
    angles = np.arange(length) * math.radians(100)
    return np.array([
        length, *[counts[a] / length for a in sorted(AA)],
        (counts["K"] + counts["R"] - counts["D"] - counts["E"]) / length,
        hyd.mean(), abs(np.sum(hyd * np.exp(1j * angles))) / length,
        sum(counts[a] for a in "FWY") / length,
    ], dtype=np.float32)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def global_edit_similarity(a, b):
    return 1 - edlib.align(a, b, mode="NW", task="distance")["editDistance"] / max(len(a), len(b))


def similarity_components(sequences, threshold=.8):
    """Connected components of all-pairs global normalized-edit similarity.

    No greedy representatives: every qualifying edge must remain within a split.
    This metric is explicitly not presented as percent identity from BLAST.
    """
    sequences = sorted(set(sequences), key=lambda s: (len(s), s))
    uf = UnionFind(len(sequences))
    edges = []
    for i, a in enumerate(sequences):
        for j in range(i + 1, len(sequences)):
            b = sequences[j]
            if len(a) / len(b) + 1e-12 < threshold:
                break
            k = int(math.floor((1 - threshold) * len(b) + 1e-9))
            distance = edlib.align(a, b, mode="NW", task="distance", k=k)["editDistance"]
            if distance >= 0:
                uf.union(i, j)
                edges.append((a, b, distance))
        if (i + 1) % 1000 == 0:
            print(f"Similarity {threshold}: {i + 1}/{len(sequences)} sequences", flush=True)
    groups = {}
    for i, seq in enumerate(sequences):
        groups.setdefault(uf.find(i), []).append(seq)
    mapping = {}
    for members in groups.values():
        key = stable_hash(sorted(members), f"EDIT{int(threshold*100)}:")
        mapping.update({seq: key for seq in members})
    return mapping, edges


def merge_alias_components(mapping, alias_edges, prefix):
    """Join existing components linked by exactly identical molecular graphs."""
    # A molecular alias can be connected through a card excluded from prediction.
    # Retain those intermediate nodes while closing the equivalence relation.
    sequences = sorted(set(mapping) | {s for edge in alias_edges for s in edge})
    index = {sequence: i for i, sequence in enumerate(sequences)}
    union = UnionFind(len(sequences))
    representatives = {}
    for sequence, component in mapping.items():
        representative = representatives.setdefault(component, sequence)
        union.union(index[sequence], index[representative])
    for a, b in alias_edges:
        union.union(index[a], index[b])
    groups = {}
    for sequence in sorted(mapping):
        groups.setdefault(union.find(index[sequence]), []).append(sequence)
    result = {}
    joins = 0
    for members in groups.values():
        originals = {mapping[sequence] for sequence in members}
        joins += len(originals)-1
        key = next(iter(originals)) if len(originals) == 1 else stable_hash(members, prefix)
        result.update({sequence: key for sequence in members})
    return result, joins


def cluster_scope(scope, thresholds=(.8, .5)):
    from .verification import structural_alias_audit
    source = ROOT / "data/curated" / scope
    data = pd.read_csv(source / "exact_assay_cells.csv", keep_default_na=False)
    sequences = sorted(set(data.backbone_sequence))
    alias_audit = structural_alias_audit(scope)
    alias_edges = list(pd.read_csv(source / "structural_alias_edges.csv").itertuples(index=False, name=None))
    identity = dict(data[["backbone_sequence", "sequence_id"]].drop_duplicates().itertuples(index=False, name=None))
    identity, identity_joins = merge_alias_components(identity, alias_edges, "IDENTITY:")
    if identity_joins:
        data["sequence_leakage_group"] = data.backbone_sequence.map(identity)
    report = {}
    for threshold in thresholds:
        mapping, edges = similarity_components(sequences, threshold)
        mapping, joins = merge_alias_components(mapping, alias_edges, f"EDIT{int(threshold*100)}_STRUCTURE:")
        column = f"cluster_{int(threshold*100)}"
        data[column] = data.backbone_sequence.map(mapping)
        counts = pd.Series(mapping).value_counts()
        pd.DataFrame(edges, columns=["sequence_a", "sequence_b", "edit_distance"]).to_csv(
            source / f"similarity_edges_{int(threshold*100)}.csv", index=False)
        report[column] = {"sequence_components": len(counts),
                          "largest_sequence_component": int(counts.max()),
                          "largest_assay_component": int(data[column].value_counts().max()),
                          "edges": len(edges), "threshold": threshold,
                          "additional_structural_alias_joins": joins}
    data.to_csv(source / "benchmark_cells.csv", index=False)
    (source / "cluster_manifest.json").write_text(json.dumps({
        "metric": "1 - global Levenshtein distance / max(length_a,length_b)",
        "method": "all-pairs edges and transitive connected components",
        "chirality": "backbones uppercased for splitting; chemical identities retained separately",
        "structural_aliases": alias_audit,
        "sequence_identity_alias_joins": identity_joins,
        "structural_identity_rule": "Exact molecular-graph aliases are indivisible across sequence and similarity holdouts; source chemical annotations are retained.",
        "clusters": report,
    }, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def embed_catalog(model_name="esm2_t33_650M_UR50D", batch_tokens=4096):
    import esm
    import torch

    output = ROOT / "data/features" / model_name
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((ROOT / "data/raw/dbaasp_peptide_index.json").read_text())
    seqs = {str(r.get("sequence") or "").upper() for r in index}
    # Add GRAMPA so the same frozen representation can be used for external checks.
    grampa = pd.read_csv(ROOT / "data/raw/grampa.csv", usecols=["sequence"])
    seqs.update(grampa.sequence.dropna().str.upper().str.strip())
    for path in (ROOT / "data/curated").glob("*/peptides.csv"):
        curated = pd.read_csv(path, usecols=["backbone_sequence"], keep_default_na=False)
        seqs.update(curated.backbone_sequence)
    sequences = sorted(s for s in seqs if s and set(s) <= AA and len(s) <= 1022)
    entries = [(stable_hash(s, "SEQ:"), s) for s in sequences]
    remaining = [(key, seq) for key, seq in entries
                 if not (output / (key.replace(":", "_") + ".npy")).exists()]
    if not remaining:
        print("All embeddings cached.", flush=True)
        return
    print(f"Embedding {len(remaining)}/{len(entries)} sequences", flush=True)
    torch.set_num_threads(4)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}.pt"
    regression_path = checkpoint.with_name(model_name + "-contact-regression.pt")
    if checkpoint.is_file():
        # These are the user's existing official ESM checkpoints. The legacy
        # file contains model configuration objects, hence explicit full loading.
        model_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        regression = (torch.load(regression_path, map_location="cpu", weights_only=False)
                      if regression_path.exists() else None)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression)
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    remaining.sort(key=lambda r: (len(r[1]), r[0]))
    batches, batch = [], []
    for entry in remaining:
        if batch and (len(entry[1]) + 2) * (len(batch) + 1) > batch_tokens:
            batches.append(batch)
            batch = []
        batch.append(entry)
    if batch:
        batches.append(batch)
    done = 0
    for batch in batches:
        _, _, tokens = batch_converter(batch)
        with torch.inference_mode():
            out = model(tokens.to(device), repr_layers=[model.num_layers], return_contacts=False)
            rep = out["representations"][model.num_layers]
            for i, (key, seq) in enumerate(batch):
                vector = rep[i, 1:len(seq)+1].mean(0).cpu().numpy().astype(np.float32)
                np.save(output / (key.replace(":", "_") + ".npy"), vector)
        done += len(batch)
        if done % 100 < len(batch) or done == len(remaining):
            print(f"ESM embeddings {done}/{len(remaining)} on {device}", flush=True)
    manifest = {
        "model": model_name, "sequences": len(entries), "layer": model.num_layers,
        "pooling": "mean over residue tokens; BOS/EOS excluded", "fine_tuning": False,
        "normalization": "uppercase canonical backbone; chemical modifications modeled separately",
        "device": device, "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
                             if checkpoint.exists() else None,
        "sequences_sha256": stable_hash(sequences),
        "pretraining_overlap": "not established; the pretraining corpus is not an independent sequence holdout",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def chemical_feature_store(include_expansion=False):
    """Mean chiral Morgan fingerprints over source tautomer representations."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    output = ROOT / "data/features/morgan_radius2"
    output.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    counts = {"computed": 0, "cached": 0, "missing": 0}
    for n, path in enumerate(card_paths(include_expansion)):
        card = json.loads(path.read_text())
        chem = chemistry(card)
        target = output / (chem["chemical_id"].replace(":", "_") + ".npy")
        if target.exists():
            previous = np.load(target)
            if previous[-1] == 0 or not card.get("smiles"):
                counts["cached"] += 1
                continue
        vectors, seen = [], set()
        for entry in card.get("smiles") or []:
            smi = entry.get("smiles") if isinstance(entry, dict) else entry
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canonical in seen:
                continue
            seen.add(canonical)
            bits = generator.GetFingerprintAsNumPy(mol).astype(np.float32)
            additional = np.array([
                Descriptors.MolWt(mol), rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcTPSA(mol),
                sum(atom.GetFormalCharge() for atom in mol.GetAtoms()), 0,
            ], dtype=np.float32)
            vectors.append(np.concatenate([bits, additional]))
        if vectors:
            vector = np.mean(vectors, axis=0).astype(np.float32)
        else:
            vector = np.zeros(2054, dtype=np.float32)
            vector[-1] = 1
            counts["missing"] += 1
        np.save(target, vector)
        counts["computed"] += 1
        if (n+1) % 1000 == 0:
            print(f"Chemical representations {n+1}: {counts}", flush=True)
    (output / "manifest.json").write_text(json.dumps({
        "representation": "2048-bit radius-2 Morgan fingerprints with chirality; mean over unique source SMILES",
        "additional_features": ["average_molecular_weight", "HBD", "HBA", "TPSA", "formal_charge", "missing_structure"],
        "no_target_information_used": True, "counts_this_run": counts,
    }, indent=2))

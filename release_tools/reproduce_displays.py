#!/usr/bin/env python3
"""Regenerate numerical claims and displays into a NEW directory from release inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; archived outputs are never overwritten.')
    import amp_rebuild.journal_materials as jm
    original = jm.GEN
    expected = json.loads((original / 'numeric_provenance.json').read_text())
    for relative, digest in expected['source_sha256'].items():
        path = ROOT / relative
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError(f'Input hash mismatch: {relative}')
    destination.mkdir(parents=True)
    jm.GEN = destination
    material = jm.Materials()
    data = jm.load_inputs(material)
    jm.numeric_claims(material, data)
    jm.further_numbers(material, data)
    jm.make_figures(material, data)
    jm.supplementary_figures(material, data)
    jm.make_tables(material, data)
    material.save()
    regenerated = json.loads((destination / 'numeric_provenance.json').read_text())
    numeric_equal = regenerated['numbers'] == expected['numbers']
    tabular = sorted(original.glob('figure_*_data.csv')) + sorted(original.glob('table_*.tex')) + [original / 'numbers.tex']
    comparisons = {p.name: p.read_bytes() == (destination / p.name).read_bytes() for p in tabular}
    receipt = {'passed': numeric_equal and all(comparisons.values()),
               'input_hashes_verified': len(expected['source_sha256']),
               'numeric_entries_identical': numeric_equal,
               'numeric_entries': len(expected['numbers']),
               'tabular_and_macro_byte_comparisons': comparisons,
               'figures_regenerated': len(material.figures), 'tables_regenerated': len(material.tables),
               'scope': 'Recreates claims and displays from retained analyses; does not refit models or repeat acquisition.',
               'pdf_identity_expected': False}
    (destination / 'reproduction_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

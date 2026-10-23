# Reproducibility scope

## Supported portable checks

- Archive integrity: every payload member has a path, byte count and SHA-256 digest, and ZIP CRCs are checked during streaming extraction.
- Both retained test modules run from the clean extracted repository. Two historical sample tables exist only for their regression cases; all manuscript results use expanded inputs.
- `release_tools/reproduce_displays.py` rebuilds numbers, tables and figures from audited summary inputs, into a new directory. It compares 14,198 numerical entries, the generated macro file, plotted data and table source. This is a reproduction of presentation/aggregation from retained analyses, not independent refitting.
- The original 19-gate workspace audit and 55-test editorial closure remain available as historical receipts with preserved hashes.

## More expensive reanalysis

Download archives 05–09 to examine the complete retained Stan chains. The corresponding `result.json` and design/prepared-data files are in archive 03. Absolute paths in preserved result/configuration files document the original machine; use `release_tools.resolve_archived_paths.resolve_archived_path(value, repository_root)` to resolve them locally without editing their bytes.

The original Stan toolchain was CmdStan 2.40.0. Install a compatible CmdStan toolchain separately for refitting; compiled local binaries are intentionally excluded. Source acquisition and renewed publication verification can change upstream content and are not expected to reproduce the frozen snapshot byte for byte. ESM-2 650M embeddings used by the analysis are supplied; the pretrained model checkpoint is an upstream dependency, not part of the code release.

Neural weights, fit configuration, preprocessing objects and training histories are distributed with archive 04. Reanalysis uses the saved classical split assignments in archive 03. Determinism and seeds describe the audited CPU/environment configuration; identical floating-point results across arbitrary hardware/software are not claimed.

## Historical scripts

Scripts 25, 35 and 37 enforce assumptions about the original complete workspace, including historical documentary inputs and installed executables. Some of those inputs are intentionally outside the public redistribution set. Their historical audit manifests are not a manifest of the clean repository. The portable utility verifies the distributed inputs directly and does not silently bypass or falsify those original global audit gates.

The submission-readiness review installed all pinned dependencies in a fresh virtual environment on the same macOS arm64 host, with system site packages disabled. Cached package wheels were reused. All 55 cases passed from a separate verified extraction. No complete model refit or renewed acquisition was performed. The current manuscript and supplement bytes are unchanged by this readiness review; dataset complement 10 supplies those current PDFs and the additional evidence. Earlier archives preserve historical documents for provenance, not as alternative submission files.

## Submission readiness checks

The fresh-environment reproduction verified 226 source hashes and exactly matched 14,198 numeric entries, 11 plotted-data CSV files, 10 table sources and the generated macro file. All 11 regenerated PNG figures were pixel-identical to the reviewed current figures.

`release_tools/check_neural_predictions.py` independently checks all 91 saved partitions, 182 prediction files and 3,015,078 individual neural predictions, including inner publication separation and protocol-specific leakage constraints. It recomputes all 48 aggregate metric rows. The per-fit receipts use float32 network values; aggregate summaries use the original float64 CSV parsing. The comparison tolerance is 1e-10. It does not train the neural model.

The review also replayed both full temporal predictions from retained weights, checked all training histories, re-executed the archived alternative anchor/interval checks with documentary writes redirected, and separately recomputed all 256 paired calibration contrasts with 2,000 publication-bootstrap replicates. Evidence is in dataset complement 10 under `editorial_readiness_20261022_v5`. These are automated checks and alternative calculations, not an independent human audit.

# AMP assay calibration: reproducibility code

Code accompanying **Reference measurements improve antimicrobial peptide prediction beyond recorded assay metadata**.
Software version: `v1.0.0`.

This repository contains the scientific source, model definitions and both retained test modules. Large data, predictions, fitted weights and Stan chains belong in the companion Zenodo **Dataset** record. A frozen copy of this repository is archived separately as **Software**. The archival identifiers below were reserved before upload; a DOI becomes publicly accessible only after its Zenodo record is published.

Sole author: **Jorge F. Beltrán**.

Universidad de La Frontera, Facultad de Ingeniería y Ciencias, Departamento de Ingeniería Química, Avenida Francisco Salazar 01145, 4811230 Temuco, Chile

Correspondence: jorge.beltran@ufrontera.cl

Use `CITATION.cff` to cite the sole author and the version-specific software archive. The DOI identifies the code, separately from the companion data.

## Quick start: reproduce retained results

The audited environment used Python 3.11.15 on macOS arm64. Exact package versions are in `requirements-reconstruction.lock.txt`. A fresh Python virtual environment with system site packages disabled was installed from this lock file on the same macOS arm64 host and passed all 55 cases. The data and code were extracted into a separate directory. Installation on other operating systems or hardware has not been certified; record any dependency changes.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reconstruction.lock.txt
python release_tools/verify_archives.py /path/to/downloaded/data/01*.zip /path/to/downloaded/data/03*.zip /path/to/downloaded/data/04*.zip --extract-to .
python -m pytest -q
python release_tools/reproduce_displays.py --output reproduced_displays
python release_tools/check_neural_predictions.py --output neural_prediction_check.json
```

The archive utility verifies all member CRCs and SHA-256 hashes and refuses to overwrite differing files. The display command checks all manuscript input hashes, then regenerates numerical macros, 11 figures and 10 tables into a **new** directory. It compares the numerical registry, plotted CSVs, table source and macro file with the retained version. PDF binary identity is not required because rendering metadata can vary. Read `REPRODUCIBILITY.md` for exactly what was and was not rerun.

## Main scientific entry points

| Scripts | Purpose |
|---|---|
| `09`–`11`, `17` | Acquisition, curation, features and reference verification |
| `12`–`15`, `19`, `23`, `26`, `29`–`31` | Classical evaluation, statistics, censored models, intervals and diagnostics |
| `32_evaluate_assay_calibration.py` | Few-shot reference calibration |
| `33_benchmark_deep_contextual.py` | Multimodal MLP and FiLM comparator |
| `34_analyze_posterior_tail_gaps.py` | Conditional tail sensitivity |
| `35`–`37` | Original workspace manuscript build and closure, retained for provenance |
| `release_tools/reproduce_displays.py` | Portable, non-overwriting display reproduction from the distributed inputs |

`amp_rebuild/` contains the analysis implementations; `models/*.stan` contains the probabilistic model definitions. The legacy prospective module is retained only because it belongs to the original tested source tree; no prospective or experimental validation was performed for this study.

Do not run original closure scripts directly on the frozen downloaded results. They were designed for the full historical workspace and can write audit outputs. Work in a fresh copy for new scientific fits, preserve saved fold assignments, and distinguish new runs from the archived version. The historical 42-test / 19-gate closure and the additive 13-test editorial closure are retained in the data archive; the release-specific verification is documented separately.

## Data and code citation

- Source repository: [jfbldevs/amp-assay-calibration](https://github.com/jfbldevs/amp-assay-calibration).
- Software archive, `v1.0.0`: [10.5281/zenodo.23582421](https://doi.org/10.5281/zenodo.23582421).
- Companion Dataset: [10.5281/zenodo.23582323](https://doi.org/10.5281/zenodo.23582323).

Cite the specific versions used, alongside the article and DBAASP: Pirtskhalava et al. (2021), [10.1093/nar/gkaa991](https://doi.org/10.1093/nar/gkaa991). GitHub is the maintained source location; the Zenodo software DOI is the archival citation.

The code is distributed under the existing MIT license in `LICENSE`. That license does not replace the database's upstream attribution terms. See the data release's `RIGHTS_AND_ATTRIBUTION.md`.

## Current manuscript revision

The companion dataset supplement `10_editorial_documents.zip` contains the current authored manuscript, Supplementary Information, bibliography audit and self-contained LaTeX sources. Earlier data archive 04 retains the original documents as historical evidence. Compile the current sources directly with Tectonic; numerical displays are unchanged.

"""Scientific invariants and independent edge cases for the reconstructed study."""
import math

import pytest

from amp_rebuild.curation import (
    canonical_publication, chemistry, parse_concentration, resolve_references, unit_factor,
)


def test_curation_subsets_preserve_mixed_unit_cells_and_graph_support(tmp_path):
    import pandas as pd
    from amp_rebuild.statistics import curation_sensitivity_membership
    cells = pd.DataFrame({
        "assay_cell_id": ["a", "b", "c", "d"],
        "chemical_id": ["CHEM:1", "CHEM:1", "CHEM:2", "CHEM:3"],
        "publication_id": ["PMID:1", "PMID:2", "PMID:2", "PMID:3"],
        "log2_mic": [1., 2., 3., 4.], "context_reporting_count": [0, 1, 2, 3],
    })
    activities = pd.DataFrame({
        "assay_cell_id": ["a", "a", "b", "c", "d"],
        "eligible_exact_analysis": [True]*5,
        "conversion_rule": ["native_molar", "mass_to_molar", "native_molar",
                            "native_molar", "native_molar"],
        "duplicate_candidate": [False, False, True, False, False],
    })
    curation_sensitivity_membership(cells, activities, tmp_path)
    flags = pd.read_csv(tmp_path/"curation_sensitivity_membership.csv").set_index("assay_cell_id")
    assert not flags.loc["a", "native_molar_only"]
    assert flags.loc[["b", "c", "d"], "native_molar_only"].all()
    assert not flags.loc["b", "no_candidate_duplicates"]
    assert flags.loc[["a", "b", "c"], "largest_connected_component"].all()
    assert not flags.loc["d", "largest_connected_component"]


def test_native_molar_values_do_not_depend_on_molecular_weight():
    assert unit_factor("µM", math.nan) == (1.0, "native_molar")
    assert unit_factor("µM", 1830.25) == (1.0, "native_molar")
    assert parse_concentration("0.011").point == .011
    assert unit_factor("µg/ml", 2000)[0] == .5
    assert math.isnan(unit_factor("µg/ml", math.nan)[0])
    assert math.isnan(unit_factor("µg/g", 2000)[0])


def test_conformal_order_statistic_keeps_small_sample_infinity():
    from amp_rebuild.intervals import conformal_quantile
    assert conformal_quantile(range(1,10), .9) == 9
    assert math.isinf(conformal_quantile([1,2,3,4], .9))
    assert conformal_quantile(range(1,20), .95) == 19


def test_interval_calibration_keeps_entire_publications_separate():
    import numpy as np
    import pandas as pd
    from amp_rebuild.intervals import calibration_partition
    data = pd.DataFrame({"publication_id": np.repeat([f"P{i}" for i in range(20)], 3)})
    fitting, calibration = calibration_partition(data, np.arange(len(data)), 17)
    assert not set(data.iloc[fitting].publication_id)&set(data.iloc[calibration].publication_id)
    assert len(calibration) == 12
    assert set(fitting)|set(calibration) == set(range(len(data)))


@pytest.mark.parametrize("raw,kind,lo,hi", [
    ("4", "exact", 4, 4), ("1e-5", "exact", 1e-5, 1e-5),
    (">128", "right", 128, math.inf), ("≤ 0.5", "left", 0, .5),
    ("4–8", "interval", 4, 8), ("1E-5-2E-5", "interval", 1e-5, 2e-5),
    ("8±1", "summary_mean", 8, 8), ("0,5", "exact", .5, .5),
])
def test_concentration_bounds(raw, kind, lo, hi):
    value = parse_concentration(raw)
    assert value.kind == kind
    assert value.lower == lo and value.upper == hi
    if kind in {"left", "right", "interval"}:
        assert math.isnan(value.point)


@pytest.mark.parametrize("raw", ["64->128", "6.3-3.1", "not active", "0", "-4", "~8", "4/8"])
def test_ambiguous_values_are_not_invented(raw):
    assert parse_concentration(raw).status != "parsed"


def test_full_titles_do_not_collide_and_doi_is_supported():
    a = {"title": "x" * 80 + " first", "year": 2020}
    b = {"title": "x" * 80 + " second", "year": 2020}
    assert canonical_publication(a) != canonical_publication(b)
    assert canonical_publication({"doi": "https://doi.org/10.1234/ABC"})[0] == "DOI:10.1234/abc"


def test_ambiguous_references_are_not_assigned_to_first_article():
    papers = [{"pubmed": {"pubmedId": "1"}, "year": 2020},
              {"pubmed": {"pubmedId": "2"}, "year": 2021}]
    assert resolve_references({"reference": "1, 2"}, papers)["publication_id"] == ""
    assert resolve_references({"reference": "8"}, papers)["publication_id"] == ""
    assert resolve_references({}, papers)["publication_id"] == ""
    assert resolve_references({}, papers[:1])["publication_id"] == "PMID:1"


def test_modified_peptides_never_get_unmodified_fallback_mass():
    base = {"id": 1, "sequence": "ACD", "complexity": {"name": "Monomer"}}
    assert chemistry(base)["mw_da"] > 0
    modified = {**base, "cTerminus": {"name": "AMD"}}
    assert math.isnan(chemistry(modified)["mw_da"])
    assert chemistry(base)["chemical_id"] != chemistry(modified)["chemical_id"]
    assert chemistry(base)["sequence_id"] == chemistry(modified)["sequence_id"]
    assert chemistry(base)["chemical_id"] != chemistry({**base, "sequence": "AcD"})["chemical_id"]


def test_structure_mass_accounts_for_chemistry():
    # Glycine free acid and glycinamide differ by approximately 0.984 Da.
    acid = chemistry({"id": 1, "sequence": "G", "smiles": [{"smiles": "NCC(=O)O"}]})
    amide = chemistry({"id": 2, "sequence": "G", "cTerminus": {"name": "AMD"},
                       "smiles": [{"smiles": "NCC(=O)N"}]})
    assert acid["mw_da"] - amide["mw_da"] == pytest.approx(.984, abs=.002)


def test_similarity_components_keep_transitive_neighbors_together():
    from amp_rebuild.features import similarity_components
    seqs = ["AAAAA", "AAAAC", "AAACC", "WWWWW"]
    mapping, _ = similarity_components(seqs, .8)
    assert mapping["AAAAA"] == mapping["AAAAC"] == mapping["AAACC"]
    assert mapping["AAAAA"] != mapping["WWWWW"]


def test_cyclic_molecular_aliases_cannot_cross_identity_or_similarity_splits():
    import pandas as pd
    from amp_rebuild.features import merge_alias_components
    from amp_rebuild.splits import make_splits
    mapping = {"VKYPKLYP": "A", "YPVKYPKL": "B", "WWWW": "C", "AAAA": "D"}
    merged, joins = merge_alias_components(mapping, [("VKYPKLYP", "YPVKYPKL")], "IDENTITY:")
    assert joins == 1
    assert merged["VKYPKLYP"] == merged["YPVKYPKL"]
    assert merged["WWWW"] == "C"
    data = pd.DataFrame({"sequence_id": list(mapping.values()),
                         "sequence_leakage_group": [merged[s] for s in mapping]})
    for train, test in make_splits(data, "sequence", folds=3):
        assert not set(data.iloc[train].sequence_leakage_group) & set(data.iloc[test].sequence_leakage_group)
    # The graph representative need not itself be eligible for prediction.
    merged, joins = merge_alias_components(
        {"B": "id_b", "C": "id_c", "D": "id_d"}, [("A", "B"), ("A", "C")], "IDENTITY:")
    assert merged["B"] == merged["C"] and merged["D"] == "id_d" and joins == 1


@pytest.mark.parametrize("kind", ["exact", "left", "right", "interval"])
def test_integrating_singleton_normal_effect_preserves_censored_likelihood(kind):
    from scipy.integrate import quad
    from scipy.stats import norm
    mu, sigma, tau, low, high = .7, 1.1, .8, -.2, 1.8
    def likelihood(mean, sd):
        if kind == "exact":
            return norm.pdf(low, mean, sd)
        if kind == "left":
            return norm.cdf(high, mean, sd)
        if kind == "right":
            return norm.sf(low, mean, sd)
        return norm.cdf(high, mean, sd)-norm.cdf(low, mean, sd)
    numerical = quad(lambda effect: likelihood(mu+effect, sigma)*norm.pdf(effect, 0, tau),
                     -math.inf, math.inf, epsabs=1e-11)[0]
    assert numerical == pytest.approx(likelihood(mu, math.sqrt(sigma**2+tau**2)), abs=1e-10)


def test_collapsed_design_retains_repeated_levels_and_integrates_only_singletons():
    import numpy as np
    import pandas as pd
    from amp_rebuild.censored import prepare_stan_data
    data = pd.DataFrame({
        "chemical_id": ["repeated"]*20+[f"unique_{i}" for i in range(10)],
        "publication_id": ["paper"]*30, "target_exact": ["target"]*30,
        "medium": ["MHB"]*20+["LBB"]*10, "supplement_category": ["Unreported"]*30,
        "log10_cfu": [np.nan]*30, "ph": [np.nan]*30,
        "nacl_reported_concentration": [np.nan]*30,
        "parse_kind": ["exact"]*30, "log2_lower": [3.0]*30, "log2_upper": [3.0]*30,
    })
    payload, metadata = prepare_stan_data(data, 1.5)
    assert payload["J_chem"] == 1 and payload["J_pub"] == 1
    assert payload["chem"] == [2]*20+[1]*10
    assert np.asarray(payload["singleton"])[:, 0].tolist() == [0]*20+[1]*10
    assert np.asarray(payload["singleton"])[:, 1].sum() == 0
    assert metadata["marginalized_singleton_levels"]["chem"] == 10
    assert payload["N_exact"] == 30 and payload["exact_index"] == list(range(1,31))
    assert len(payload["X"]) == 30 and payload["K"] > 0


def test_splits_prevent_backbone_and_publication_leakage():
    import pandas as pd
    from amp_rebuild.splits import audit_split, make_splits
    rows = []
    for p in range(12):
        for s in range(6):
            rows.append({"assay_cell_id": f"{p}_{s}", "sequence_id": f"s{s}",
                         "chemical_id": f"c{s}", "publication_id": f"p{p}",
                         "cluster_80": f"g{s//2}", "cluster_50": f"h{s//3}",
                         "publication_year": 2000 + p})
    data = pd.DataFrame(rows)
    for protocol in ["random", "sequence", "publication", "homology80", "homology50", "joint", "temporal"]:
        splits = make_splits(data, protocol)
        assert len(splits) >= 1
        for train, test in splits:
            check = audit_split(data, train, test, protocol)
            if protocol == "temporal":
                assert check["train_year_max"] < check["test_year_min"]


@pytest.mark.parametrize("scope", ["sample", "expanded"])
def test_curated_source_values_and_censoring_invariants(scope):
    import numpy as np
    import pandas as pd
    from amp_rebuild.curation import ROOT
    source = ROOT / "data/curated" / scope / "activities.csv"
    if not source.exists():
        pytest.skip(f"Run {scope} curation first")
    data = pd.read_csv(source, low_memory=False, keep_default_na=False, na_values=[""])
    assert not data.record_id.duplicated().any()
    raw_molar = data[(data.unit_raw == "µM") & (data.parse_kind == "exact")]
    expected = raw_molar.concentration_raw.map(lambda v: parse_concentration(v).point)
    assert np.allclose(raw_molar.point_uM, expected)
    eligible = data[data.eligible_censored_analysis]
    assert eligible.log2_lower.notna().all()
    assert eligible.log2_upper.notna().all()
    assert (eligible.log2_lower <= eligible.log2_upper).all()
    assert not eligible.parse_kind.eq("summary_mean").any()
    assert eligible.publication_id.notna().all()
    assert eligible.kingdom.eq("Bacteria").all()
    benchmark = data[data.eligible_benchmark]
    assert benchmark.parse_kind.eq("exact").all()
    assert benchmark.valid_sequence.all()
    censored = eligible[eligible.parse_kind != "exact"]
    assert censored.point_uM.isna().all()
    # "NA" is Nutrient Agar, a genuine medium label, not a missing-value token.
    assert data.medium.eq("NA").any()
    assert not eligible.medium.isna().any()


def test_icc_matches_balanced_anova_and_keeps_equal_measurements():
    import numpy as np
    import pandas as pd
    from amp_rebuild.statistics import icc_from_sufficient
    observations = np.array([[1., 1.], [2., 2.], [3., 3.]])
    table = pd.DataFrame({"n": [2, 2, 2], "total": observations.sum(1),
                          "squares": (observations**2).sum(1)})
    assert icc_from_sufficient(table) == pytest.approx(1)
    observations = np.array([[0., 2.], [1., 3.], [2., 4.]])
    table["total"], table["squares"] = observations.sum(1), (observations**2).sum(1)
    # Between-group and within-group mean squares are both 2.
    assert icc_from_sufficient(table) == pytest.approx(0)


def test_bootstrap_paired_effect_is_zero_for_identical_predictions():
    import pandas as pd
    from amp_rebuild.benchmark import paired_crossed_bootstrap
    frame = pd.DataFrame({"log2_mic": [1., 2., 3., 4.], "base": [2., 2., 2., 2.],
                          "alternative": [2., 2., 2., 2.], "publication_id": ["a","a","b","b"],
                          "sequence_id": ["x","y","x","y"]})
    result = paired_crossed_bootstrap(frame, "base", "alternative", repeats=100)
    assert result["mae_reduction_log2"] == 0
    assert result["ci_low"] == result["ci_high"] == 0


def test_repeated_cv_averages_losses_not_predictions():
    import pandas as pd
    from amp_rebuild.benchmark import aggregate_prediction_losses, metrics_from_losses
    frame = pd.DataFrame({"assay_cell_id": ["a", "a", "b", "b"], "log2_mic": [1,1,3,3],
                          "prediction": [0,2,2,4]})
    cells = aggregate_prediction_losses(frame, ["assay_cell_id"])
    result = metrics_from_losses(cells)
    assert result["mae_log2"] == 1
    assert result["rmse_log2"] == 1
    assert result["r2"] == 0


def test_screening_regret_is_zero_only_for_correct_selection():
    import pandas as pd
    from amp_rebuild.evaluation import screening_metrics
    data = pd.DataFrame({"assay_cell_id": list("abcde"), "log2_mic": [1,2,3,4,5],
                         "prediction": [1,2,3,4,5]})
    good = screening_metrics(data)
    assert good["regret_to_observed_oracle_log2"] == 0
    assert good["gain_over_random_log2"] == 2
    data["prediction"] = [5,4,3,2,1]
    bad = screening_metrics(data)
    assert bad["regret_to_observed_oracle_log2"] == 4
    assert bad["gain_over_random_log2"] == -2


def test_energy_diagnostic_uses_within_chain_transitions():
    import numpy as np
    from amp_rebuild.posterior_checks import bfmi_by_chain
    energy = np.array([[0., 100.], [1., 102.], [0., 100.], [1., 102.]])
    # Variance is 1/3 (or 4/3); mean squared transitions are 1 (or 4).
    # The arbitrary difference between chain means must not enter E-BFMI.
    assert bfmi_by_chain(energy) == pytest.approx([3., 3.])
    with pytest.raises(ValueError):
        bfmi_by_chain(np.ones((4, 2)))


def test_rank_threshold_does_not_turn_roundoff_ties_into_inversions():
    import numpy as np
    from amp_rebuild.statistics import rank_inversion_flags
    a = np.array([1., np.nextafter(1., 2.), 1.+2e-15, 1.002904, 2., 2.1])
    b = np.array([-3., -3., -3., -3., -3., -3.])
    assert rank_inversion_flags(a, b, 1.).tolist() == [False, False, False, True, True, True]
    assert rank_inversion_flags(a, b, 2.).tolist() == [False, False, False, False, False, True]
    assert not rank_inversion_flags(1e-15, -2.)


def test_source_medium_amendment_preserves_measurement_and_guards_identity():
    from amp_rebuild.curation import amend_source_row
    row = {"record_id":"DBAASP:21860:172770","medium":"MHB","concentration_raw":"8"}
    amendment = {"record_id":row["record_id"],"field":"medium","original":"MHB",
                 "corrected":"RPMI-1640","expected":row.copy()}
    corrected = amend_source_row(row,amendment)
    assert row["medium"] == "MHB"
    assert corrected == {**row,"medium":"RPMI-1640"}
    with pytest.raises(ValueError):
        amend_source_row({**row,"concentration_raw":"32"},amendment)


@pytest.mark.parametrize("scope", ["sample", "expanded"])
def test_actual_temporal_split_has_no_shared_paper(scope):
    import pandas as pd
    from amp_rebuild.curation import ROOT
    from amp_rebuild.splits import make_splits
    path = ROOT / "data/curated" / scope / "benchmark_cells.csv"
    if not path.exists():
        pytest.skip(f"Prepare {scope} benchmark cells first")
    data = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    train, test = make_splits(data, "temporal")[0]
    assert data.iloc[train].publication_year.max() < data.iloc[test].publication_year.min()
    assert not set(data.iloc[train].publication_id) & set(data.iloc[test].publication_id)

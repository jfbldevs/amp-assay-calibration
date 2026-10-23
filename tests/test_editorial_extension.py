"""Scientific safeguards for the additive editorial analysis."""
import numpy as np
import pandas as pd
import torch

from amp_rebuild.deep_contextual import (inner_partition, fit_preprocessor, transform,
    train_model, MORGAN_END, MultimodalRegressor, seed_everything)


def toy_data(n=30):
    rng = np.random.default_rng(55)
    raw = rng.normal(size=(n, MORGAN_END+4)).astype(np.float32)
    frame = pd.DataFrame({"assay_cell_id": [f"cell_{i}" for i in range(n)],
        "publication_id": [f"pub_{i//3}" for i in range(n)],
        "target_exact": [f"target_{i%2}" for i in range(n)],
        "medium": ["MHB"]*n, "log10_cfu": [5.]*n, "ph": [np.nan]*n,
        "nacl_reported_concentration": [np.nan]*n, "log2_mic": rng.normal(size=n)})
    return raw, frame


def test_inner_validation_is_disjoint_and_inside_outer_training():
    _, data = toy_data()
    outer = np.arange(24)
    fit, val = inner_partition(data, outer, 9)
    assert set(fit) | set(val) == set(outer)
    assert not set(data.iloc[fit].publication_id) & set(data.iloc[val].publication_id)
    assert not (set(fit) | set(val)) & set(range(24, 30))


def test_test_labels_and_extremes_cannot_change_preprocessing():
    raw, data = toy_data()
    train = np.arange(20)
    before = fit_preprocessor(raw, data, train)
    raw[20:] = 1e9
    data.loc[20:, ["ph", "log10_cfu", "log2_mic"]] = 1e8
    data.loc[20:, "target_exact"] = "unseen test pathogen"
    after = fit_preprocessor(raw, data, train)
    assert before == after
    tensor = transform(raw, data, np.arange(20, 30), before)
    assert torch.all(tensor[1] == 0)
    assert torch.isfinite(tensor[0]).all() and torch.isfinite(tensor[3]).all()


def test_missing_context_has_explicit_flags_and_finite_values():
    raw, data = toy_data()
    data.loc[4, "medium"] = np.nan
    prep = fit_preprocessor(raw, data, np.arange(20))
    tensors = transform(raw, data, np.arange(30), prep)
    assert torch.all(tensors[3][:, 4:6] == 1)
    assert tensors[3][4, 6] == 1
    assert tensors[2][4] != 0


def test_cpu_training_is_bitwise_repeatable():
    raw, data = toy_data()
    prep = fit_preprocessor(raw, data, np.arange(20))
    tensors = transform(raw, data, np.arange(20), prep)
    first, history1, _ = train_model(tensors, None, prep, "multimodal_film_context", 77, fixed_epochs=2)
    second, history2, _ = train_model(tensors, None, prep, "multimodal_film_context", 77, fixed_epochs=2)
    assert history1 == history2
    assert all(torch.equal(first.state_dict()[k], second.state_dict()[k]) for k in first.state_dict())


def test_film_starts_as_identity_on_paired_backbone():
    seed_everything(3)
    baseline = MultimodalRegressor(4, 2, 2, False).eval()
    seed_everything(3)
    film = MultimodalRegressor(4, 2, 2, True).eval()
    x = torch.randn(6, MORGAN_END+4)
    category = torch.ones(6, dtype=torch.long)
    context = torch.randn(6, 7)
    assert torch.equal(baseline(x, category, category, context), film(x, category, category, context))


from amp_rebuild.assay_calibration import (variance_components, shrinkage, query_mask,
    calibration_halves, build_plan, score_radii, evaluate_queries, weights_for)


def test_random_intercept_moments_recover_known_balanced_anova():
    # Group means 0 and 4; within-group squared error totals 4, df=2.
    result = variance_components([-1., 1., 3., 5.], pd.Series(["a", "a", "b", "b"]))
    assert result["sigma_e2"] == 2.
    assert result["sigma_b2"] == 7.
    assert shrinkage(2, 7., 2.) == 14/16
    assert shrinkage(5, 0., 2.) == 0


def test_anchor_entity_is_excluded_in_every_other_context():
    frame = pd.DataFrame({"chemical_id": ["a", "a", "b", "c"]})
    block = {"rows": np.arange(4), "anchors": np.array([[0, 2], [2, 3]])}
    q = query_mask(frame, block, 1)
    assert q.tolist() == [[False, False, True, True], [True, True, False, True]]


def test_variance_and_conformal_halves_share_no_publications():
    _, frame = toy_data()
    a, b = calibration_halves(frame, np.arange(30), "test")
    assert set(a) | set(b) == set(range(30))
    assert not set(frame.iloc[a].publication_id) & set(frame.iloc[b].publication_id)


def test_anchor_selection_cannot_see_test_outcomes():
    _, frame = toy_data(30)
    frame["publication_id"] = "one study"
    frame["chemical_id"] = [f"chem{i//2}" for i in range(30)]
    counts = {"chem0": 15, "chem4": 20}
    plan, _ = build_plan(frame, np.arange(30), "publication", "standard", counts, "test", "test")
    assert frame.iloc[plan[0]["anchors"][0, :2]].chemical_id.tolist() == ["chem4", "chem0"]
    frame["log2_mic"] = np.arange(30)*1e9
    changed, _ = build_plan(frame, np.arange(30), "publication", "standard", counts, "test", "test")
    assert np.array_equal(plan[0]["anchors"], changed[0]["anchors"])


def test_conformal_unavailable_rank_is_infinite():
    scores = np.arange(5.).reshape(1, 5, 1, 1)
    assert np.isinf(score_radii(scores, .95)).all()
    scores = np.arange(20.).reshape(1, 20, 1, 1)
    assert score_radii(scores, .95).item() == 19


def test_offsets_use_anchor_labels_only_and_correct_sign():
    frame = pd.DataFrame({"chemical_id": list("abcdef"), "log2_mic": [2.]*6})
    block = {"rows": np.arange(6), "anchors": np.array([[0, 1, 2, 3, 4]]), "publication_id": "p"}
    residual = np.ones((6, 4))*2.
    weights = np.array([[0., 1., .5]]*4)
    _, stats, _, _ = evaluate_queries(frame, [block], residual, 1, weights)
    assert np.all(stats[0, :, 0, 3] == 10.)
    assert np.all(stats[0, :, 1, 3] == 0.)
    assert np.all(stats[0, :, 2, 3] == 5.)
    # Altering a query label cannot alter the estimated anchor offset.
    altered = residual.copy(); altered[1:, :] += 1
    _, other, _, _ = evaluate_queries(frame, [block], altered, 1, weights)
    assert np.all(other[0, :, 1, 5] == -5.)


from amp_rebuild.posterior_tail_gaps import truncated_gaussian_from_uniform, source_window_scenarios
from scipy.special import ndtr, ndtri


def test_truncated_normal_matches_analytic_conditional_quantile():
    mu, sd, upper = np.array([0.]), np.array([1.]), np.array([1.])
    actual, mass, clips = truncated_gaussian_from_uniform(mu, sd, np.array([.95]), upper)
    assert np.allclose(actual, ndtri(.95*ndtr(1.)))
    assert mass.item() == ndtr(1.) and clips == 0
    free, mass, clips = truncated_gaussian_from_uniform(mu, sd, np.array([.95]), np.array([np.inf]))
    assert np.allclose(free, ndtri(.95)) and mass.item() == 1.


def test_window_constraints_respect_original_mass_units_and_contradictions():
    frame = pd.DataFrame({"record_id": list("abcde"), "publication_id": ["p"]*3+["q"]*2,
        "target_exact": ["t"]*5, "unit_raw": ["ug/ml"]*5,
        "parse_kind": ["right", "exact", "exact", "right", "exact"],
        "conversion_factor": [1., .5, .25, 1., 1.],
        "log2_lower": [6., 4., 3., 5., 6.]})
    limits, ledger = source_window_scenarios(frame)
    # The shared 64 ug/ml bound is 32 or 16 uM for the exact compounds.
    assert limits["unique_right_limit_compatible"][1:3].tolist() == [5., 4.]
    assert np.isinf(limits["unique_right_limit_compatible"][4])
    assert ledger.loc[ledger.publication_id.eq("q"), "exact_records_above_maximum_right_limit"].item() == 1

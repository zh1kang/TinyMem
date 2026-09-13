"""Behavioral oracles for the fixed four-bit content decoder."""
import numpy as np

from tinymem.research.independent_fact_content_probe import fit_ridge, predict_ridge


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(91, 16))
    x[:, 5] = 7
    y = (rng.normal(size=(91, 4)) > 0).astype(float)
    alpha = .03
    model = fit_ridge(x, y, alpha=alpha)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    penalty = np.column_stack((np.eye(16), np.zeros(16))) * np.sqrt(alpha)
    a = np.vstack((design / np.sqrt(4 * len(x)), penalty))
    b = np.vstack((y / np.sqrt(4 * len(x)), np.zeros((16, 4))))
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(predict_ridge(model, x), design @ coefficients, atol=1e-11)
    shifted = x + 15
    before = {key: value.copy() for key, value in model.items() if isinstance(value, np.ndarray)}
    np.testing.assert_allclose(predict_ridge(model, shifted),
        np.column_stack(((shifted - mean) / scale, np.ones(len(x)))) @ coefficients, atol=1e-11)
    for key, value in before.items(): np.testing.assert_array_equal(model[key], value)


def test_symbol_control_and_constant_feature_baseline():
    from tinymem.research.independent_fact_content_probe import select_ridge
    y = ((np.arange(64)[:, None] % 16) >> np.arange(4)) & 1
    x = np.zeros((64, 16)); x[:, :4] = y
    folds = np.repeat(np.arange(4), 16)
    model, scores = select_ridge(x, y, folds, alphas=[1e-6, .001, 1])
    np.testing.assert_array_equal(predict_ridge(model, x) >= .5, y)
    assert all(row['n'] == 64 for score in scores for row in score['folds'])
    constant = fit_ridge(np.ones_like(x), y, alpha=1)
    np.testing.assert_array_equal(predict_ridge(constant, x), np.full((64, 4), .5))


def test_regularization_ties_choose_larger_alpha():
    from tinymem.research.independent_fact_content_probe import select_ridge
    x = np.zeros((16, 16)); y = np.zeros((16, 4))
    model, _ = select_ridge(x, y, np.repeat(np.arange(4), 4), alphas=[.001, 1, 10])
    assert model['alpha'] == 10


def test_shuffle_keeps_whole_label_trajectories_inside_families():
    from tinymem.research.independent_fact_content_probe import permute_family_labels
    y = ((np.arange(64).reshape(16, 4, 1)) >> np.arange(4)) & 1
    families = np.repeat(np.arange(4), 4)
    shuffled, donors = permute_family_labels(y, families, seed=42)
    np.testing.assert_array_equal(shuffled, y[donors])
    np.testing.assert_array_equal(families[donors], families)
    assert sorted(donors.tolist()) == list(range(16)) and np.any(donors != np.arange(16))
    shuffled[:] = 0
    assert y.any()


def test_invalid_fit_data_fails_at_boundary():
    import pytest
    x = np.zeros((8, 16)); y = np.zeros((8, 4))
    for alpha in (0, -1, float('nan'), True):
        with pytest.raises(ValueError): fit_ridge(x, y, alpha=alpha)
    with pytest.raises(ValueError): fit_ridge(x + np.nan, y, alpha=1)
    with pytest.raises(ValueError): fit_ridge(x, y + 2, alpha=1)


def test_endpoint_groups_keep_worlds_and_paired_damage_separate():
    from scripts.evaluate_independent_fact_content_probe import endpoint, views
    streams = [{'id': f's{i}'} for i in range(64)]
    names = [f's{i}/{branch}/08' for i in range(64)
             for branch in ('prefix', 'balanced', 'single0', 'single1', 'single2', 'single3')]
    data = {'manifest': {'streams': streams}, 'names': names}
    selected = views(data)
    assert {key: len(value[0]) for key, value in selected.items()} == {
        'balanced': 256, 'spoken': 256, 'unspoken': 768, 'single_mean': 1024}
    for condition, (before, after, q, owners) in selected.items():
        for src, dst, fact, owner in zip(before, after, q, owners, strict=True):
            assert names[src] == f's{owner}/prefix/08'
            assert names[dst].startswith(f's{owner}/')
            if condition == 'spoken': assert names[dst] == f's{owner}/single{fact}/08'
            if condition == 'unspoken': assert names[dst] != f's{owner}/single{fact}/08'
    target = np.zeros((len(names), 4), dtype=int)
    scores = np.zeros_like(target, dtype=float)
    # One old wrong answer is repaired; two different old correct facts are damaged.
    scores[names.index('s0/prefix/08'), 0] = 1
    scores[names.index('s0/balanced/08'), 1:3] = 1
    lm_ok = np.ones_like(target, dtype=bool)
    lm_ok[names.index('s0/balanced/08'), 3] = False
    draws = np.tile(np.arange(64), (10, 1))
    result = endpoint(scores, target, lm_ok, selected['balanced'], draws=draws, whole_world=True)
    assert (result['before_correct'], result['after_correct']) == (255, 254)
    assert (result['repairs'], result['damage'], result['gain']) == (1, 2, -1)
    assert result['complete_worlds'] == 63 and result['worlds'] == 64
    assert result['lm_damage_n'] == result['lm_damage_probe_correct_at_both_ends'] == 1
    assert result['gain_interval95'] == [-1/256, -1/256]
    assert result['both_correct'] + result['lm_wrong_probe_correct'] == 254

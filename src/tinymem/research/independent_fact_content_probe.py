"""CPU linear recovery of four fact bits from a fixed sixteen-value state."""
from collections.abc import Mapping

import numpy as np


def _features(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 16 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('features must be a nonempty finite N by 16 matrix')
    return x


def fit_ridge(features, targets, *, alpha):
    """Minimize squared error / (4N) + alpha * squared weights; do not penalize bias."""
    x = _features(features)
    y = np.asarray(targets, dtype=np.float64)
    if y.shape != (len(x), 4) or not np.isin(y, (0, 1)).all():
        raise ValueError('targets must be an N by 4 binary matrix')
    if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale == 0, 1, scale)
    z = (x - mean) / scale
    bias = y.mean(0)
    gram = z.T @ z / (4 * len(x)) + alpha * np.eye(16)
    rhs = z.T @ (y - bias) / (4 * len(x))
    weights = np.linalg.solve(gram, rhs)
    if not np.isfinite(weights).all():
        raise ValueError('nonfinite fitted coefficients')
    return {'mean': mean, 'scale': scale, 'weights': weights, 'bias': bias,
            'alpha': float(alpha)}


def predict_ridge(model: Mapping, features):
    x = _features(features)
    return ((x - model['mean']) / model['scale']) @ model['weights'] + model['bias']


def select_ridge(features, targets, folds, *, alphas):
    """Select only on the four old fact-order folds, then refit all supplied rows."""
    x = _features(features)
    y = np.asarray(targets)
    groups = np.asarray(folds)
    if groups.shape != (len(x),) or set(groups.tolist()) != set(range(4)):
        raise ValueError('four nonempty fact-order folds are required')
    candidates = sorted(set(alphas))
    if not candidates:
        raise ValueError('at least one alpha is required')
    scores = []
    for alpha in candidates:
        fold_scores = []
        for fold in range(4):
            train = groups != fold
            model = fit_ridge(x[train], y[train], alpha=alpha)
            predicted = predict_ridge(model, x[~train])
            fold_scores.append({'fold': fold, 'mse': float(np.square(predicted - y[~train]).mean()),
                                'correct': int(((predicted >= .5) == y[~train]).sum()),
                                'n': int(y[~train].size)})
        scores.append({'alpha': float(alpha), 'folds': fold_scores,
                       'mean_mse': float(np.mean([s['mse'] for s in fold_scores]))})
    minimum = min(row['mean_mse'] for row in scores)
    chosen = max(row['alpha'] for row in scores
                 if np.isclose(row['mean_mse'], minimum, rtol=1e-12, atol=1e-15))
    return fit_ridge(x, y, alpha=chosen), scores


def permute_family_labels(labels, families, *, seed):
    """Move complete four-step/four-bit labels within each order/action family."""
    y = np.asarray(labels)
    family = np.asarray(families)
    if y.ndim != 3 or y.shape[1:] != (4, 4) or not np.isin(y, (0, 1)).all():
        raise ValueError('labels must have shape trajectories by four steps by four bits')
    if family.shape != (len(y),):
        raise ValueError('one family is required per trajectory')
    rng = np.random.default_rng(seed)
    donor = np.arange(len(y))
    for identity in sorted(set(family.tolist())):
        indices = np.flatnonzero(family == identity)
        donor[indices] = rng.permutation(indices)
    return y[donor].copy(), donor

"""Conformal prediction: sets and intervals with a coverage guarantee.

If calibration and test data are exchangeable, each method below covers the truth at least
1 - alpha of the time on average, whatever the model. The guarantee is marginal, and it
fails under distribution shift, so `coverage_report` checks subgroups too.
"""
import numpy as np


def conformal_quantile(scores, alpha):
    """The (n + 1)(1 - alpha) / n empirical quantile, rounded up. Returns inf when n is too small."""
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(scores[k - 1])


class LAC:
    """Least ambiguous sets: smallest average set size. Score is 1 minus the true class probability."""

    def fit(self, probs, labels, alpha=0.1):
        probs, labels = np.asarray(probs, float), np.asarray(labels, int)
        self.q = conformal_quantile(1 - probs[np.arange(len(labels)), labels], alpha)
        return self

    def predict(self, probs):
        return np.asarray(probs, float) >= 1 - self.q - 1e-12


class APS:
    """Adaptive prediction sets: add classes from most to least likely until the mass reaches a limit.
    Sets grow when the model is unsure, which gives better coverage across hard and easy cases."""

    def __init__(self, randomised=True, seed=0):
        self.randomised = randomised
        self.rng = np.random.default_rng(seed)

    def _scores(self, probs, labels):
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        rank = np.argmax(order == labels[:, None], axis=1)
        n = np.arange(len(labels))
        s = cum[n, rank]
        if self.randomised:
            s = s - sorted_p[n, rank] * (1 - self.rng.random(len(labels)))
        return s

    def fit(self, probs, labels, alpha=0.1):
        probs, labels = np.asarray(probs, float), np.asarray(labels, int)
        self.q = conformal_quantile(self._scores(probs, labels), alpha)
        return self

    def predict(self, probs):
        probs = np.asarray(probs, float)
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        before = cum - sorted_p
        u = self.rng.random(len(probs))[:, None] if self.randomised else 1.0
        keep = before + sorted_p * u <= self.q + 1e-12
        out = np.zeros_like(probs, bool)
        np.put_along_axis(out, order, keep, axis=1)
        return out


class Mondrian:
    """Class-conditional coverage: one threshold for each class, using LAC scores, so rare classes
    are protected too. It needs enough calibration examples of each class."""

    def fit(self, probs, labels, alpha=0.1):
        probs, labels = np.asarray(probs, float), np.asarray(labels, int)
        self.q = np.array([conformal_quantile(1 - probs[labels == c, c], alpha) for c in range(probs.shape[1])])
        return self

    def predict(self, probs):
        return np.asarray(probs, float) >= 1 - self.q - 1e-12


class IntervalRegressor:
    """Split conformal intervals around any point prediction: y_hat ± q."""

    def fit(self, predictions, targets, alpha=0.1):
        self.q = conformal_quantile(np.abs(np.asarray(targets, float) - np.asarray(predictions, float)), alpha)
        return self

    def predict(self, predictions):
        p = np.asarray(predictions, float)
        return p - self.q, p + self.q


class NormalisedRegressor:
    """Intervals scaled by a difficulty estimate, so they widen where the model is less sure."""

    def fit(self, predictions, spread, targets, alpha=0.1):
        s = np.maximum(np.asarray(spread, float), 1e-9)
        self.q = conformal_quantile(np.abs(np.asarray(targets, float) - np.asarray(predictions, float)) / s, alpha)
        return self

    def predict(self, predictions, spread):
        p, s = np.asarray(predictions, float), np.maximum(np.asarray(spread, float), 1e-9)
        return p - self.q * s, p + self.q * s


class CQR:
    """Conformalised quantile regression: corrects lower and upper quantile predictions."""

    def fit(self, lower, upper, targets, alpha=0.1):
        lo, hi, y = (np.asarray(v, float) for v in (lower, upper, targets))
        self.q = conformal_quantile(np.maximum(lo - y, y - hi), alpha)
        return self

    def predict(self, lower, upper):
        return np.asarray(lower, float) - self.q, np.asarray(upper, float) + self.q


def coverage(sets, labels):
    sets, labels = np.asarray(sets, bool), np.asarray(labels, int)
    return float(sets[np.arange(len(labels)), labels].mean())


def interval_coverage(lower, upper, targets):
    y = np.asarray(targets, float)
    return float(((y >= lower) & (y <= upper)).mean())


def coverage_report(sets, labels, groups=None):
    """Coverage and set size overall, for each class, and for each group, with empty-set rate."""
    sets, labels = np.asarray(sets, bool), np.asarray(labels, int)
    n = np.arange(len(labels))
    hit = sets[n, labels]
    size = sets.sum(axis=1)
    out = {"coverage": float(hit.mean()), "mean_size": float(size.mean()), "empty": float((size == 0).mean()), "singleton": float((size == 1).mean()),
           "by_class": {int(c): float(hit[labels == c].mean()) for c in np.unique(labels)}}
    if groups is not None:
        groups = np.asarray(groups)
        out["by_group"] = {str(g): {"coverage": float(hit[groups == g].mean()), "mean_size": float(size[groups == g].mean()), "n": int((groups == g).sum())} for g in np.unique(groups)}
    return out

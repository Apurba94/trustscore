"""Is a model's confidence honest? Calibration metrics and temperature scaling."""
import numpy as np


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _prepare(probs, labels):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    if probs.ndim != 2 or len(probs) != len(labels):
        raise ValueError("probs must be N×C and labels length N")
    return probs, labels


def _bins(conf, n_bins, strategy):
    if strategy == "width":
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    elif strategy == "mass":
        order = np.argsort(conf, kind="stable")
        idx = np.empty(len(conf), int)
        for b, part in enumerate(np.array_split(order, n_bins)):
            idx[part] = b
    else:
        raise ValueError("strategy must be width or mass")
    return idx


def reliability(probs, labels, n_bins=15, strategy="width"):
    """One row per non-empty bin: count, mean confidence and accuracy."""
    probs, labels = _prepare(probs, labels)
    conf = probs.max(axis=1)
    right = probs.argmax(axis=1) == labels
    idx = _bins(conf, n_bins, strategy)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": b, "count": int(mask.sum()), "confidence": float(conf[mask].mean()), "accuracy": float(right[mask].mean())})
    return rows


def ece(probs, labels, n_bins=15, strategy="width"):
    """Expected calibration error: the weighted mean gap between confidence and accuracy."""
    rows = reliability(probs, labels, n_bins, strategy)
    n = sum(r["count"] for r in rows)
    return sum(r["count"] / n * abs(r["confidence"] - r["accuracy"]) for r in rows)


def mce(probs, labels, n_bins=15, strategy="width"):
    """Maximum calibration error: the worst bin's gap."""
    return max(abs(r["confidence"] - r["accuracy"]) for r in reliability(probs, labels, n_bins, strategy))


def classwise_ece(probs, labels, n_bins=15):
    """Calibration of each class probability, not only the top one."""
    probs, labels = _prepare(probs, labels)
    total = 0.0
    for c in range(probs.shape[1]):
        p = probs[:, c]
        hit = labels == c
        idx = _bins(p, n_bins, "width")
        for b in range(n_bins):
            m = idx == b
            if m.any():
                total += m.mean() * abs(p[m].mean() - hit[m].mean())
    return total / probs.shape[1]


def brier(probs, labels):
    """Multi-class Brier score: mean squared error against one-hot labels."""
    probs, labels = _prepare(probs, labels)
    one_hot = np.eye(probs.shape[1])[labels]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def nll(probs, labels):
    probs, labels = _prepare(probs, labels)
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)
    return float(-np.log(p).mean())


def accuracy(probs, labels):
    probs, labels = _prepare(probs, labels)
    return float((probs.argmax(axis=1) == labels).mean())


def fit_temperature(logits, labels, low=0.05, high=20.0, tol=1e-5):
    """Find the temperature that minimises NLL. NLL is convex in 1/T, so golden-section search works."""
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, int)
    if np.isnan(logits).any():
        raise ValueError("logits contain NaN")

    def loss(beta):
        z = logits * beta
        z = z - z.max(axis=1, keepdims=True)
        return float(-(z[np.arange(len(labels)), labels] - np.log(np.exp(z).sum(axis=1))).mean())
    a, b = 1 / high, 1 / low
    ratio = (5 ** 0.5 - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return 2 / (a + b)


def report(probs, labels, n_bins=15):
    return {"accuracy": accuracy(probs, labels), "ece": ece(probs, labels, n_bins), "ace": ece(probs, labels, n_bins, "mass"), "mce": mce(probs, labels, n_bins), "classwise_ece": classwise_ece(probs, labels, n_bins), "brier": brier(probs, labels), "nll": nll(probs, labels)}


def bootstrap_ci(metric, probs, labels, rounds=1000, level=0.95, seed=0):
    """A percentile bootstrap interval for any metric(probs, labels)."""
    probs, labels = _prepare(probs, labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    values = [metric(probs[i], labels[i]) for i in (rng.integers(0, n, n) for _ in range(rounds))]
    a = (1 - level) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))

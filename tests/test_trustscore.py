import unittest

import numpy as np

from trustscore import APS, CQR, LAC, IntervalRegressor, Mondrian, NormalisedRegressor, accuracy, bootstrap_ci, brier, classwise_ece, conformal_quantile, coverage, coverage_report, ece, fit_temperature, interval_coverage, mce, nll, reliability, report, softmax


def simulate(n, classes=5, sharpness=2.0, overconfidence=3.0, seed=0):
    """True probabilities come from softmax(z); the model reports z scaled up, so it is overconfident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, classes)) * sharpness
    p = softmax(z)
    labels = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return z * overconfidence, labels


class TestMetrics(unittest.TestCase):
    def test_hand_example(self):
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        labels = np.array([0, 1, 1, 0])
        self.assertEqual(accuracy(probs, labels), 0.75)
        self.assertAlmostEqual(ece(probs, labels, 10), (abs(0.9 - 1) + abs(0.8 - 0) + abs(0.7 - 1) + abs(0.6 - 1)) / 4)
        self.assertAlmostEqual(mce(probs, labels, 10), 0.8)
        self.assertAlmostEqual(brier(probs[:1], labels[:1]), 0.02)
        self.assertAlmostEqual(nll(probs[:1], labels[:1]), -np.log(0.9))

    def test_perfect_calibration_gives_small_ece(self):
        logits, labels = simulate(200000, overconfidence=1.0)
        self.assertLess(ece(softmax(logits), labels), 0.005)

    def test_overconfidence_is_detected(self):
        logits, labels = simulate(50000, overconfidence=3.0)
        self.assertGreater(ece(softmax(logits), labels), 0.05)

    def test_reliability_rows(self):
        logits, labels = simulate(5000)
        rows = reliability(softmax(logits), labels, 10)
        self.assertEqual(sum(r["count"] for r in rows), 5000)
        self.assertTrue(all(0 <= r["accuracy"] <= 1 for r in rows))

    def test_mass_bins_are_even(self):
        logits, labels = simulate(1000)
        counts = [r["count"] for r in reliability(softmax(logits), labels, 10, "mass")]
        self.assertEqual(set(counts), {100})

    def test_confidence_of_one_lands_in_the_last_bin(self):
        self.assertEqual(reliability(np.array([[1.0, 0.0]]), np.array([0]), 10)[0]["bin"], 9)

    def test_classwise(self):
        logits, labels = simulate(50000, overconfidence=1.0)
        self.assertLess(classwise_ece(softmax(logits), labels, 10), 0.01)

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            ece(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            ece(np.zeros((3, 2)), np.zeros(2))

    def test_softmax_is_stable_for_huge_logits(self):
        p = softmax(np.array([[1000.0, 0.0], [-1000.0, 1000.0]]))
        self.assertFalse(np.isnan(p).any())
        self.assertAlmostEqual(p[0, 0], 1)

    def test_bootstrap_interval_contains_the_estimate(self):
        logits, labels = simulate(2000)
        p = softmax(logits)
        lo, hi = bootstrap_ci(accuracy, p, labels, rounds=200)
        self.assertLess(lo, accuracy(p, labels))
        self.assertGreater(hi, accuracy(p, labels))
        self.assertLess(hi - lo, 0.1)


class TestTemperature(unittest.TestCase):
    def test_recovers_the_true_temperature(self):
        for t in (0.5, 1.0, 3.0):
            logits, labels = simulate(100000, overconfidence=t, seed=1)
            self.assertAlmostEqual(fit_temperature(logits, labels), t, delta=0.08 * t)

    def test_improves_calibration_without_changing_accuracy(self):
        logits, labels = simulate(40000, seed=2)
        t = fit_temperature(logits[:20000], labels[:20000])
        before, after = softmax(logits[20000:]), softmax(logits[20000:], t)
        self.assertLess(ece(after, labels[20000:]), ece(before, labels[20000:]) / 3)
        self.assertLess(nll(after, labels[20000:]), nll(before, labels[20000:]))
        self.assertEqual(accuracy(after, labels[20000:]), accuracy(before, labels[20000:]))

    def test_nan_refused(self):
        with self.assertRaises(ValueError):
            fit_temperature(np.array([[np.nan, 0.0]]), np.array([0]))

    def test_report_keys(self):
        logits, labels = simulate(1000)
        self.assertEqual(set(report(softmax(logits), labels)), {"accuracy", "ece", "ace", "mce", "classwise_ece", "brier", "nll"})


class TestConformal(unittest.TestCase):
    def test_quantile_rule(self):
        s = np.arange(1, 20) / 20
        self.assertAlmostEqual(conformal_quantile(s, 0.1), s[17])
        self.assertEqual(conformal_quantile([0.5], 0.1), np.inf)
        self.assertEqual(conformal_quantile([], 0.1), np.inf)

    def coverage_of(self, make, alpha=0.1, trials=30, cal=600, test=1000):
        out = []
        for seed in range(trials):
            logits, labels = simulate(cal + test, seed=seed)
            p = softmax(logits)
            method = make().fit(p[:cal], labels[:cal], alpha)
            out.append(coverage(method.predict(p[cal:]), labels[cal:]))
        return float(np.mean(out))

    def test_guarantee_holds_without_being_wasteful(self):
        for make in (LAC, APS, Mondrian, lambda: APS(randomised=False)):
            for alpha in (0.05, 0.1, 0.2):
                c = self.coverage_of(make, alpha)
                self.assertGreater(c, 1 - alpha - 0.012)
                self.assertLess(c, 1 - alpha + 0.03 if make is not Mondrian else 1 - alpha + 0.05)

    def test_lac_makes_smaller_sets_than_aps(self):
        logits, labels = simulate(4000, seed=4)
        p = softmax(logits)
        a = coverage_report(LAC().fit(p[:2000], labels[:2000]).predict(p[2000:]), labels[2000:])
        b = coverage_report(APS().fit(p[:2000], labels[:2000]).predict(p[2000:]), labels[2000:])
        self.assertLess(a["mean_size"], b["mean_size"])

    def test_mondrian_protects_a_rare_class(self):
        rng = np.random.default_rng(5)
        n = 30000
        labels = (rng.random(n) < 0.05).astype(int)
        scores = rng.normal(0, 1, n) + np.where(labels == 1, 0.8, -0.8)
        p1 = 1 / (1 + np.exp(-scores * 0.5 - 2))         # a model that gives the two classes unequal scores
        p = np.column_stack([p1, 1 - p1])
        half = n // 2
        common = LAC().fit(p[:half], labels[:half]).predict(p[half:])
        per_class = Mondrian().fit(p[:half], labels[:half]).predict(p[half:])
        rep_a = coverage_report(common, labels[half:])["by_class"]
        rep_b = coverage_report(per_class, labels[half:])["by_class"]
        worst_a = min(rep_a.values())
        self.assertGreater(min(rep_b.values()), 0.87)
        self.assertLess(worst_a, min(rep_b.values()))

    def test_full_coverage_with_too_little_calibration_data(self):
        p = np.array([[0.7, 0.3]] * 5)
        s = LAC().fit(p, np.zeros(5, int), 0.01).predict(p)
        self.assertTrue(s.all())

    def test_report_by_group(self):
        s = np.array([[True, False], [True, True], [False, True]])
        r = coverage_report(s, np.array([0, 1, 0]), groups=["a", "a", "b"])
        self.assertAlmostEqual(r["coverage"], 2 / 3)
        self.assertEqual(r["by_group"]["a"]["coverage"], 1)
        self.assertEqual(r["by_group"]["b"]["n"], 1)
        self.assertAlmostEqual(r["singleton"], 2 / 3)

    def test_shift_breaks_the_guarantee(self):
        """The documented failure: conformal cannot promise coverage on shifted data."""
        rng = np.random.default_rng(6)
        pred = rng.normal(size=20000)
        y = pred + rng.normal(size=20000)
        m = IntervalRegressor().fit(pred[:10000], y[:10000], 0.1)
        shifted = pred[10000:] + rng.normal(scale=3, size=10000)
        lo, hi = m.predict(pred[10000:])
        self.assertLess(interval_coverage(lo, hi, shifted), 0.7)


class TestRegression(unittest.TestCase):
    def data(self, n, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 5, n)
        noise = 0.3 + x * 0.6
        return x, np.sin(x) + rng.normal(size=n) * noise, noise

    def test_interval_coverage(self):
        x, y, _ = self.data(40000, 1)
        pred = np.sin(x)
        m = IntervalRegressor().fit(pred[:2000], y[:2000], 0.1)
        self.assertAlmostEqual(interval_coverage(*m.predict(pred[2000:]), y[2000:]), 0.9, delta=0.02)

    def test_normalised_intervals_adapt_and_are_shorter(self):
        x, y, noise = self.data(40000, 2)
        pred = np.sin(x)
        fixed = IntervalRegressor().fit(pred[:5000], y[:5000], 0.1)
        adaptive = NormalisedRegressor().fit(pred[:5000], noise[:5000], y[:5000], 0.1)
        lo, hi = adaptive.predict(pred[5000:], noise[5000:])
        self.assertAlmostEqual(interval_coverage(lo, hi, y[5000:]), 0.9, delta=0.02)
        a, b = fixed.predict(pred[5000:])
        self.assertLess((hi - lo).mean(), (b - a).mean())
        low_x = x[5000:] < 1
        self.assertGreater(interval_coverage(lo[low_x], hi[low_x], y[5000:][low_x]), 0.86)
        self.assertGreater(interval_coverage(a[low_x], b[low_x], y[5000:][low_x]), 0.98)

    def test_cqr_corrects_bad_quantiles(self):
        x, y, noise = self.data(40000, 3)
        lower, upper = np.sin(x) - 0.5 * noise, np.sin(x) + 0.5 * noise      # far too narrow on purpose
        m = CQR().fit(lower[:5000], upper[:5000], y[:5000], 0.1)
        self.assertLess(interval_coverage(lower[5000:], upper[5000:], y[5000:]), 0.5)
        self.assertAlmostEqual(interval_coverage(*m.predict(lower[5000:], upper[5000:]), y[5000:]), 0.9, delta=0.02)


if __name__ == "__main__":
    unittest.main()

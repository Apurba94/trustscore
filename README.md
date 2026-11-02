# TrustScore

Can you trust what a model says? This toolkit answers that with two ideas from current research: **calibration** (does 90% confidence mean right 90% of the time?) and **conformal prediction** (a set of answers that contains the truth with a chosen probability). Medical image models are a natural use, because a wrong confident answer there can do harm.

```python
from trustscore import APS, coverage_report, ece, fit_temperature, softmax

t = fit_temperature(val_logits, val_labels)             # fix overconfidence
print(ece(softmax(test_logits), test_labels), "->", ece(softmax(test_logits, t), test_labels))

aps = APS().fit(softmax(cal_logits, t), cal_labels, alpha=0.1)     # aim for 90% coverage
sets = aps.predict(softmax(test_logits, t))
print(coverage_report(sets, test_labels, groups=test_sources))
```

From a saved results file:

```
python -m trustscore results.npz 0.1
```

## What is included

**Calibration:** ECE with equal-width or equal-mass bins, maximum error, class-wise ECE, Brier score, NLL, reliability tables, bootstrap intervals, and temperature scaling.

**Conformal classification:**

| Method | Good for |
|---|---|
| `LAC` | The smallest sets on average |
| `APS` | Sets that adapt: larger when a case is hard |
| `Mondrian` | Coverage for each class, including rare ones |

**Conformal regression:** `IntervalRegressor`, `NormalisedRegressor` (intervals that widen with difficulty) and `CQR` (conformalised quantile regression). These suit tasks such as cell-count intervals.

## How it is tested

Tests use simulated models whose true probabilities are known. Temperature scaling must recover the temperature that was applied. Each conformal method must reach its target coverage across 30 random splits at three levels of alpha, without overshooting by much. One test deliberately applies distribution shift and shows that coverage then fails.

## An honest warning

The conformal guarantee covers the average over all cases, and it assumes calibration and test data come from the same distribution. A new laboratory, scanner or stain can break it without warning, so always check `coverage_report` by source when you can. Temperature scaling on a validation set does not fix calibration on shifted data either.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.

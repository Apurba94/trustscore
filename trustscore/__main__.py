"""Check a model's trustworthiness from a .npz file holding `logits` and `labels`.

Usage: python -m trustscore results.npz [alpha]
The first half calibrates, and the second half is used for the conformal check.
"""
import json
import sys

import numpy as np

from . import APS, coverage_report, fit_temperature, report, softmax


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    data = np.load(argv[1])
    logits, labels = data["logits"], data["labels"]
    alpha = float(argv[2]) if len(argv) > 2 else 0.1
    half = len(labels) // 2
    t = fit_temperature(logits[:half], labels[:half])
    before = report(softmax(logits[half:]), labels[half:])
    after = report(softmax(logits[half:], t), labels[half:])
    aps = APS().fit(softmax(logits[:half], t), labels[:half], alpha)
    sets = coverage_report(aps.predict(softmax(logits[half:], t)), labels[half:])
    print(json.dumps({"temperature": t, "before": before, "after": after, "conformal": sets}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

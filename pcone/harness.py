"""Shared evaluation harness for post-processing ideas on top of the combined detector.

Everything is scored with photo-grouped nested CV on TRAIN only:
for each of 5 seeds x 5 folds, parameters are chosen on the 4 training folds (max F1) and
applied to the held-out fold; the OOF predictions of all folds give one F1 per seed.

Usage inside an experiment:
    from harness import H
    h = H()                                     # loads labels, groups, combined detections
    base = h.baseline_preds()                   # {(thr, delta): pred over train crops}
    print(h.nested_cv(base))                    # mean, std, per-seed list
    # your idea: build {param_tuple: pred_vector} (include the thr/delta params in the key)
    print(h.compare(base, mine))                # paired per-seed difference vs baseline
Test-side: h.crop_dist(paths, det) / h.test_paths give what is needed to write a submission.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from data import read_list  # noqa: E402
from eval_detect import crop_dist, predict  # noqa: E402
from mosaic import parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2, 3, 4)


class H:
    def __init__(self, det_csv=ROOT / "data/detect/combined/detections.csv"):
        tr = read_list(ROOT / "data/contest/train_label.csv")
        te = read_list(ROOT / "data/contest/test.txt")
        self.train_paths = [r[0] for r in tr]
        self.test_paths = [r[0] for r in te]
        self.y = np.array([int(r[1]) for r in tr])
        self.groups = np.array([parse(p)[0] for p in self.train_paths])
        self.train_photos = sorted(set(self.groups))
        self.test_photos = sorted({parse(p)[0] for p in self.test_paths})
        self.det = pd.read_csv(det_csv, dtype={"photo": str})
        self.crop_dist = crop_dist
        self.predict = predict
        self._splits = {s: list(StratifiedGroupKFold(5, shuffle=True, random_state=s)
                                .split(self.train_paths, self.y, self.groups)) for s in SEEDS}

    def thr_grid(self, det=None, n=60):
        det = self.det if det is None else det
        return np.unique(np.quantile(det["score"], np.linspace(0, 0.99, n)))

    def baseline_preds(self, det=None, deltas=np.arange(-8, 9, 1.0)):
        det = self.det if det is None else det
        cd = crop_dist(self.train_paths, det)
        return {(float(t), float(d)): predict(cd, t, d) for d in deltas for t in self.thr_grid(det)}

    def nested_cv(self, preds):
        keys = list(preds)
        P = np.stack([preds[k] for k in keys])  # (K, N)
        y = self.y
        per_seed, chosen = [], []
        for s in SEEDS:
            oof = np.zeros(len(y), int)
            for tri, vai in self._splits[s]:
                tp = (P[:, tri] & (y[tri] == 1)).sum(1)
                fp = (P[:, tri] & (y[tri] == 0)).sum(1)
                fn = ((1 - P[:, tri]) & (y[tri] == 1)).sum(1)
                f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
                k = int(np.argmax(f1))
                oof[vai] = P[k, vai]; chosen.append(keys[k])
            per_seed.append(f1_score(y, oof))
        return {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                "per_seed": [round(v, 4) for v in per_seed], "chosen": chosen}

    def full_fit(self, preds):
        """Best params on ALL train (for the final submission) and its train F1."""
        best = max(preds, key=lambda k: f1_score(self.y, preds[k]))
        return best, float(f1_score(self.y, preds[best]))

    def compare(self, base, mine):
        a, b = self.nested_cv(base), self.nested_cv(mine)
        diff = np.array(b["per_seed"]) - np.array(a["per_seed"])
        return {"baseline": a["mean"], "mine": b["mean"], "mine_std": b["std"],
                "paired_diff_mean": float(diff.mean()), "paired_diff_per_seed": diff.round(4).tolist(),
                "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}


if __name__ == "__main__":
    h = H()
    r = h.nested_cv(h.baseline_preds())
    print("baseline nested CV", round(r["mean"], 4), "+-", round(r["std"], 4), r["per_seed"])

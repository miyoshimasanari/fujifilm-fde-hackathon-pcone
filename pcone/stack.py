"""Stack CNN probabilities with label-free detector geometry.

Features per crop (all computable for test the same way as for train):
  - logit of CNN prob (train: out-of-fold), max CNN prob over the 8 neighbouring crops
  - per detector: max detection score among circles within gap <= g px of the crop square,
    for several g (the circle-vs-square margin the labellers used is fuzzy)
Model: logistic regression, evaluated with StratifiedGroupKFold by photo (out-of-fold), threshold
chosen to maximise OOF F1. Writes submission.csv.

  python pcone/stack.py --cnn runs/ctx_small_m32 --out runs/stack1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data import read_list  # noqa: E402
from eval_detect import crop_dist  # noqa: E402
from mosaic import COLS, ROWS, parse  # noqa: E402
from train_cv import best_threshold  # noqa: E402

GAPS = (-8, -4, 0, 4, 8)


def det_features(paths, det_csv):
    det = pd.read_csv(det_csv, dtype={"photo": str})
    if len(det):
        s = det["score"].to_numpy(float)
        det["score"] = (s - s.min()) / (s.max() - s.min() + 1e-9)  # per-detector 0..1
    cd = crop_dist(paths, det)
    F = np.zeros((len(paths), len(GAPS)))
    for i, x in enumerate(cd):
        for j, g in enumerate(GAPS):
            m = x[:, 1] <= g if len(x) else np.zeros(0, bool)
            F[i, j] = x[m, 0].max() if m.any() else 0.0
    return F


def neighbour_max(paths, prob):
    key = {}
    for i, p in enumerate(paths):
        q = parse(p)
        if q:
            key[(q[0], q[1])] = max(key.get((q[0], q[1]), 0), prob[i])
    out = np.zeros(len(paths))
    for i, p in enumerate(paths):
        q = parse(p)
        if not q:
            continue
        r, c = divmod(q[1] - 1, COLS)
        best = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                    best = max(best, key.get((q[0], (r + dr) * COLS + c + dc + 1), 0.0))
        out[i] = best
    return out


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def build(paths, cnn_prob, detectors):
    cols = [logit(cnn_prob), logit(neighbour_max(paths, cnn_prob))]
    names = ["cnn", "cnn_nb"]
    for name, csv in detectors.items():
        F = det_features(paths, csv)
        cols += list(F.T); names += [f"{name}_g{g}" for g in GAPS]
    return np.stack(cols, 1), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", nargs="+", required=True, help="run dirs with oof.npy/test_prob.npy (averaged)")
    ap.add_argument("--det", nargs="*", default=["log", "hough", "contour"])
    ap.add_argument("--train", default="data/contest/train_label.csv")
    ap.add_argument("--test", default="data/contest/test.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=0.3)
    a = ap.parse_args()
    tr = read_list(a.train); te = read_list(a.test)
    trp = [r[0] for r in tr]; tep = [r[0] for r in te]
    y = np.array([int(r[1]) for r in tr])
    groups = np.array([parse(p)[0] for p in trp])
    oof = np.mean([np.load(Path(d) / "oof.npy") for d in a.cnn], 0)
    tp = np.mean([np.load(Path(d) / "test_prob.npy") for d in a.cnn], 0)
    assert not np.isnan(oof).any(), "CNN OOF incomplete"
    dets = {n: f"data/detect/{n}/detections.csv" for n in a.det}
    Xtr, names = build(trp, oof, dets)
    Xte, _ = build(tep, tp, dets)

    res = {}
    th, f = best_threshold(y, oof); res["cnn_only"] = (f, th)
    # stacked logistic regression, OOF by photo
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=a.C, class_weight="balanced",
                                                                       max_iter=2000))
    soof = np.zeros(len(y))
    for seed in range(5):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + seed)
        for tri, vai in skf.split(Xtr, y, groups):
            soof[vai] += model().fit(Xtr[tri], y[tri]).predict_proba(Xtr[vai])[:, 1] / 5
    th_s, f_s = best_threshold(y, soof); res["stack"] = (f_s, th_s)
    for k, v in res.items():
        print(f"{k:10s} OOF F1={v[0]:.4f} @ {v[1]:.2f}")
    full = model().fit(Xtr, y)
    coef = dict(zip(names, np.round(full[-1].coef_[0], 3)))
    print("coef", coef)
    pt = full.predict_proba(Xte)[:, 1]
    pred = (pt >= th_s).astype(int)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{p},{v}\n" for p, v in zip(tep, pred))
    np.save(out / "stack_oof.npy", soof); np.save(out / "stack_test.npy", pt)
    json.dump({"results": res, "coef": {k: float(v) for k, v in coef.items()}, "threshold": th_s,
               "test_pos": int(pred.sum()), "cnn": a.cnn, "det": a.det}, open(out / "report.json", "w"),
              indent=1)
    print(f"test positives={pred.sum()}/{len(pred)} -> {out/'submission.csv'}")


if __name__ == "__main__":
    main()

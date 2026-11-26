"""Turn hole detections (circles on half-res mosaics) into crop labels and score them.

The contest labels mark every crop whose square the P-cone circle touches (verified visually on
training photos), so a crop is positive when dist(circle centre, crop square) <= r + delta.
Score threshold and delta are tuned on TRAINING photos only, then applied to test photos.

  python pcone/eval_detect.py data/detect/log/detections.csv [--out runs/det_log]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from data import read_list  # noqa: E402
from mosaic import cell, parse  # noqa: E402

T = 128  # half-res crop size


def crop_dist(rel_paths, det):
    """For each crop: list of (score, distance-from-circle-edge) of detections in its photo."""
    by_photo = {str(k): g[["x", "y", "r", "score"]].to_numpy(float) for k, g in det.groupby("photo")}
    out = []
    for rp in rel_paths:
        pid, idx, _ = parse(rp)
        d = by_photo.get(pid)
        if d is None or len(d) == 0:
            out.append(np.zeros((0, 2))); continue
        r, c = cell(idx)
        x0, y0 = c * T, r * T
        dx = np.maximum(np.maximum(x0 - d[:, 0], 0), d[:, 0] - (x0 + T))
        dy = np.maximum(np.maximum(y0 - d[:, 1], 0), d[:, 1] - (y0 + T))
        gap = np.hypot(dx, dy) - d[:, 2]  # <=0 means the circle touches the square
        out.append(np.stack([d[:, 3], gap], 1))
    return out


def predict(cd, thr, delta):
    return np.array([int(((x[:, 0] >= thr) & (x[:, 1] <= delta)).any()) if len(x) else 0 for x in cd])


def crop_score(cd, delta):
    """Max detection score among circles touching the crop (for ensembling)."""
    return np.array([x[x[:, 1] <= delta, 0].max() if len(x) and (x[:, 1] <= delta).any() else 0.0
                     for x in cd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections")
    ap.add_argument("--train", default="data/contest/train_label.csv")
    ap.add_argument("--test", default="data/contest/test.txt")
    ap.add_argument("--out")
    a = ap.parse_args()
    det = pd.read_csv(a.detections, dtype={"photo": str})
    tr = read_list(a.train); te = read_list(a.test)
    y = np.array([int(r[1]) for r in tr])
    cd_tr = crop_dist([r[0] for r in tr], det)
    scores = np.unique(det["score"].to_numpy())
    qs = np.unique(np.quantile(scores, np.linspace(0, 0.99, 100))) if len(scores) else [0]
    best = (-1, None, None)
    for delta in np.arange(-8, 9, 1.0):
        for thr in qs:
            f = f1_score(y, predict(cd_tr, thr, delta), zero_division=0)
            if f > best[0]:
                best = (f, float(thr), float(delta))
    f, thr, delta = best
    p = predict(cd_tr, thr, delta)
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    print(f"{a.detections}: train F1={f:.4f} thr={thr:.4g} delta={delta:+.0f}px  TP={tp} FP={fp} FN={fn}")
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        cd_te = crop_dist([r[0] for r in te], det)
        pt = predict(cd_te, thr, delta)
        with open(out / "submission.csv", "w", newline="\n") as fh:
            fh.writelines(f"{r[0]},{v}\n" for r, v in zip(te, pt))
        np.save(out / "train_score.npy", crop_score(cd_tr, delta))
        np.save(out / "test_score.npy", crop_score(cd_te, delta))
        json.dump({"train_f1": f, "thr": thr, "delta": delta, "tp": tp, "fp": fp, "fn": fn,
                   "test_pos": int(pt.sum())}, open(out / "report.json", "w"), indent=1)
        print(f"test positives={pt.sum()}/{len(pt)} -> {out/'submission.csv'}")


if __name__ == "__main__":
    main()

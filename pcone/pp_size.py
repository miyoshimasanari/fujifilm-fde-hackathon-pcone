"""Per-photo size prior for the combined P-cone detector (idea "size").

Field premise: P-cones are standard parts, and one photo shows one wall from one distance, so the
P-cone marks in one photo should appear at (nearly) the same size.  The detectors estimate each
mark's radius independently, so a radius that disagrees with the rest of the photo is either a
misfit (rim vs halo vs inner shadow) or not a P-cone at all (spall, pipe, odd object).

Rule (label-free, applied identically to train and test photos; only the 2-3 numbers below are
chosen by photo-grouped nested CV on train):
  C(photo)  = detections with score >= S_CONF ("confident": >= two methods at their cap).
  For each detection i:  m_i = median radius of the OTHER confident detections of its photo,
                         n_i = their number (leave-one-out, so a detection never votes for itself).
  (a) snap   if n_i >= MIN_N:                  r_i <- (1 - a) * r_i + a * m_i
  (b) demote if n_i >= MIN_N and max(r_i/m_i, m_i/r_i) > k:   score_i <- DEMOTE * score_i
  Photos with fewer than MIN_N other confident detections are left unchanged.
Then the usual crop rule: positive iff some detection with score >= thr has gap <= delta
(gap = dist(centre, crop square) - r, half-res px), thr/delta tuned as in the baseline.

  .venv/bin/python pcone/pp_size.py      -> prints the paired comparison, writes runs/pp_size/
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from harness import H, ROOT  # noqa: E402

OUT = ROOT / "runs/pp_size"
S_CONF = 0.9          # "confident" = evidence >= 1.8 (two methods at their cap)
DEMOTE = 0.5          # score multiplier for size outliers
MIN_N = 2             # need >= 2 other confident marks in the photo to trust its median
GRID_A = (0.0, 0.25, 0.5)          # snap strength
GRID_K = (np.inf, 2.0)             # size-outlier ratio (inf = no demotion)
DELTAS = np.arange(-8, 9, 1.0)     # same as the baseline


def size_prior(det, a, k, min_n=MIN_N, s_conf=S_CONF, demote=DEMOTE):
    """Return a copy of det with radii snapped toward / scores demoted by the photo's size prior."""
    det = det.copy()
    r_new = det["r"].to_numpy(float).copy()
    s_new = det["score"].to_numpy(float).copy()
    det["m_loo"], det["n_loo"] = np.nan, 0
    for _, g in det.groupby("photo"):
        idx = g.index.to_numpy()
        r = g["r"].to_numpy(float)
        conf = g["score"].to_numpy(float) >= s_conf
        for j, i in enumerate(idx):
            others = conf.copy(); others[j] = False
            n = int(others.sum())
            det.loc[i, "n_loo"] = n
            if n < min_n:
                continue
            m = float(np.median(r[others]))
            det.loc[i, "m_loo"] = m
            pos = det.index.get_loc(i)
            r_new[pos] = (1 - a) * r[j] + a * m
            if max(r[j] / m, m / r[j]) > k:
                s_new[pos] = demote * s_new[pos]
    det["r_orig"], det["score_orig"] = det["r"], det["score"]
    det["r"], det["score"] = r_new, s_new
    return det


def _flat(h, paths, det):
    cd = h.crop_dist(paths, det)
    ci = np.concatenate([np.full(len(x), i) for i, x in enumerate(cd)]) if cd else np.zeros(0, int)
    X = np.concatenate([x for x in cd if len(x)]) if any(len(x) for x in cd) else np.zeros((0, 2))
    return ci.astype(int), X


def _preds_for(h, paths, det, thrs, deltas):
    """{(thr, delta): uint8 pred} with the baseline rule, vectorised over thresholds."""
    ci, X = _flat(h, paths, det)
    out = {}
    for d in deltas:
        ms = np.full(len(paths), -np.inf)
        ok = X[:, 1] <= d
        np.maximum.at(ms, ci[ok], X[ok, 0])
        for t in thrs:
            out[(float(t), float(d))] = (ms >= t).astype(np.uint8)
    return out


def train_preds(h, grid_a=GRID_A, grid_k=GRID_K):
    thrs = h.thr_grid()           # same threshold grid as the baseline (original scores)
    preds = {}
    for a, k in itertools.product(grid_a, grid_k):
        dm = size_prior(h.det, a, k)
        for (t, d), p in _preds_for(h, h.train_paths, dm, thrs, DELTAS).items():
            preds[(t, d, float(a), float(k))] = p
    return preds


def test_pred(h, params):
    t, d, a, k = params
    dm = size_prior(h.det, a, k)
    return _preds_for(h, h.test_paths, dm, [t], [d])[(float(t), float(d))]


def errors(h, pred):
    fn = [Path(p).stem for p, y, q in zip(h.train_paths, h.y, pred) if y == 1 and q == 0]
    fp = [Path(p).stem for p, y, q in zip(h.train_paths, h.y, pred) if y == 0 and q == 1]
    return fn, fp


def diagnostics(h, thr_delta):
    """How well does the premise hold on TRAIN photos, and what each variant changes at fixed thr/delta."""
    c = h.det[h.det.photo.isin(h.train_photos) & (h.det.score >= S_CONF)]
    lr = np.log(c.r)
    within = lr - lr.groupby(c.photo).transform("mean")
    out = {"sd_log_r_within_photo": float(np.sqrt((within ** 2).sum() / (len(c) - c.photo.nunique()))),
           "sd_log_r_between_photos": float(np.log(c.groupby("photo").r.median()).std())}
    base = _preds_for(h, h.train_paths, h.det, [thr_delta[0]], [thr_delta[1]])[thr_delta]
    for a, k in itertools.product(GRID_A, GRID_K):
        if a == 0 and np.isinf(k):
            continue
        p = _preds_for(h, h.train_paths, size_prior(h.det, a, k), [thr_delta[0]], [thr_delta[1]])[thr_delta]
        ch = np.nonzero(p != base)[0]
        out[f"a={a},k={k}"] = {"fixed": [Path(h.train_paths[i]).stem for i in ch if p[i] == h.y[i]],
                               "new": [Path(h.train_paths[i]).stem for i in ch if p[i] != h.y[i]]}
    return out


def main():
    h = H()
    base = h.baseline_preds()
    mine = train_preds(h)
    cmp_ = h.compare(base, mine)
    print("compare:", json.dumps({k: v for k, v in cmp_.items()}, indent=None))
    nc = h.nested_cv(mine)
    from collections import Counter
    print("nested chosen (a, k):", Counter((c[2], c[3]) for c in nc["chosen"]).most_common())

    bk, bf = h.full_fit(base)
    mk, mf = h.full_fit(mine)
    print("baseline full fit", bk, round(bf, 4), "| mine full fit", mk, round(mf, 4))
    bfn, bfp = errors(h, base[bk]); mfn, mfp = errors(h, mine[mk])
    fixed = sorted(set(bfn + bfp) - set(mfn + mfp)); new = sorted(set(mfn + mfp) - set(bfn + bfp))
    p = mine[mk]
    tp = int(((p == 1) & (h.y == 1)).sum()); fp = int(((p == 1) & (h.y == 0)).sum())
    fn = int(((p == 0) & (h.y == 1)).sum())
    print(f"mine full fit TP{tp} FP{fp} FN{fn}; fixed={fixed} new={new}")

    pt = test_pred(h, mk)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{rp},{int(v)}\n" for rp, v in zip(h.test_paths, pt))
    rep = dict(cmp_, full_fit_params=list(mk), full_fit_train_f1=mf, tp=tp, fp=fp, fn=fn,
               fixed_errors=fixed, new_errors=new, test_positives=int(pt.sum()),
               baseline_full_fit=list(bk), baseline_full_fit_f1=bf,
               nested_chosen=[list(c) for c in nc["chosen"]],
               grid=dict(a=list(GRID_A), k=[str(k) for k in GRID_K], min_n=MIN_N, s_conf=S_CONF,
                         demote=DEMOTE),
               diagnostics=diagnostics(h, (float(bk[0]), float(bk[1]))))
    print("diagnostics:", json.dumps(rep["diagnostics"]))
    json.dump(rep, open(OUT / "report.json", "w"), indent=1, default=float)
    print(f"test positives={int(pt.sum())}/{len(pt)} -> {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()

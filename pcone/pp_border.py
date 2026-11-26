"""Border-cut P-cone marks (撮影の枠で切れたPコン跡): a label-free candidate step along the photo frame.

Field premise: the photographer frames a wall section, not the marks, so a P-cone mark can be cut by
the photo's outer frame.  The labels still count such a crop as "contains a mark" even when only a
thin dark sliver is visible.  The three main detectors (LoG / Hough / contour) verify a circle by its
rim and deliberately suppress circles with < ~40-50 % of the rim inside the photo, so slivers at the
frame get no (or too little) evidence.

This module adds one dedicated, fixed step (no labels, no per-photo tuning; same code on test):
  1. Same preprocessing as detect_log.py: grey / large-scale median background -> N (1 = wall),
     darkness dk = 1 - N, light Gaussian blur.  Missing (black) cells are filled and excluded.
  2. For each of the four outer edges, take an edge-aligned strip (t = depth 0..DS-1 from the frame,
     s = position along the frame).  Mask dk > TAU, open 3x3, keep connected components that TOUCH
     the frame (rows t <= 1).
  3. Each component is treated as the visible part of a disc cut by the frame: its contact interval
     on the frame (chord) and its inner boundary (deepest pixel per column) are fitted with one circle
     (Kasa least squares; the centre may lie outside the photo).
  4. Physical gates:
       chord >= 16, visible depth >= 6 half-res px, extent along the frame <= 2.2 r (no elongated
       bands / vignetting / joints), the component must end inside the strip (big dark objects,
       pipes, hoses run on), must not touch filled (missing) cells;
       20 <= r <= 64 half-res px.  The upper bound is the P-cone size prior (~95th pct of confident
       train detections).  The lower bound is the LoG detector's own pore rule ("r < ~20 px must be a
       near-perfect ellipse: pores / bug holes"): a sliver at the frame cannot be checked for a full
       ellipse, so small border blobs are indistinguishable from bug holes and are not used.
  5. score = ramp(mean darkness, 0.15 -> 0.45) * ramp(circle residual / r, 0.25 -> 0.05): dark AND
     round.  Elongated dark streaks and soft shadows fail the roundness term; faint stains fail the
     darkness term.
  6. A crop is positive from this step when a border candidate with score >= bthr touches it
     (same rule as the main pipeline: dist(centre, crop square) <= r + delta).  Final prediction =
     baseline (combined detector at thr, delta) OR border step (bthr, delta).
     bthr (incl. "off" = 9.0) is chosen by photo-grouped nested CV together with thr and delta.

Run:  .venv/bin/python pcone/pp_border.py
  -> prints h.compare(baseline, border), full fit, writes runs/pp_border/{submission.csv, report.json,
     border_cands.csv (features of all frame-touching components, train + test, computed blindly),
     fixed_train.jpg (train-only overlay of the crops the step changes)}.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from detect_log import load, normalise  # noqa: E402  (identical preprocessing to the LoG detector)
from eval_detect import crop_dist, predict  # noqa: E402
from harness import H, ROOT  # noqa: E402
from mosaic import cell, parse  # noqa: E402

OUT = ROOT / "runs/pp_border"
CACHE = OUT / "border_cands.csv"

ROWS, COLS, T = 14, 18, 128
DS = 80              # strip depth (half-res px) searched from each frame edge
TAU = 0.15           # darkness threshold (relative to the large-scale wall background)
MIN_AREA = 15
CHORD_MIN = 16       # contact length on the frame (half-res px)
DEPTH_MIN = 6        # visible depth into the photo (half-res px = 12 full-res px)
R_MIN, R_MAX = 20.0, 64.0
WIDTH_MAX = 2.2      # extent along the frame <= WIDTH_MAX * r
BTHR = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9.0)   # 9.0 = border step off
COLUMNS = ["photo", "edge", "x", "y", "r", "chord", "depth", "width", "area", "res", "mdk", "inside", "bad"]


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def kasa(px, py):
    A = np.stack([px, py, np.ones_like(px)], 1)
    b = -(px ** 2 + py ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    return cx, cy, math.sqrt(max(cx * cx + cy * cy - F, 1e-6))


def strips(A, H_, W_):
    """Edge-aligned views (row t = 0 is the frame) and their (s, t) -> (x, y) maps."""
    return [("top", A[:DS, :], lambda s, t: (s, t)),
            ("bottom", A[::-1][:DS, :], lambda s, t: (s, H_ - 1 - t)),
            ("left", A[:, :DS].T, lambda s, t: (t, s)),
            ("right", A[:, ::-1][:, :DS].T, lambda s, t: (W_ - 1 - t, s))]


def border_components(photo):
    """Features of every dark component touching the photo frame (no gating, no score)."""
    grey, _, valid = load(photo)
    N = cv2.GaussianBlur(normalise(grey, valid), (0, 0), 1.0)
    dk = 1.0 - N
    H_, W_ = dk.shape
    bad = cv2.dilate((~valid).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = []
    for (edge, V, to_xy), (_, Bd, _) in zip(strips(dk, H_, W_), strips(bad, H_, W_)):
        m = ((V > TAU) & ~Bd).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, lab = cv2.connectedComponents(m, connectivity=8)
        for k in sorted(set(np.unique(lab[:2])) - {0}):
            comp = lab == k
            ts, ss = np.nonzero(comp)
            if len(ts) < MIN_AREA:
                continue
            contact = np.nonzero(comp[:2].any(0))[0]
            s_a, s_b = int(contact.min()), int(contact.max())
            s_lo, s_hi = int(ss.min()), int(ss.max())
            depth = int(ts.max()) + 1
            prof = np.zeros(s_hi - s_lo + 1)
            np.maximum.at(prof, ss - s_lo, ts + 1)
            bx = np.r_[np.arange(s_lo, s_hi + 1, dtype=float), s_a - 0.5, s_b + 0.5]
            by = np.r_[prof - 0.5, -0.5, -0.5]
            cs, ct, r = kasa(bx, by)
            res = float(np.sqrt(np.mean((np.hypot(bx - cs, by - ct) - r) ** 2)) / r)
            x, y = to_xy(cs, ct)
            rows.append(dict(photo=photo, edge=edge, x=x, y=y, r=r, chord=s_b - s_a + 1, depth=depth,
                             width=s_hi - s_lo + 1, area=len(ts), res=res, mdk=float(V[comp].mean()),
                             inside=depth < DS,
                             bad=bool(Bd[:min(DS, depth + 2), max(0, s_lo - 2):s_hi + 3].any())))
    return rows


def all_components(photos, refresh=False):
    """Frame-touching components of all given photos (cached in runs/pp_border/border_cands.csv)."""
    meta = OUT / "border_cands_photos.json"
    if CACHE.exists() and meta.exists() and not refresh and set(photos) <= set(json.loads(meta.read_text())):
        d = pd.read_csv(CACHE, dtype={"photo": str})
        if list(d.columns) == COLUMNS:
            return d[d.photo.isin(photos)].reset_index(drop=True)
    rows = []
    for p in photos:
        rows += border_components(p)
    d = pd.DataFrame(rows, columns=COLUMNS)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE, index=False, float_format="%.4f")
    meta.write_text(json.dumps(sorted(photos)))
    return d


def score_components(c):
    """Physical gates + dark-and-round score -> detections (photo, x, y, r, score)."""
    gate = ((c.chord >= CHORD_MIN) & (c.depth >= DEPTH_MIN) & (c.r >= R_MIN) & (c.r <= R_MAX)
            & (c.width <= WIDTH_MAX * c.r) & c.inside.astype(bool) & ~c.bad.astype(bool))
    g = c[gate].copy()
    g["score"] = [ramp(m, 0.15, 0.45) * ramp(e, 0.25, 0.05) for m, e in zip(g.mdk, g.res)]
    return g[g.score > 0].reset_index(drop=True)


def border_dets(h):
    return score_components(all_components(sorted(set(h.train_photos) | set(h.test_photos))))


def train_preds(h):
    base = h.baseline_preds()
    bcd = crop_dist(h.train_paths, border_dets(h))
    deltas = sorted({k[1] for k in base})
    bp = {(b, d): predict(bcd, b, d) for b in BTHR for d in deltas}
    return {(t, d, b): (p | bp[(b, d)]).astype(np.uint8) for (t, d), p in base.items() for b in BTHR}


def test_pred(h, params):
    thr, delta, bthr = params
    p = predict(crop_dist(h.test_paths, h.det), thr, delta)
    return (p | predict(crop_dist(h.test_paths, border_dets(h)), bthr, delta)).astype(np.uint8)


def draw_train(h, paths, dets, out):
    """Overlay (train photos only) of crops changed by the step + the border circles."""
    train = set(h.train_photos)
    lab = {p: y for p, y in zip(h.train_paths, h.y)}
    tiles = []
    for rp in paths:
        pid, idx, _ = parse(rp)
        assert pid in train
        r, c = cell(idx)
        M = cv2.cvtColor(np.load(ROOT / f"data/mosaic_half/{pid}.npy"), cv2.COLOR_RGB2BGR)
        y0, x0 = min(max(r * T - 64, 0), ROWS * T - 256), min(max(c * T - 64, 0), COLS * T - 256)
        sub = M[y0:y0 + 256, x0:x0 + 256].copy()
        cv2.rectangle(sub, (c * T - x0, r * T - y0), (c * T - x0 + T - 1, r * T - y0 + T - 1),
                      (0, 0, 255) if lab[rp] else (200, 200, 200), 2)
        for _, d in dets[dets.photo == pid].iterrows():
            cv2.circle(sub, (int(d.x - x0), int(d.y - y0)), int(d.r), (0, 255, 255), 1)
        cv2.putText(sub, f"{Path(rp).stem} y={lab[rp]}", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(sub)
    if tiles:
        cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    h = H()
    base = h.baseline_preds()
    mine = train_preds(h)
    cmp_ = h.compare(base, mine)
    print("compare:", json.dumps(cmp_))
    params, f1 = h.full_fit(mine)
    print("full fit:", params, round(f1, 4))
    nb = h.nested_cv(mine)
    print("bthr chosen on inner folds:", dict(Counter(k[2] for k in nb["chosen"])))
    bparams, _ = h.full_fit(base)
    pb, pm, y = base[bparams], mine[params], h.y
    fixed = [h.train_paths[i] for i in np.nonzero((pb != y) & (pm == y))[0]]
    new = [h.train_paths[i] for i in np.nonzero((pb == y) & (pm != y))[0]]
    tp, fp, fn = (int(((pm == 1) & (y == 1)).sum()), int(((pm == 1) & (y == 0)).sum()),
                  int(((pm == 0) & (y == 1)).sum()))
    print(f"full-fit train TP{tp} FP{fp} FN{fn}; fixed vs baseline full fit: {fixed}; new errors: {new}")
    dets = border_dets(h)
    trd = dets[dets.photo.isin(h.train_photos)]
    print(f"border detections: train {len(trd)} (score>=bthr: {(trd.score >= params[2]).sum()}), "
          f"test {(~dets.photo.isin(h.train_photos)).sum()}")
    OUT.mkdir(parents=True, exist_ok=True)
    draw_train(h, fixed + new, trd[trd.score >= params[2]], OUT / "fixed_train.jpg")
    pt = test_pred(h, params)
    with open(OUT / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{p},{int(v)}\n" for p, v in zip(h.test_paths, pt))
    rep = dict(compare=cmp_, nested_cv_mine=dict(mean=nb["mean"], std=nb["std"], per_seed=nb["per_seed"]),
               full_fit_params=dict(thr=params[0], delta=params[1], bthr=params[2]), full_fit_train_f1=f1,
               tp=tp, fp=fp, fn=fn, fixed_errors=fixed, new_errors=new, test_positives=int(pt.sum()),
               baseline_full_fit=list(bparams),
               constants=dict(DS=DS, TAU=TAU, CHORD_MIN=CHORD_MIN, DEPTH_MIN=DEPTH_MIN, R_MIN=R_MIN,
                              R_MAX=R_MAX, WIDTH_MAX=WIDTH_MAX, BTHR=list(BTHR)))
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("test positives:", int(pt.sum()), "->", OUT / "submission.csv")


if __name__ == "__main__":
    main()

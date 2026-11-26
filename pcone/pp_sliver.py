"""Post-processing idea "sliver": decide borderline crops the way the labeller saw them.

Field premise (現場の前提): the person who labelled the data looked at ONE 256px crop at a time,
not at the whole wall photo. A crop whose square is only grazed by a P-cone circle is labelled 1
when a piece of the mark is actually visible inside those 256 px (a dark crescent of the recess
at the crop edge), and 0 when the circle only touches the crop on paper (fitted radius slightly
too large, bevel / halo, shading).  The geometric rule "dist(circle, square) <= r + delta" cannot
see that; the crop pixels can.

Rule (applied identically to every train / test crop, no labels used at test time):
  For every combined-detector circle with score >= thr near the crop:
    gap = dist(centre, crop square) - r      (half-res px, <= 0 means the circle touches it)
    - gap <= delta                   -> positive (baseline geometric rule)            [unchanged]
    - delta < gap <= G               -> positive only if the crop shows a visible sliver:
         evidence >= E, where evidence = number of full-res pixels of the 256px crop that
           * lie within the circle widened by MARGIN px (circle scaled x2 to full-res),
           * are at least DARK gray levels darker than the crop's own background
             (median of the crop pixels farther than R + FAR from the circle centre), and
           * belong to a dark connected region touching the crop border
             (a mark whose centre is outside the crop must enter through the border).
    - farther                        -> negative.
  Optional veto (V > 0): circles with -VB <= gap <= delta also need evidence >= V.
MARGIN / DARK / FAR are fixed (chosen once from train inspection); thr, delta, G, E are tuned
by photo-grouped nested CV in pcone/harness.py (the veto V was inspected on train and left off:
the two grazing positives 7078-149/150 show no dark pixels at all, so it can only trade errors).

  .venv/bin/python pcone/pp_sliver.py      -> prints CV comparison, writes runs/pp_sliver/
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from harness import H, ROOT  # noqa: E402
from mosaic import cell, parse  # noqa: E402

T = 128                      # half-res crop size
GAP_LO, GAP_HI = -12.0, 10.0  # gaps (half-res px) for which the crop's evidence is measured
MIN_SCORE = 0.05             # detections below this are never used for evidence
MARGIN, DARK, FAR, BLUR = 12.0, 30.0, 32.0, 2.0   # full-res px / gray levels
OUT = ROOT / "runs/pp_sliver"

_YY, _XX = np.mgrid[0:256, 0:256].astype(np.float32)
_BORDER = np.zeros((256, 256), bool)
_BORDER[:2, :] = _BORDER[-2:, :] = _BORDER[:, :2] = _BORDER[:, -2:] = True


def sliver_evidence(gray, cx, cy, R):
    """Visible dark pixels of a mark (circle cx,cy,R in full-res crop coords) inside the crop."""
    D = np.hypot(_XX - cx, _YY - cy)
    far = D > R + FAR
    bg = gray[far] if far.sum() > 4000 else gray[D > R + 8]
    if bg.size < 500:
        return 0
    med = float(np.median(bg))
    dark = ((med - gray) >= DARK) & (D <= R + MARGIN)
    lab, n = ndi.label(dark)
    if n == 0:
        return 0
    keep = np.unique(lab[_BORDER & (lab > 0)])
    keep = keep[keep > 0]
    return int(np.isin(lab, keep).sum()) if len(keep) else 0


def crop_table(h, paths):
    """Per crop: array (n_det, 3) of (score, gap, evidence); evidence=-1 where not measured."""
    det = h.det
    by_photo = {str(k): g[["x", "y", "r", "score"]].to_numpy(float) for k, g in det.groupby("photo")}
    out = []
    for rp in paths:
        pid, idx, _ = parse(rp)
        d = by_photo.get(pid)
        if d is None or len(d) == 0:
            out.append(np.zeros((0, 3))); continue
        r, c = cell(idx)
        x0, y0 = c * T, r * T
        dx = np.maximum(np.maximum(x0 - d[:, 0], 0), d[:, 0] - (x0 + T))
        dy = np.maximum(np.maximum(y0 - d[:, 1], 0), d[:, 1] - (y0 + T))
        gap = np.hypot(dx, dy) - d[:, 2]
        ev = np.full(len(d), -1.0)
        todo = np.where((gap >= GAP_LO) & (gap <= GAP_HI) & (d[:, 3] >= MIN_SCORE))[0]
        if len(todo):
            img = cv2.imread(str(ROOT / "data/sdnet" / rp), cv2.IMREAD_GRAYSCALE)
            g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), BLUR)
            for j in todo:
                ev[j] = sliver_evidence(g, (d[j, 0] - x0) * 2, (d[j, 1] - y0) * 2, d[j, 2] * 2)
        out.append(np.stack([d[:, 3], gap, ev], 1))
    return out


def _cached(h, split):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"evidence_{split}.npz"
    paths = h.train_paths if split == "train" else h.test_paths
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["paths"]) == list(paths):
            return list(z["tab"])
    tab = crop_table(h, paths)
    np.savez(f, paths=np.array(paths), tab=np.array(tab, dtype=object))
    return tab


def _rows(tab):
    """Flatten the per-crop tables into rows (crop index, score, gap, evidence)."""
    ci = np.concatenate([np.full(len(x), i) for i, x in enumerate(tab)]).astype(int)
    X = np.concatenate([x for x in tab if len(x)]) if any(len(x) for x in tab) else np.zeros((0, 3))
    keep = X[:, 1] <= 20  # circles farther than this never decide anything
    return len(tab), ci[keep], X[keep, 0], X[keep, 1], X[keep, 2]


def predict(rows, thr, delta, G, E, V=0.0, VB=8.0, rel=False):
    """G = upper gap of the rescue band (absolute, half-res px); rel=True -> band is
    (delta, delta + G] (first, discarded parametrisation, kept for the report)."""
    n, ci, s, g, e = rows
    ok = s >= thr
    geo = ok & (g <= delta)
    if V > 0:  # veto: grazing circles inside the geometric rule still need a visible sliver
        geo &= (g < -VB) | (e >= V)
    hi = delta + G if rel else G
    rescue = ok & (g > delta) & (g <= hi) & (e >= E)
    out = np.zeros(n, np.uint8)
    out[ci[geo | rescue]] = 1
    return out


G_GRID = (0.0, 2.0, 4.0)        # rescue band upper gap, ascending  (least change first)
E_GRID = (500.0, 350.0, 200.0)  # evidence threshold, descending   (least change first)
DELTAS = np.arange(-8, 9, 1.0)


def _grid(h):
    """Keys (thr, delta, G, E). G=-inf means 'no rescue' (= baseline rule). Keys are ordered
    so that ties in the inner-fold arg-max fall to the least change: plain geometric rule first,
    then the narrowest band and the strictest evidence threshold."""
    thrs = [float(t) for t in h.thr_grid()]
    keys = [(t, float(d), -np.inf, np.inf) for d in DELTAS for t in thrs]
    keys += [(t, float(d), G, E) for G in G_GRID for E in E_GRID for d in DELTAS for t in thrs]
    return keys


def train_preds(h, keys=None):
    rows = _rows(_cached(h, "train"))
    keys = _grid(h) if keys is None else keys
    return {k: predict(rows, *k) for k in keys}


def test_pred(h, params):
    return predict(_rows(_cached(h, "test")), *params).astype(int)


def variants(h, base):
    """Honest sensitivity checks (all with the same nested CV); none of them is the submission."""
    rows = _rows(_cached(h, "train"))
    thrs = [float(t) for t in h.thr_grid()]
    basek = [(t, float(d), -np.inf, np.inf) for d in DELTAS for t in thrs]
    out = {}

    def cmp(name, preds):
        c = h.compare(base, preds)
        out[name] = {k: c[k] for k in ("mine", "paired_diff_mean", "paired_diff_per_seed", "wins", "losses")}
        print(f"  {name:34s} diff {c['paired_diff_mean']:+.4f}  W{c['wins']}/L{c['losses']}  {c['paired_diff_per_seed']}")

    # v1 = first attempt: band (delta, delta + A], permissive-first tie order -> ties picked E=200.
    k1 = [(t, float(d), A, E) for A in (2.0, 4.0, 6.0) for E in (200.0, 350.0, 500.0) for d in DELTAS for t in thrs]
    cmp("v1_relative_band_permissive_ties", {**{k: predict(rows, *k) for k in basek},
                                             **{k: predict(rows, *k, rel=True) for k in k1}})
    # evidence grid shifted (400/300 instead of 500/350/200)
    k2 = [(t, float(d), G, E) for G in G_GRID for E in (400.0, 300.0) for d in DELTAS for t in thrs]
    cmp("E_grid_400_300", {k: predict(rows, *k) for k in basek + k2})
    # rescue only on top of a near-baseline geometric rule (delta >= -4)   [post-hoc, not adopted]
    k3 = [(t, float(d), G, E) for G in G_GRID for E in E_GRID for d in DELTAS if d >= -4 for t in thrs]
    cmp("rescue_only_delta_ge_-4(post-hoc)", {k: predict(rows, *k) for k in basek + k3})
    # symmetric band as first envisaged: evidence decides BOTH ways for gap in [-B, G]
    sym = {k: predict(rows, *k) for k in basek}
    for B in (4.0, 6.0, 8.0):
        for G in G_GRID:
            for E in (100.0, 200.0, 350.0):
                for t in thrs:
                    sym[(t, G, G, E, E, B)] = predict(rows, t, G, G, E, V=E, VB=B)
    cmp("symmetric_veto_and_rescue", sym)
    return out


def _errors(h, p):
    names = np.array([Path(x).stem for x in h.train_paths])
    return (sorted(names[(p == 1) & (h.y == 0)]), sorted(names[(p == 0) & (h.y == 1)]))


if __name__ == "__main__":
    h = H()
    base = h.baseline_preds()
    mine = train_preds(h)
    cmp_ = h.compare(base, mine)
    print("compare:", json.dumps(cmp_))
    nc = h.nested_cv(mine)
    print("nested CV mine:", round(nc["mean"], 4), "+-", round(nc["std"], 4), nc["per_seed"])
    from collections import Counter
    print("inner-fold choices (G,E):", Counter((k[2], k[3]) for k in nc["chosen"]).most_common())
    best, f1 = h.full_fit(mine)
    bbest, bf1 = h.full_fit(base)
    p = mine[best].astype(int)
    fp, fn = _errors(h, p)
    bfp, bfn = _errors(h, base[bbest])
    print("full fit:", best, "train F1", round(f1, 4), "| baseline", bbest, round(bf1, 4))
    print("  FP", fp, "\n  FN", fn)
    print("  fixed:", sorted(set(bfp + bfn) - set(fp + fn)), " new:", sorted(set(fp + fn) - set(bfp + bfn)))
    print("variants:")
    var = variants(h, base)
    pt = test_pred(h, best)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{rp},{v}\n" for rp, v in zip(h.test_paths, pt))
    rep = {"compare": cmp_, "mine_nested": {k: v for k, v in nc.items() if k != "chosen"},
           "full_fit_params": dict(zip(["thr", "delta", "G", "E"], map(float, best))),
           "full_fit_train_f1": f1, "baseline_full_fit": list(map(float, bbest)), "baseline_train_f1": bf1,
           "train_fp": fp, "train_fn": fn, "test_positives": int(pt.sum()),
           "fixed_vs_baseline": sorted(set(bfp + bfn) - set(fp + fn)),
           "new_vs_baseline": sorted(set(fp + fn) - set(bfp + bfn)),
           "variants": var,
           "fixed_constants": {"MARGIN": MARGIN, "DARK": DARK, "FAR": FAR, "BLUR": BLUR}}
    json.dump(rep, open(OUT / "report.json", "w"), indent=1, default=float)
    print("test positives", int(pt.sum()), "->", OUT / "submission.csv")

"""Label-free ensemble of the three P-cone (Pコン跡) detectors: LoG (detect_log.py),
Hough + polar ellipse (detect_hough.py) and segmentation + contour (detect_contour.py).

Run:  .venv/bin/python pcone/detect_combined.py      (no arguments, ~2 s incl. overlays)

One fixed rule, applied identically to every photo and crop (no labels, no per-photo edits):

  1. Normalise each method's score so that its own default threshold maps to 0.5, capped at 0.9:
         n_m = min(N_CAP, 0.5 * score_m / TH_m),   TH = {log: 0.45, hough: 0.50, contour: 0.50}.
     The cap means no single method can carry a crop on its own, however confident it is.
     Detections with score < FLOOR (0.10) are ignored.
  2. Cluster detections of different methods in the same photo (union-find): two detections are
     the same mark when their centres are within max(10 px, 0.6 * larger radius) (half-res px)
     and their radii differ by at most 2.5x (rim vs inner-shadow vs halo fits stay linked; a
     LoG circle swallowing a whole pipe flange does not).
     Within a cluster each method keeps its best-scoring detection.
  3. Evidence that a crop contains the cluster's mark = sum over the cluster's methods of
         n_m * w(ov_m),   w(ov) = clip((ov - OV_LO) / (OV_HI - OV_LO), 0, 1),  OV_LO=10, OV_HI=60,
     where ov_m is the area (half-res px, pixel centres) of method m's own shape (LoG circle,
     Hough ellipse from detections_full.csv, contour circle) inside the crop's 128x128 half-res
     square.  Crop evidence = max over clusters.
     Reading: a crop needs "two methods at their own threshold" worth of evidence, i.e. two (or
     three) methods agree that a mark is there AND that their shapes reach into this crop by more
     than a rim sliver.  A method scoring below threshold still adds its support.
  4. score = min(1, evidence / 2) in [0, 1]; default decision threshold THRESH = 0.5, i.e.
     evidence >= 1 = two methods exactly at their own threshold, both fully covering the crop
     (one confident method at the 0.9 cap needs >= 0.1 of support from a second one).
     votes = number of methods in that cluster with score >= own threshold whose shape covers
     >= (OV_LO + OV_HI) / 2 px of the crop.
     overlap_px = evidence-weighted mean overlap (half-res px) of the methods' shapes.

Why (from label-free inspection of disagreement cases, data/detect/combined/disagree/):
  - All clusters with 3 votes and (all but one pipe flange) with 2 votes looked like real marks.
  - Single-method detections are mostly distractors: Hough-only = air voids, spalls, pipes,
    bug holes; LoG-only = pores, stains, border slivers, pipes, faint ghost patches;
    contour-only = spalls. The real ones among them nearly always have a sub-threshold score
    from a second method (e.g. Hough 1.0 + contour 0.48), which the evidence sum accepts.
  - Crop disagreement between methods is mostly geometry: rim slivers of a few px, and crops
    reached only by one method's oversized fit (contour halo, LoG 1.3x widening) or missed by an
    inner-shadow Hough fit.  Requiring two methods' shapes with more than a sliver of overlap
    removes the one-method crops and down-weights slivers smoothly.

Outputs (data/detect/combined/):
  crops.csv       rel_path,score,overlap_px,votes   (every W/*/ jpg on disk, 18,138 rows)
  viz/<photo>.jpg quarter-res overlays: positive crops shaded red, 0.25-0.5 outlined orange,
                  member shapes of clusters with evidence >= 0.5 (LoG red, Hough green, contour blue)
  detections.csv  photo,x,y,r,score,evidence,votes,n_log,n_hough,n_contour  (one row per cluster;
                  x,y = n-weighted mean centre, r = median radius; a cluster can only make crops
                  positive when evidence >= 1)
"""
import csv
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # project root (relative to this file)
DET = ROOT / "data/detect"
SDNET_W = ROOT / "data/sdnet/W"
MOSAIC = ROOT / "data/mosaic_half"
OUT = DET / "combined"

COLS, ROWS, T = 18, 14, 128
NAME = re.compile(r"^(\d+)-(\d+)(?:_(\d+))?$")

METHODS = ("log", "hough", "contour")
TH = {"log": 0.45, "hough": 0.50, "contour": 0.50}   # each method's own default threshold
N_CAP = 0.90          # max normalised score of one method (no method carries a crop alone)
FLOOR = 0.10          # ignore detections below this raw score
LINK_MIN, LINK_FRAC = 10.0, 0.6   # cluster link: dist <= max(LINK_MIN, LINK_FRAC * max r)
LINK_RATIO = 2.5                  # ... and max r / min r <= LINK_RATIO
OV_LO, OV_HI = 10, 60   # coverage ramp (half-res px^2 inside the crop; x4 for full-res px)
EVID_BAR = 1.0        # evidence needed: two methods at their own threshold
THRESH = 0.5          # decision threshold on score = min(1, evidence / (2 * EVID_BAR))


def to_score(ev):
    return min(1.0, ev / (2 * EVID_BAR))


def norm(m, s):
    return min(N_CAP, 0.5 * s / TH[m])


def cover(ov):
    return min(1.0, max(0.0, (ov - OV_LO) / (OV_HI - OV_LO)))


def load():
    rows = []
    for m in METHODS:
        d = pd.read_csv(DET / m / "detections.csv", dtype={"photo": str})
        d = d[d.score >= FLOOR].copy()
        d["m"] = m
        d["a"], d["b"], d["phi"] = d.r, d.r, 0.0
        if m == "hough" and (DET / "hough/detections_full.csv").exists():
            f = pd.read_csv(DET / "hough/detections_full.csv", dtype={"photo": str})
            for i, r in d.iterrows():
                q = f[(f.photo == r.photo)]
                if not len(q):
                    continue
                dist = np.hypot(q.x - r.x, q.y - r.y) + np.abs(q.score - r.score) * 100
                j = dist.idxmin()
                if dist[j] < 1.0:
                    d.loc[i, ["a", "b", "phi"]] = q.loc[j, ["a", "b", "phi"]].values
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    D["n"] = [norm(m, s) for m, s in zip(D.m, D.score)]
    return D


def cluster(D):
    par = list(range(len(D)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for _, g in D.groupby("photo"):
        idx = g.index.to_numpy()
        X = g[["x", "y", "r"]].to_numpy(float)
        mm = g.m.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if mm[i] == mm[j]:
                    continue
                rmax, rmin = max(X[i, 2], X[j, 2]), max(min(X[i, 2], X[j, 2]), 1e-3)
                if (math.hypot(X[i, 0] - X[j, 0], X[i, 1] - X[j, 1]) <= max(LINK_MIN, LINK_FRAC * rmax)
                        and rmax / rmin <= LINK_RATIO):
                    par[find(idx[i])] = find(idx[j])
    D = D.copy()
    D["cl"] = [find(i) for i in range(len(D))]
    # each method keeps its best detection per cluster
    D = D.sort_values("score", ascending=False).drop_duplicates(["cl", "m"]).sort_index()
    return D


def shape_cell_overlap(d):
    """{(row, col): area} of the detection's ellipse/disk inside each 128 px cell (pixel centres)."""
    x, y, a, b, phi = d["x"], d["y"], max(d["a"], 0.5), max(d["b"], 0.5), d["phi"]
    R = max(a, b)
    x0, x1 = max(0, int(math.floor(x - R))), min(COLS * T, int(math.ceil(x + R)) + 1)
    y0, y1 = max(0, int(math.floor(y - R))), min(ROWS * T, int(math.ceil(y + R)) + 1)
    if x0 >= x1 or y0 >= y1:
        return {}
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = xx - x, yy - y
    c, s = math.cos(phi), math.sin(phi)
    u, v = (dx * c + dy * s) / a, (-dx * s + dy * c) / b
    inside = (u * u + v * v) <= 1.0
    out = {}
    ys, xs = np.nonzero(inside)
    if not len(ys):
        return out
    cells = ((ys + y0) // T) * COLS + (xs + x0) // T
    for k, cnt in zip(*np.unique(cells, return_counts=True)):
        out[divmod(int(k), COLS)] = int(cnt)
    return out


def draw(photo, cells, members, path):
    """Quarter-res overlay: positive crops shaded red, 0.25 <= score < 0.5 outlined orange;
    method shapes of clusters with evidence >= 0.5 (LoG red, Hough green, contour blue)."""
    img = cv2.cvtColor(np.load(MOSAIC / f"{photo}.npy"), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ov = img.copy()
    for (r, c), (ev, _, _) in cells.items():
        sc = to_score(ev)
        p0, p1 = (c * T // 2, r * T // 2), ((c + 1) * T // 2 - 1, (r + 1) * T // 2 - 1)
        if sc >= THRESH:
            cv2.rectangle(ov, p0, p1, (0, 0, 255), -1)
        elif sc >= 0.25:
            cv2.rectangle(img, p0, p1, (0, 165, 255), 2)
    img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
    col = {"log": (0, 0, 255), "hough": (0, 200, 0), "contour": (255, 120, 0)}
    for _, d in members.iterrows():
        cv2.ellipse(img, (int(d.x / 2), int(d.y / 2)), (max(1, int(d.a / 2)), max(1, int(d.b / 2))),
                    float(np.rad2deg(d.phi)), 0, 360, col[d.m], 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    D = cluster(load())

    # ---- per-cluster summary + per-crop evidence
    cell_best = {}      # (photo, row, col) -> (evidence, overlap_px, votes)
    det_rows = []
    for cl, g in D.groupby("cl"):
        photo = g.photo.iloc[0]
        ev_cluster = float(g.n.sum())
        votes = int(sum(s >= TH[m] for m, s in zip(g.m, g.score)))
        w = g.n.to_numpy()
        det_rows.append(dict(photo=photo, x=float(np.average(g.x, weights=w)), y=float(np.average(g.y, weights=w)),
                             r=float(np.median(g.r)), score=to_score(ev_cluster), evidence=ev_cluster, votes=votes,
                             **{f"n_{m}": float(g.n[g.m == m].sum()) for m in METHODS}))
        if ev_cluster < EVID_BAR - 1e-9:
            continue            # cannot reach the bar on any crop
        per_cell = {}
        for _, d in g.iterrows():
            for cell, ov in shape_cell_overlap(d).items():
                w = d.n * cover(ov)
                if w > 0:
                    per_cell.setdefault(cell, []).append((w, ov, d.score >= TH[d.m] and ov >= (OV_LO + OV_HI) / 2))
        for cell, lst in per_cell.items():
            ev = sum(w for w, _, _ in lst)
            ov = sum(w * o for w, o, _ in lst) / max(ev, 1e-9)
            vt = sum(1 for _, _, v in lst if v)
            key = (photo,) + cell
            if key not in cell_best or ev > cell_best[key][0]:
                cell_best[key] = (ev, ov, vt)

    (OUT / "viz").mkdir(exist_ok=True)
    big = {cl for cl, g in D.groupby("cl") if g.n.sum() >= 0.5}
    for photo in sorted(p.stem for p in MOSAIC.glob("*.npy") if not p.stem.endswith("_filled")):
        cells = {k[1:]: v for k, v in cell_best.items() if k[0] == photo}
        draw(photo, cells, D[(D.photo == photo) & D.cl.isin(big)], OUT / "viz" / f"{photo}.jpg")

    pd.DataFrame(det_rows).sort_values(["photo", "score"], ascending=[True, False]).to_csv(
        OUT / "detections.csv", index=False, float_format="%.4f")

    rows = []
    for sub in sorted(p.name for p in SDNET_W.iterdir() if p.is_dir()):
        for fn in sorted(p.name for p in (SDNET_W / sub).glob("*.jpg")):
            m = NAME.match(Path(fn).stem)
            ev, ov, vt = 0.0, 0.0, 0
            if m:
                idx = int(m.group(2))
                if 1 <= idx <= COLS * ROWS:
                    r, c = divmod(idx - 1, COLS)
                    ev, ov, vt = cell_best.get((m.group(1), r, c), (0.0, 0.0, 0))
            rows.append((f"W/{sub}/{fn}", round(to_score(ev), 4), ov, vt))
    with open(OUT / "crops.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rel_path", "score", "overlap_px", "votes"])
        for rp, s, ov, vt in rows:
            wr.writerow([rp, f"{s:.4f}", f"{ov:.0f}", vt])

    n_pos = sum(1 for _, s, _, _ in rows if s >= THRESH)
    dets = pd.DataFrame(det_rows)
    n_det = int((dets.evidence >= EVID_BAR - 1e-9).sum())
    photos_with = dets[dets.evidence >= EVID_BAR - 1e-9].photo.nunique()
    print(f"crops={len(rows)} positive={n_pos} ({100 * n_pos / len(rows):.2f}%) clusters_pos={n_det} "
          f"photos_with_pos={photos_with}/72 time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

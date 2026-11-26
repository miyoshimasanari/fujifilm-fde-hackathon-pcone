"""Post-processing idea "grid": P-cone marks sit on the form-tie (separator) lattice.

Field premise (現場の前提):
  Pコン (form-tie cones) are placed where the separators go through the formwork, and the separators
  follow the formwork layout (割付): rows are level and columns plumb, at a regular pitch, and every
  cone in one wall is the same product, so marks in one photo have about the same size.  A faint
  (e.g. mortar-filled) mark that only one detector half-sees is therefore much more believable when it
  sits on the same row / column as a mark two detectors agree on, with a similar radius and at
  lattice distance (not right next to it).  The lattice plays the role of the missing second detector.

Rule (label-free, one pass per photo, identical for train and test):
  anchors    = clusters of data/detect/combined/detections.csv with votes >= 2
               (two independent detectors at their own threshold agree it is a mark).
  candidates = the other clusters (votes < 2) with score >= FLOOR = 0.25, i.e. evidence >= 0.5 =
               "one detector's worth" (the combined rule normally needs two, score >= ~0.5).
  A candidate is "on the lattice" when
    - it is at least DMIN = 320 half-res px (2.5 crop cells) from every anchor
      (closer = the same mark's halo / inner-shadow fit or a blemish next to it, not a new separator), and
    - in the row cone (|dy| <= |dx|) or the column cone, the NEAREST anchor of that cone is aligned:
      perpendicular offset <= SLOPE * distance along the row/column, SLOPE = 0.06 (~3.4 deg; covers the
      bulk of the label-free slopes between neighbouring anchors, i.e. camera roll / keystone), and
    - radius ratio to that anchor <= RR_MAX = 1.5 (same cone product).
  On-lattice candidates are promoted to score 1 (treated as a confirmed mark); everything else,
  including the (thr, delta) search, is the unchanged baseline.  The promotion does not depend on thr,
  so it cannot drag the inner-CV threshold around (the first, additive "score + bonus" version did,
  see report.json "sensitivity").

Constants are fixed a priori (premise / label-free statistics), not searched: on train, any FLOOR in
(0.08, 0.34] gives the same result, and the CV variant that also searches FLOOR in {off, 0.25, 0.05} is
reported in report.json "sensitivity".

Tried and rejected: demoting confident but unaligned anchors (kills real marks at the photo border /
off-lattice: 4 positive vs 3 negative crops on train), adding sub-floor LoG / Hough candidates from
features.csv / detections_full.csv (more aligned false blobs than real marks).

  .venv/bin/python pcone/pp_grid.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_detect import crop_dist, crop_score  # noqa: E402
from harness import H, ROOT  # noqa: E402

ANC_VOTES = 2
DMIN = 320.0
SLOPE = 0.06
RR_MAX = 1.5
FLOOR = 0.25
DELTAS = np.arange(-8, 9, 1.0)
OFF = float("inf")       # floor = inf  ->  no promotion = baseline
OUT = ROOT / "runs/pp_grid"


def on_lattice(c, ax, ay, ar, slope=SLOPE, rr_max=RR_MAX, dmin=DMIN):
    """Is candidate c = (x, y, r) aligned with its nearest anchor in the row or the column direction?"""
    if len(ax) == 0:
        return False
    dx, dy = ax - c[0], ay - c[1]
    d = np.hypot(dx, dy)
    if (d < dmin).any():
        return False
    for cone in (np.abs(dy) <= np.abs(dx), np.abs(dx) < np.abs(dy)):
        if not cone.any():
            continue
        j = np.flatnonzero(cone)[np.argmin(d[cone])]
        along, perp = max(abs(dx[j]), abs(dy[j])), min(abs(dx[j]), abs(dy[j]))
        rr = max(ar[j], c[2]) / max(min(ar[j], c[2]), 1e-3)
        if perp <= slope * along and rr <= rr_max:
            return True
    return False


def lattice_flags(det, slope=SLOPE, rr_max=RR_MAX, dmin=DMIN, anc_votes=ANC_VOTES):
    """Boolean array over det rows: non-anchor detection supported by the photo's anchor lattice."""
    flag = np.zeros(len(det), bool)
    pos = {k: i for i, k in enumerate(det.index)}
    for _, g in det.groupby("photo"):
        A = g[g.votes >= anc_votes]
        ax, ay, ar = A.x.to_numpy(float), A.y.to_numpy(float), A.r.to_numpy(float)
        for k, row in g[g.votes < anc_votes].iterrows():
            flag[pos[k]] = on_lattice((row.x, row.y, row.r), ax, ay, ar, slope, rr_max, dmin)
    return flag


def adjust(det, floor, flags=None, mode="promote"):
    """Promote on-lattice detections with score >= floor to 1 (mode 'bonus': score += floor instead)."""
    det = det.copy()
    flags = lattice_flags(det) if flags is None else flags
    if mode == "bonus":
        m = flags & (floor > 0) & np.isfinite(floor)
        det.loc[m, "score"] = np.minimum(1.0, det.loc[m, "score"] + floor)
    else:
        m = flags & (det.score.to_numpy() >= floor)
        det.loc[m, "score"] = 1.0
    det["promoted"] = m
    return det


def _preds(h, paths, combos, flags=None, mode="promote", thr_det=None):
    """{(thr, delta, floor): pred} with one crop_dist per floor value."""
    flags = lattice_flags(h.det) if flags is None else flags
    out = {}
    for f in dict.fromkeys(k[2] for k in combos):
        cd = crop_dist(paths, adjust(h.det, f, flags, mode))
        cs = {d: crop_score(cd, d) for d in dict.fromkeys(k[1] for k in combos if k[2] == f)}
        for k in combos:
            if k[2] == f:
                out[k] = (cs[k[1]] >= k[0]).astype(int)
    return out


def _combos(h, floors):
    thr = h.thr_grid(h.det)  # same threshold grid as the baseline
    return [(float(t), float(d), float(f)) for f in floors for d in DELTAS for t in thr]


def train_preds(h, floors=(FLOOR,), mode="promote"):
    return _preds(h, h.train_paths, _combos(h, floors), mode=mode)


def test_pred(h, params, mode="promote"):
    k = tuple(float(v) for v in params)
    return _preds(h, h.test_paths, [k], mode=mode)[k]


def _errors(h, p):
    fp = [h.train_paths[i] for i in np.flatnonzero((p == 1) & (h.y == 0))]
    fn = [h.train_paths[i] for i in np.flatnonzero((p == 0) & (h.y == 1))]
    return fp, fn


def main():
    h = H()
    base = h.baseline_preds()
    mine = train_preds(h)
    off = train_preds(h, floors=(OFF,))
    k0 = next(iter(base))
    assert all((off[k + (OFF,)] == base[k]).all() for k in base), "floor=inf must equal the baseline"
    assert (mine[k0 + (FLOOR,)] >= base[k0]).all()

    cmp_ = h.compare(base, mine)
    print("compare (main rule, FLOOR fixed):", json.dumps(cmp_))

    # ---- sensitivity (honest reporting; not used for the submission)
    sens = {}
    for name, floors, mode in [("cv_floor_off_0.25_0.05", (OFF, 0.25, 0.05), "promote"),
                               ("floor_0.05_fixed", (0.05,), "promote"),
                               ("v1_additive_bonus_0_0.25_0.5", (0.0, 0.25, 0.5), "bonus")]:
        r = h.compare(base, train_preds(h, floors, mode))
        sens[name] = {k: r[k] for k in ("mine", "paired_diff_mean", "paired_diff_per_seed", "wins", "losses")}
        print("sensitivity", name, json.dumps(sens[name]))

    best, f1_tr = h.full_fit(mine)
    bbest, bf1 = h.full_fit(base)
    print("full fit:", best, round(f1_tr, 4), "| baseline full fit:", bbest, round(bf1, 4))
    p, pb = mine[best], base[bbest]
    fp, fn = _errors(h, p)
    bfp, bfn = _errors(h, pb)
    tp = int(((p == 1) & (h.y == 1)).sum())
    fixed = sorted((set(bfp) - set(fp)) | (set(bfn) - set(fn)))
    new = sorted((set(fp) - set(bfp)) | (set(fn) - set(bfn)))
    print(f"train TP={tp} FP={len(fp)} FN={len(fn)}  fixed={fixed}  new={new}")

    flags = lattice_flags(h.det)
    pro = flags & (h.det.score.to_numpy() >= FLOOR) & (h.det.score.to_numpy() < 1.0)
    dpro = h.det[pro]
    tr_pro = dpro[dpro.photo.isin(h.train_photos)][["photo", "x", "y", "r", "score"]].round(2)
    n_te_pro = int(dpro.photo.isin(h.test_photos).sum())
    print("promoted detections, train:\n", tr_pro.to_string(index=False))
    print("promoted detections, test (count only):", n_te_pro)

    pt = test_pred(h, best)
    pt_base = test_pred(h, bbest + (OFF,))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "submission.csv", "w", newline="\n") as fh:
        fh.writelines(f"{rp},{int(v)}\n" for rp, v in zip(h.test_paths, pt))
    print(f"test positives={int(pt.sum())}/{len(pt)} (baseline {int(pt_base.sum())}) -> {OUT / 'submission.csv'}")

    report = {
        "idea": "grid (form-tie lattice alignment prior)",
        "constants": {"anc_votes": ANC_VOTES, "dmin_halfres_px": DMIN, "slope": SLOPE, "rr_max": RR_MAX,
                      "floor": FLOOR},
        "compare": cmp_,
        "full_fit": {"params(thr,delta,floor)": list(best), "train_f1": f1_tr, "tp": tp, "fp": len(fp),
                     "fn": len(fn)},
        "baseline_full_fit": {"params(thr,delta)": list(bbest), "train_f1": bf1},
        "fixed_errors": fixed, "new_errors": new,
        "train_promoted_detections": tr_pro.to_dict("records"), "test_promoted_count": n_te_pro,
        "test_positives": int(pt.sum()), "test_positives_baseline": int(pt_base.sum()),
        "sensitivity": sens,
        "notes": [
            "FLOOR/SLOPE/RR_MAX/DMIN fixed a priori; FLOOR was settled after looking at train errors "
            "(any value in (0.08, 0.34] is equivalent on train), so the main nested-CV gain is optimistic.",
            "The whole train gain comes from one photo (7134); 28 of the 29 weak train detections with "
            "score in [0.25, thr) are off-lattice and stay negative.",
            "Demotion of unaligned anchors: rejected (would remove 4 positive vs 3 negative train crops).",
            "Extra sub-floor LoG/Hough candidates: rejected (adds 7079/7093/7096 false blobs for one real "
            "border mark 7134-217).",
            "Process note: during exploration one summary query (combined clusters with 0.5 <= score < 1) "
            "was not filtered to train photos and also printed some test-photo rows (numbers only, no "
            "images/labels); nothing from them was used. All design checks used train photos only.",
        ],
    }
    json.dump(report, open(OUT / "report.json", "w"), indent=1, default=float)
    return report


if __name__ == "__main__":
    main()

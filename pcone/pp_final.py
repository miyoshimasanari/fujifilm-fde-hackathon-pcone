"""Final pipeline: compose the field-premise post-processing steps that helped on their own.

Steps, in field order (every step is label-free and identical for train and test photos):
  1. grid   (pp_grid)   型枠の割付: a weak one-detector mark (score >= 0.25) that sits on the same
                        row / column as a two-detector anchor, at lattice distance and with a similar
                        radius, is promoted to score 1.  No searched parameter (FLOOR etc. fixed).
  2. border (pp_border) 撮影の枠: dark discs cut by the photo frame are added as extra circles; a crop
                        is positive when such a circle with score >= bthr touches it (same delta).
  -  size   (pp_size)   NOT used: it lost on 5/5 seeds on its own (real marks in one photo differ in
                        apparent size by up to x2: perspective, mortar-filled vs open recess).  The
                        ablation "border+grid+size" below re-checks this inside the composition.
  3. sliver (pp_sliver) ラベル付けは256pxの1枚: a crop just outside a circle (delta < gap <= G) is
                        rescued when the crop itself shows >= E dark pixels of the mark entering
                        through its border.  Applied to the (grid-promoted) combined detections.

  pred(crop) = [ max(score of circles with gap <= delta,
                     score of circles with delta < gap <= G and sliver evidence >= E) >= thr ]
               OR [ border circle with score >= bthr and gap <= delta ]

Joint parameters, ALL chosen together on the inner folds of h.nested_cv (photo-grouped, 5 seeds):
  thr (baseline grid, 60) x delta (-8..8) x bthr (pp_border grid, 8 incl. off)
  x sliver (G, E) (pp_sliver grid: off + {0,2,4} x {500,350,200}).
  Each module keeps the grid and the tie order it was validated with (key order = arg-max tie order):
  sliver: off first, then narrowest band / strictest evidence; border: bthr ascending, off last.
  A smaller sliver grid (off + the diagonal (0,500), (2,350), (4,200)) is reported as a sensitivity.

Selection rule (fixed before running): candidates are the compositions B+G+S and B+G.  A candidate
qualifies when it helps vs the baseline AND vs the best single idea (border) in h.compare
(paired_diff_mean > 0 and at most 1 losing seed in each).  The final pipeline is the qualifying
candidate with the highest nested-CV mean (ties -> fewer steps); if none qualifies, border alone.

  .venv/bin/python pcone/pp_final.py
  -> runs/pp_final/report.json, submission/submission_v2.csv (path,label; test.txt order; no header)
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import pp_border  # noqa: E402
import pp_grid  # noqa: E402
import pp_size  # noqa: E402
import pp_sliver  # noqa: E402
from eval_detect import crop_dist  # noqa: E402
from harness import H, ROOT  # noqa: E402

OUT = ROOT / "runs/pp_final"
SUB = ROOT / "submission/submission_v2.csv"
V1 = ROOT / "submission/submission_detcombined.csv"
DELTAS = np.arange(-8, 9, 1.0)
B_OFF = 9.0
BTHR = tuple(pp_border.BTHR)                     # (0.2 .. 0.8, 9.0 = off): ascending, off last
S_OFF = (-np.inf, np.inf)
SLIVER = (S_OFF,) + tuple((G, E) for G in pp_sliver.G_GRID for E in pp_sliver.E_GRID)
SLIVER_SMALL = (S_OFF, (0.0, 500.0), (2.0, 350.0), (4.0, 200.0))
FLOOR = pp_grid.FLOOR


def _flat(cd):
    ci = np.concatenate([np.full(len(x), i) for i, x in enumerate(cd)]).astype(int)
    X = np.concatenate([x for x in cd if len(x)]) if any(len(x) for x in cd) else np.zeros((0, 2))
    return ci, X


class Tables:
    """Per-split flat tables: combined circles (score, gap, sliver evidence) and border circles."""

    def __init__(self, h, split, grid=True, size=None, sliver=True):
        paths = h.train_paths if split == "train" else h.test_paths
        det = pp_grid.adjust(h.det, FLOOR) if grid else h.det          # step 1 (same row order)
        if size is not None:                                            # ablation only
            det = pp_size.size_prior(det, *size)
        cd = crop_dist(paths, det)
        self.n = len(paths)
        self.ci, X = _flat(cd)
        self.s, self.g = X[:, 0], X[:, 1]
        self.e = np.full(len(self.s), -1.0)
        if sliver:
            tab = pp_sliver._cached(h, split)                           # evidence per (crop, circle)
            assert len(tab) == len(cd)
            for a, b in zip(cd, tab):
                assert len(a) == len(b) and np.allclose(a[:, 1], b[:, 1]), "circle order mismatch"
            self.e = np.concatenate([b[:, 2] for b in tab if len(b)])
        bci, B = _flat(crop_dist(paths, pp_border.border_dets(h)))     # step 2
        self.bci, self.bs, self.bg = bci, B[:, 0], B[:, 1]

    def _max(self, ci, s, m):
        out = np.full(self.n, -np.inf)
        np.maximum.at(out, ci[m], s[m])
        return out

    def geo(self, d):
        return self._max(self.ci, self.s, self.g <= d)

    def rescue(self, d, G, E):                                          # step 3
        if not np.isfinite(G):
            return np.full(self.n, -np.inf)
        return self._max(self.ci, self.s, (self.g > d) & (self.g <= G) & (self.e >= E))

    def border(self, d):
        return self._max(self.bci, self.bs, self.bg <= d)


def build(T, thrs, bthrs=(B_OFF,), slivers=(S_OFF,), deltas=DELTAS, tag=()):
    """{(thr, delta, bthr, G, E, *tag): uint8 pred}; insertion order = inner-fold tie order."""
    K = len(slivers) * len(deltas) * len(thrs) * len(bthrs)
    P = np.empty((K, T.n), np.uint8)
    keys, i = [], 0
    for G, E in slivers:
        for d in deltas:
            sc = np.maximum(T.geo(d), T.rescue(d, G, E))
            bc = T.border(d)
            for t in thrs:
                m = sc >= t
                for b in bthrs:
                    P[i] = m | (bc >= b)
                    keys.append((float(t), float(d), float(b), float(G), float(E)) + tuple(tag))
                    i += 1
    return {k: P[j] for j, k in enumerate(keys)}


def _num(v):
    """JSON-safe number: +-inf (step switched off) -> the string 'off'."""
    return float(v) if np.isfinite(v) else "off"


def counts(y, p):
    return dict(tp=int(((p == 1) & (y == 1)).sum()), fp=int(((p == 1) & (y == 0)).sum()),
                fn=int(((p == 0) & (y == 1)).sum()))


def stems(h, idx):
    return [Path(h.train_paths[i]).stem for i in idx]


def main():
    t0 = time.time()
    h = H()
    y = h.y
    thrs = [float(t) for t in h.thr_grid()]
    base = h.baseline_preds()
    nb = h.nested_cv(base)
    print("baseline", round(nb["mean"], 4), nb["per_seed"], flush=True)
    T0 = Tables(h, "train", grid=False)
    TG = Tables(h, "train", grid=True)
    rep = {"baseline": {k: nb[k] for k in ("mean", "std", "per_seed")}}

    # ---- single ideas, rebuilt here; must reproduce the modules' own nested CV exactly
    singles = {"border": build(T0, thrs, BTHR), "grid": build(TG, thrs),
               "sliver": build(T0, thrs, slivers=SLIVER)}
    module = {"border": pp_border, "grid": pp_grid, "sliver": pp_sliver}
    rep["singles"] = {}
    for name, p in singles.items():
        c = h.compare(base, p)
        own = h.nested_cv(module[name].train_preds(h))["per_seed"]
        assert own == h.nested_cv(p)["per_seed"], f"{name}: rebuild differs from module"
        rep["singles"][name] = {k: c[k] for k in ("mine", "mine_std", "paired_diff_mean",
                                                  "paired_diff_per_seed", "wins", "losses")}
        print("single", name, json.dumps(rep["singles"][name]), flush=True)
    best_single = max(rep["singles"], key=lambda k: rep["singles"][k]["mine"])
    sp = singles[best_single]
    del singles

    # ---- compositions (joint nested CV) + ablations
    variants = [
        ("B+G+S", lambda: build(TG, thrs, BTHR, SLIVER), True),
        ("B+G", lambda: build(TG, thrs, BTHR), True),
        ("B+G+S_small_sliver_grid", lambda: build(TG, thrs, BTHR, SLIVER_SMALL), False),
        ("B+S", lambda: build(T0, thrs, BTHR, SLIVER), False),
        ("G+S", lambda: build(TG, thrs, slivers=SLIVER), False),
    ]
    rep["compositions"] = {}
    for name, fn, candidate in variants:
        p = fn()
        cb, cs = h.compare(base, p), h.compare(sp, p)
        nc = h.nested_cv(p)
        r = {"n_keys": len(p), "candidate": candidate,
             "nested_cv": {k: nc[k] for k in ("mean", "std", "per_seed")},
             "vs_baseline": {k: cb[k] for k in ("paired_diff_mean", "paired_diff_per_seed", "wins", "losses")},
             f"vs_best_single_{best_single}": {k: cs[k] for k in ("paired_diff_mean", "paired_diff_per_seed",
                                                                    "wins", "losses")},
             "inner_choices_bthr_G_E": {str(k): v for k, v in
                                        Counter((c[2], c[3], c[4]) for c in nc["chosen"]).most_common()},
             "inner_choices_thr_delta": {str(k): v for k, v in
                                         Counter((round(c[0], 4), c[1]) for c in nc["chosen"]).most_common()}}
        rep["compositions"][name] = r
        print(name, json.dumps({k: r[k] for k in ("n_keys", "nested_cv", "vs_baseline")}),
              "| vs", best_single, json.dumps(r[f"vs_best_single_{best_single}"]),
              f"| {time.time() - t0:.0f}s", flush=True)
        del p

    # ablation: size prior (not used) inside the composition: grid -> size -> border (no sliver:
    # the sliver evidence was measured with the unsnapped radii)
    sz = {}
    for a in pp_size.GRID_A:
        for k in pp_size.GRID_K:
            sz.update(build(Tables(h, "train", grid=True, size=(a, k), sliver=False), thrs, BTHR,
                            tag=(float(a), float(k))))
    bg = build(TG, thrs, BTHR)
    c = h.compare(bg, sz)
    rep["ablation_size_on_B+G"] = {"vs_B+G": {k: c[k] for k in ("mine", "paired_diff_mean",
                                                               "paired_diff_per_seed", "wins", "losses")},
                                   "inner_choices_a_k": {str(k): v for k, v in Counter(
                                       (q[5], q[6]) for q in h.nested_cv(sz)["chosen"]).most_common()}}
    print("ablation size on B+G", json.dumps(rep["ablation_size_on_B+G"]), flush=True)
    del sz, bg

    # step-wise paired differences between compositions (same arithmetic as h.compare: per-seed
    # nested-CV F1 difference; nested_cv is deterministic, so this equals h.compare(a, b))
    def pdiff(a, b):
        d = np.array(b["nested_cv"]["per_seed"]) - np.array(a["nested_cv"]["per_seed"])
        return {"paired_diff_mean": float(d.mean()), "paired_diff_per_seed": d.round(4).tolist(),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    C = rep["compositions"]
    rep["stepwise"] = {"sliver_on_top_of_B+G (B+G+S vs B+G)": pdiff(C["B+G"], C["B+G+S"]),
                       "grid_on_top_of_B+S (B+G+S vs B+S)": pdiff(C["B+S"], C["B+G+S"])}
    print("stepwise", json.dumps(rep["stepwise"]), flush=True)

    # ---- pre-declared selection
    def helps(d):
        return d["paired_diff_mean"] > 0 and d["losses"] <= 1
    order = {"B+G+S": 3, "B+G": 2}
    q = [n for n, r in rep["compositions"].items() if r["candidate"] and helps(r["vs_baseline"])
         and helps(r[f"vs_best_single_{best_single}"])]
    if q:
        final = max(q, key=lambda n: (round(rep["compositions"][n]["nested_cv"]["mean"], 6), -order[n]))
    else:
        final = best_single
    rep["final"] = final
    print("final pipeline:", final, flush=True)

    spec = {"B+G+S": (True, BTHR, SLIVER), "B+G": (True, BTHR, (S_OFF,)),
            "border": (False, BTHR, (S_OFF,)), "grid": (True, (B_OFF,), (S_OFF,)),
            "sliver": (False, (B_OFF,), SLIVER)}[final]
    T = TG if spec[0] else T0
    preds = build(T, thrs, spec[1], spec[2])
    params, f1 = h.full_fit(preds)
    bparams, bf1 = h.full_fit(base)
    p, pb = preds[params], base[bparams]
    fixed = stems(h, np.flatnonzero((pb != y) & (p == y)))
    new = stems(h, np.flatnonzero((pb == y) & (p != y)))
    fp_left = stems(h, np.flatnonzero((p == 1) & (y == 0)))
    fn_left = stems(h, np.flatnonzero((p == 0) & (y == 1)))
    print("full fit", params, round(f1, 4), counts(y, p), "fixed", fixed, "new", new, flush=True)

    # ---- test: same code, full-train params, photos processed blindly
    Tt = Tables(h, "test", grid=spec[0])
    pt = build(Tt, [params[0]], (params[2],), ((params[3], params[4]),), deltas=[params[1]])
    pt = next(iter(pt.values())).astype(int)
    v1 = [line.strip().split(",") for line in open(V1)]
    assert [r[0] for r in v1] == h.test_paths
    v1p = np.array([int(r[1]) for r in v1])
    pbase_t = build(Tables(h, "test", grid=False, sliver=False), [bparams[0]], deltas=[bparams[1]])
    pbase_t = next(iter(pbase_t.values())).astype(int)
    SUB.parent.mkdir(parents=True, exist_ok=True)
    with open(SUB, "w", newline="\n") as fh:
        fh.writelines(f"{rp},{int(v)}\n" for rp, v in zip(h.test_paths, pt))
    lines = open(SUB).read().splitlines()
    test_list = [r.strip() for r in open(ROOT / "data/contest/test.txt") if r.strip()]
    assert len(lines) == 8059 == len(test_list)
    assert [ln.split(",")[0] for ln in lines] == test_list
    assert {ln.split(",")[1] for ln in lines} <= {"0", "1"}
    print(f"test positives {int(pt.sum())} (v1 {int(v1p.sum())}, harness baseline {int(pbase_t.sum())}); "
          f"rows changed vs v1: {int((pt != v1p).sum())} (+{int(((pt == 1) & (v1p == 0)).sum())} "
          f"-{int(((pt == 0) & (v1p == 1)).sum())})", flush=True)

    fc = rep["compositions"].get(final)
    rep.update({
        "final_nested_cv": fc["nested_cv"] if fc else None,
        "final_vs_baseline": fc["vs_baseline"] if fc else rep["singles"][final],
        "full_fit_params": dict(zip(("thr", "delta", "bthr", "sliver_G", "sliver_E"), map(_num, params)),
                                grid_floor=FLOOR if spec[0] else None),
        "full_fit_train_f1": f1, "full_fit_counts": counts(y, p),
        "baseline_full_fit": {"params": list(map(float, bparams)), "train_f1": bf1, **counts(y, pb)},
        "fixed_vs_baseline_full_fit": fixed, "new_vs_baseline_full_fit": new,
        "train_fp_left": fp_left, "train_fn_left": fn_left,
        "test_positives": int(pt.sum()), "test_positives_v1": int(v1p.sum()),
        "test_positives_harness_baseline": int(pbase_t.sum()),
        "rows_changed_vs_v1": int((pt != v1p).sum()),
        "rows_added_vs_v1": int(((pt == 1) & (v1p == 0)).sum()),
        "rows_removed_vs_v1": int(((pt == 0) & (v1p == 1)).sum()),
        "submission": str(SUB),
        "fixed_constants": {"grid": dict(FLOOR=FLOOR, DMIN=pp_grid.DMIN, SLOPE=pp_grid.SLOPE,
                                         RR_MAX=pp_grid.RR_MAX, ANC_VOTES=pp_grid.ANC_VOTES),
                            "border": dict(DS=pp_border.DS, TAU=pp_border.TAU, R_MIN=pp_border.R_MIN,
                                           R_MAX=pp_border.R_MAX, CHORD_MIN=pp_border.CHORD_MIN,
                                           DEPTH_MIN=pp_border.DEPTH_MIN, WIDTH_MAX=pp_border.WIDTH_MAX),
                            "sliver": dict(MARGIN=pp_sliver.MARGIN, DARK=pp_sliver.DARK, FAR=pp_sliver.FAR,
                                           BLUR=pp_sliver.BLUR)},
        "searched_grid": {"thr": "h.thr_grid() (60)", "delta": list(map(float, DELTAS)), "bthr": list(BTHR),
                          "sliver_G_E": [list(map(_num, s)) for s in SLIVER]},
        "runtime_s": round(time.time() - t0, 1),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print("->", OUT / "report.json", SUB)


if __name__ == "__main__":
    main()

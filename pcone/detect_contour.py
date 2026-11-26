"""Label-free P-cone (Pコン跡) detector: segmentation + contour shape analysis.

Pipeline (identical for every photo; no learned weights, no labels, no per-photo tuning):
  1. Gray image of the half-res mosaic (crop grid 18x14, 128 px cells), light Gaussian blur.
  2. Local background = median filter at 1/8 scale (window ~248 half-res px); local noise scale
     sigma = robust median |gray - background| (capped at 2x the photo median so a blotchy stain
     cannot hide a hole). z = (background - gray) / sigma  ("darkness").
  3. Candidate selection masks:
       dark : hysteresis threshold on z (z > 2 grown from seeds z > 4)        -> open holes
       ring : |z| > 2.5, 5 px closing, fill enclosed holes                    -> recesses whose rim is
              a dark arc + bright arc around a wall-toned (shallow / mortar-filled) interior
     each opened with discs of 5 / 11 / 21 px so thin cracks / joint lines are cut off.
  4. Each selected component is re-segmented with a local adaptive threshold on depth below the
     background (fraction of the blob's own depth, chosen where the contour has maximal edge
     support), then connected-component shape features: circularity relative to the fitted
     ellipse (perimeter ratio), solidity, moment-ellipse axis ratio, IoU(blob, ellipse), mean
     darkness inside vs. a local annulus. Blobs cut by the photo edge get a circle fit to the arc.
  5. Radial edge analysis: 48 rays, outermost sharp intensity step per ray (window ~7% of r),
     robust ellipse fit -> refined rim geometry, "edge coverage" (closed sharp outline) and outline
     irregularity (residual spread).
  6. Score = sqrt(product of smoothstep factors): size, shape, regularity, contrast, edge coverage,
     annulus clutter; small blobs must be near-perfect black ellipses (far cone holes), big blobs
     are judged more leniently. Extra factors: interior roughness (spalls / broken voids are rough
     inside, cast cones are smooth) and photo context (a blob <~1/3 the size of a confident cone in
     the same photo is a bug hole unless it is very black).
  7. Non-max suppression; per crop score = max score of detections whose disk intersects the crop.

Outputs (half-res pixel coordinates, r = area-equivalent radius of the rim ellipse):
  data/detect/contour/detections.csv   photo,x,y,r,score
  data/detect/contour/crops.csv        rel_path,score,overlap_px  (every W/*/ jpg on disk)
  data/detect/contour/viz/<photo>.jpg  overlays (red = score >= DEFAULT_THRESHOLD, orange below)
"""
import csv
import math
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # project root (relative to this file)
MOSAIC = ROOT / "data" / "mosaic_half"
SDNET_W = ROOT / "data" / "sdnet" / "W"
OUT = ROOT / "data" / "detect" / "contour"

COLS, ROWS, T = 18, 14, 128          # crop grid; half-res crop size
DEFAULT_THRESHOLD = 0.5               # reported default; tuned by visual sanity checks only
MIN_KEEP = 0.05                       # detections below this score are not written
NAME = re.compile(r"^(\d+)-(\d+)(?:_(\d+))?$")

cv2.setNumThreads(1)


# ----------------------------------------------------------------------------- utilities
def smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ell_perim(a, b):
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def fill_holes(m):
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = m
    inv = ff.copy()
    cv2.floodFill(inv, None, (0, 0), 1)
    ff[inv == 0] = 1
    return ff[1:-1, 1:-1]


def hysteresis(z, lo, hi):
    m = (z > lo).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    seed = np.zeros(n, bool)
    seed[np.unique(lab[z > hi])] = True
    seed[0] = False
    return seed[lab].astype(np.uint8)


# ----------------------------------------------------------------------------- 1-2 preprocessing
def preprocess(rgb, valid_cells):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = np.kron(valid_cells, np.ones((T, T), bool))
    if not valid.all():                      # missing crops: neutral fill, no artificial edges
        g[~valid] = np.median(g[valid])
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))   # fine texture (roughness) map
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    H, W = g.shape
    small = cv2.resize(g, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), 31).astype(np.float32)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    dev = g - bg
    ad = cv2.resize(np.abs(dev), (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    sig = cv2.medianBlur(np.clip(ad * 8, 0, 255).astype(np.uint8), 31).astype(np.float32) / 8 * 1.4826
    sig = np.maximum(cv2.resize(sig, (W, H), interpolation=cv2.INTER_CUBIC), 1.0)
    # cap: a blotchy stain must not inflate the noise scale so much that a hole inside it vanishes
    sig = np.minimum(sig, 2.0 * float(np.median(sig)))
    z = -dev / sig
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8 / sig
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8 / sig
    return dict(g=g, z=z, depth=-dev, sig=sig, gn=g / sig, gx=gx, gy=gy, gm=np.sqrt(gx * gx + gy * gy),
                hp=hp, valid=valid)


# ----------------------------------------------------------------------------- 3 segmentation
SE = {k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) for k in (3, 5, 11, 21)}


def segment(z):
    """-> dict of base masks and list of (source, k, opened_mask) used to *select* blobs."""
    dark = hysteresis(z, 2.0, 4.0)
    absm = (np.abs(z) > 2.5).astype(np.uint8)
    absm = cv2.morphologyEx(absm, cv2.MORPH_CLOSE, SE[5])
    ring = fill_holes(absm)
    bases = {"dark": dark, "ring": cv2.morphologyEx(ring, cv2.MORPH_OPEN, SE[5])}
    sel = []
    for src, base in (("dark", dark), ("ring", ring)):
        for k in (5, 11, 21):
            sel.append((src, k, cv2.morphologyEx(base, cv2.MORPH_OPEN, SE[k])))
    return bases, sel


def _piece(m, comp):
    """Hole-filled connected piece of m that overlaps comp most (None if none)."""
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, SE[3])
    n, bl = cv2.connectedComponents(m, connectivity=8)
    if n < 2:
        return None
    cnts = np.bincount(bl[comp > 0], minlength=n)
    cnts[0] = 0
    if cnts.max() == 0:
        return None
    return fill_holes((bl == int(np.argmax(cnts))).astype(np.uint8))


def _edge_support(blob, gm):
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=len)[:, 0, :]
    return float(np.mean(gm[c[:, 1], c[:, 0]]))


FRACS = (0.15, 0.25, 0.35, 0.5)


def blob_mask(src, k, comp, roi, P, bases):
    """Pixel-accurate mask of the object selected by `comp` (opened component, ROI coords).
    dark: local adaptive threshold on depth below the local background; the threshold (a fraction
          of the blob's own depth) is chosen so that the contour lies on the sharpest intensity
          steps (max mean gradient along the contour) -> the cast rim, not a halo or terminator;
    ring: the filled-rim mask, lightly opened."""
    Y0, Y1, X0, X1 = roi
    d = k // 2 + 2
    near = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    if src != "dark":
        return _piece(bases["ring"][Y0:Y1, X0:X1] & near, comp)
    depth = P["depth"][Y0:Y1, X0:X1]
    gm = P["gm"][Y0:Y1, X0:X1]
    D = float(np.percentile(depth[comp > 0], 95))
    if D <= 0:
        return None
    best, best_s = None, -1.0
    for fr in FRACS:
        b = _piece(((depth > fr * D) & (near > 0)).astype(np.uint8), comp)
        if b is None or b.sum() < 60:
            continue
        s = _edge_support(b, gm)
        if s > best_s:
            best, best_s = b, s
    return best


# ----------------------------------------------------------------------------- 4 blob features
def _circle_fit(p):
    """Kasa least-squares circle through points p (N,2) -> (cx, cy, r) or None."""
    x, y = p[:, 0], p[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    rhs = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def blob_features(blob, roi, P):
    Y0, Y1, X0, X1 = roi
    z = P["z"]
    H, W = z.shape
    A = int(blob.sum())
    if A < 60:
        return None
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea)
    if len(cnt) < 6:
        return None
    bx, by, bw, bh = cv2.boundingRect(cnt)
    border = X0 + bx <= 1 or Y0 + by <= 1 or X0 + bx + bw >= W - 1 or Y0 + by + bh >= H - 1
    if border:
        # truncated by the photo edge: fit the ellipse to the contour away from the edge
        pts = (cnt[:, 0, :] + (X0, Y0)).astype(np.float32)
        keep = (pts[:, 0] > 2) & (pts[:, 0] < W - 3) & (pts[:, 1] > 2) & (pts[:, 1] < H - 3)
        if keep.sum() < 8:
            return None
        # an arc constrains a circle far better than a free ellipse -> algebraic circle fit
        cxy = _circle_fit(pts[keep].astype(np.float64))
        if cxy is None:
            return None
        ecx, ecy, rr = cxy
        if not (2 < rr < 2.0 * max(bw, bh) + 5):   # wild fit on a tiny / straight arc
            return None
        # most of the disk must lie inside the photo (a sliver along the edge is not identifiable)
        out_x = max(0.0, -ecx, ecx - (W - 1))
        out_y = max(0.0, -ecy, ecy - (H - 1))
        if max(out_x, out_y) > 0.55 * rr:
            return None
        a = b = rr
        th = 0.0
        cx, cy = ecx - X0, ecy - Y0
    else:
        mo = cv2.moments(blob, True)
        cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
        tr, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
        disc = math.sqrt(max(tr * tr / 4 - det, 0))
        a = 2 * math.sqrt(max(tr / 2 + disc, 1e-6))
        b = 2 * math.sqrt(max(tr / 2 - disc, 1e-6))
        th = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    hull = cv2.convexHull(cnt)
    sol = cv2.contourArea(cnt) / max(cv2.contourArea(hull), 1.0)
    per = cv2.arcLength(cnt, True)
    em = np.zeros_like(blob)
    cv2.ellipse(em, ((cx, cy), (2 * a, 2 * b), math.degrees(th)), 1, -1)
    inter = int((em & blob).sum())
    iou = inter / max(int(em.sum()) + A - inter, 1)
    circ = 4 * math.pi * A / max(per, 1.0) ** 2
    # perimeter relative to the ideal ellipse (~1.05 for a clean digital ellipse; ragged > 1.3)
    smooth = per / ell_perim(a, b) if not border else 1.0 / max(iou, 0.3)
    zz = z[Y0:Y1, X0:X1]
    ann = np.zeros_like(blob)
    cv2.ellipse(ann, ((cx, cy), (4 * a + 8, 4 * b + 8), math.degrees(th)), 1, -1)
    cv2.ellipse(ann, ((cx, cy), (2.6 * a + 6, 2.6 * b + 6), math.degrees(th)), 0, -1)
    annz = zz[ann > 0]
    zin = float(zz[(em & blob) > 0].mean()) if (em & blob).any() else 0.0
    zann = float(np.median(annz)) if annz.size else 0.0
    clutter = float((annz > 3).mean()) if annz.size else 1.0
    return dict(x=cx + X0, y=cy + Y0, a=a, b=b, th=th, area=A, q=b / a, sol=sol, circ=circ,
                smooth=smooth, iou=iou, zin=zin, contrast=max(zin - zann, 0.6 * zin), clutter=clutter,
                border=border)


# ----------------------------------------------------------------------------- 5 radial edge analysis
NA = 48
ANG = np.linspace(0, 2 * np.pi, NA, endpoint=False)
SS = np.linspace(0.5, 1.6, 48)
J_OK = int(np.searchsorted(SS, 1.2))
E_T = 3.0          # "sharp step": intensity step > 3 local-noise sigma across the rim
OUTER_FRAC = 0.5


def _fit(px, py):
    try:
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(np.stack([px, py], 1).astype(np.float32))
    except cv2.error:
        return None
    if not (np.isfinite(d1) and np.isfinite(d2)) or min(d1, d2) < 2:
        return None
    a, b = max(d1, d2) / 2, min(d1, d2) / 2
    th = math.radians(ang) + (math.pi / 2 if d2 > d1 else 0.0)
    return cx, cy, a, b, th


def _rel_res(px, py, el):
    cx, cy, a, b, th = el
    ct, st = math.cos(th), math.sin(th)
    u = (px - cx) * ct + (py - cy) * st
    v = -(px - cx) * st + (py - cy) * ct
    return np.abs(np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1)


def edge_measure(f, P):
    """Cast 48 rays across the blob's ellipse. Along each ray take the intensity step across a
    window of ~max(1.5 px, 7% of the radius) (normalised by local noise sigma), and keep the
    outermost strong step (on the lit side of a cone the rim lies outside the shadow terminator).
    A cast cone rim gives a sharp step on (nearly) every ray, all on one ellipse.
    Robust ellipse fit to the step points ->
       cov    fraction of in-photo rays with a sharp step within 12% of the fitted ellipse,
       spread median relative residual of the step points (outline irregularity),
       ell    the fitted ellipse (refined rim geometry)."""
    In, H, W = P["gn"], P["gn"].shape[0], P["gn"].shape[1]
    cx, cy, a, b, th = f["x"], f["y"], f["a"], f["b"], f["th"]
    dx, dy = np.cos(ANG), np.sin(ANG)
    ct, st = math.cos(th), math.sin(th)
    u, v = dx * ct + dy * st, -dx * st + dy * ct
    rho = 1.0 / np.sqrt((u / a) ** 2 + (v / b) ** 2)
    R = rho[:, None] * SS[None, :]
    X = (cx + dx[:, None] * R).astype(np.float32)
    Y = (cy + dy[:, None] * R).astype(np.float32)
    inside = (X >= 1) & (X <= W - 2) & (Y >= 1) & (Y <= H - 2)
    I = cv2.remap(In, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    r = math.sqrt(a * b)
    step_px = (SS[1] - SS[0]) * r
    h = int(np.clip(round(max(1.5, 0.07 * r) / step_px), 1, 6))
    sc = np.zeros_like(I)
    sc[:, h:-h] = np.abs(I[:, 2 * h:] - I[:, :-2 * h])
    sc[~inside] = 0
    ia = np.arange(NA)
    mx = sc.max(axis=1, keepdims=True)
    pk = np.zeros_like(sc, bool)
    pk[:, 1:-1] = (sc[:, 1:-1] >= sc[:, :-2]) & (sc[:, 1:-1] >= sc[:, 2:])
    ok = pk & (sc >= np.maximum(E_T, OUTER_FRAC * mx))
    last = len(SS) - 1 - np.argmax(ok[:, ::-1], axis=1)
    j = np.where(ok.any(axis=1), last, np.argmax(sc, axis=1))
    j = np.clip(j, 1, len(SS) - 2)
    e = sc[ia, j]
    y0_, y1_, y2_ = sc[ia, j - 1], e, sc[ia, j + 1]
    den = y0_ - 2 * y1_ + y2_
    off = np.where(den < -1e-6, 0.5 * (y0_ - y2_) / np.where(den < -1e-6, den, -1), 0.0)
    sj = SS[j] + np.clip(off, -1, 1) * (SS[1] - SS[0])
    px, py = cx + dx * rho * sj, cy + dy * rho * sj
    valid = inside[:, J_OK]
    strong = valid & (e > E_T)
    nvalid = int(valid.sum())
    out = dict(cov=0.0, spread=1.0, ell=None, estr=float(np.median(e[valid])) if nvalid else 0.0)
    if nvalid < 12 or strong.sum() < 8:
        return out
    el = _fit(px[strong], py[strong])
    if el is None:
        return out
    res = _rel_res(px, py, el)
    inl = strong & (res < max(0.15, 2.5 * float(np.median(res[strong]))))
    if inl.sum() >= 8:
        el2 = _fit(px[inl], py[inl])
        if el2 is not None:
            el = el2
            res = _rel_res(px, py, el)
    out["cov"] = float((strong & (res < 0.12)).sum()) / nvalid
    out["spread"] = float(np.median(res[strong]))
    out["ell"] = el
    return out


# ----------------------------------------------------------------------------- 6 scoring
def score_candidate(f, e, src="dark"):
    """Soft AND of shape / regularity / contrast / edge / clutter factors. Small blobs must be
    near-perfect, very dark ellipses (a far-away cone hole is deep and black; a bug hole / air void
    of the same pixel size in a close-up is shallow, lit inside and ragged); big blobs get looser
    shape tolerances (texture makes big outlines ragged, but nothing else that big is round & dark)."""
    a, b, cx, cy, th = f["a"], f["b"], f["x"], f["y"], f["th"]
    el = e["ell"]
    if el is not None and e["cov"] >= 0.5:
        ea, eb = el[2], el[3]
        ar = (ea * eb) / (a * b)
        if 0.6 <= ar <= 2.5 and math.hypot(el[0] - cx, el[1] - cy) < 0.6 * a:
            cx, cy, a, b, th = el
    r = math.sqrt(a * b)
    q = b / a
    # ring-only blobs (rim outline filled in) are judged as if 25% smaller: a lit-floor bug hole
    # also has a dark rim + bright floor, only genuinely cone-sized recesses should pass
    r_eff = r * (0.75 if src == "ring" else 1.0)
    t = float(1 - smoothstep(r_eff, 12, 28))      # 1 = small blob (strict), 0 = big blob (lenient)
    s_size = smoothstep(r, 7, 11) * (1 - smoothstep(r, 110, 140))
    c_lo = 2 + 9 * t
    s_con = smoothstep(abs(f["contrast"]), c_lo, c_lo + 3 + 5 * t)
    sol_lo = 0.72 + 0.13 * t
    iou_lo = 0.68 + 0.16 * t
    sm_lo = 1.5 - 0.3 * t
    s_shape = (smoothstep(f["sol"], sol_lo, sol_lo + 0.12 - 0.02 * t)
               * smoothstep(f["iou"], iou_lo, iou_lo + 0.12 - 0.03 * t)
               * smoothstep(min(q, f["q"] + 0.1), 0.28, 0.45)
               * (1 - smoothstep(f["smooth"], sm_lo, sm_lo + 0.3)))
    sp_lo = 0.10 - 0.065 * t
    s_reg = 1 - smoothstep(e["spread"], sp_lo, sp_lo + 0.08 - 0.02 * t)
    cv_lo = 0.30 + 0.35 * t
    s_edge = smoothstep(e["cov"], cv_lo, cv_lo + 0.3)
    # dark clutter in the surrounding annulus: only extreme values penalised (cone holes often sit
    # in a concentric dark water-stain halo, which must not veto them)
    s_clut = 1 - 0.7 * smoothstep(f["clutter"], 0.6, 0.9)
    score = float((s_size * s_shape * s_reg * s_con * s_edge * s_clut) ** 0.5)
    return dict(x=cx, y=cy, r=r, a=a, b=b, th=th, score=score, s_size=float(s_size),
                s_shape=float(s_shape), s_reg=float(s_reg), s_con=float(s_con), s_edge=float(s_edge),
                s_clut=float(s_clut), cov=e["cov"], spread=e["spread"], estr=e["estr"])


def interior_roughness(hp, x, y, r):
    """median fine-texture inside 0.7r / median in the 1.6r-2.6r annulus. A cast cone recess has a
    smooth (formwork-cone) inner surface; spalls, bug holes and broken voids are rough inside."""
    H, W = hp.shape
    R = int(math.ceil(2.6 * r)) + 1
    x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
    if x1 <= x0 or y1 <= y0:
        return 1.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    roi = hp[y0:y1, x0:x1]
    inn, ann = roi[d < 0.7 * r], roi[(d > 1.6 * r) & (d < 2.6 * r)]
    if inn.size < 10 or ann.size < 10:
        return 1.0
    return float(np.median(inn) / max(np.median(ann), 1e-3))


# ----------------------------------------------------------------------------- per photo
def detect_photo(pid, return_all=False):
    rgb = np.load(MOSAIC / f"{pid}.npy")
    vc = np.load(MOSAIC / f"{pid}_filled.npy")
    P = preprocess(rgb, vc)
    z = P["z"]
    H, W = z.shape
    bases, sel = segment(z)
    cands = []
    for src, k, m in sel:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x0, y0, w, h, A = (int(v) for v in st[i])
            if A < 110 or A > 60000 or max(w, h) > 300:
                continue
            if max(w, h) > 5 * max(min(w, h), 1):
                continue
            pad = int(max(w, h) * 1.2) + 4 + k
            roi = (max(0, y0 - pad), min(H, y0 + h + pad), max(0, x0 - pad), min(W, x0 + w + pad))
            comp = (lab[roi[0]:roi[1], roi[2]:roi[3]] == i).astype(np.uint8)
            blob = blob_mask(src, k, comp, roi, P, bases)
            if blob is None:
                continue
            f = blob_features(blob, roi, P)
            if f is None or f["sol"] < 0.6 or f["q"] < 0.2 or abs(f["contrast"]) < 1.0:
                continue
            e = edge_measure(f, P)
            s = score_candidate(f, e, src)
            if s["score"] >= MIN_KEEP:
                s["rough"] = interior_roughness(P["hp"], s["x"], s["y"], s["r"])
                # (only meaningful once the interior spans many pixels; tiny far holes are judged
                # by the strict small-blob shape / darkness rules instead)
                pen = float(smoothstep(s["rough"], 1.9, 2.6) * smoothstep(s["r"], 14, 22))
                s["score"] *= math.sqrt(1 - pen)
            s.update(src=src, k=k, sol=f["sol"], iou=f["iou"], q=f["q"], smooth=f["smooth"],
                     contrast=f["contrast"], clutter=f["clutter"])
            cands.append(s)
    if return_all:
        return pid, cands
    # non-maximum suppression: the same object is found by several mask variants; drop a candidate
    # whose centre lies inside an already kept (higher-scoring) disk or vice versa
    cands.sort(key=lambda c: -c["score"])
    dets = []
    for c in cands:
        if c["score"] < MIN_KEEP:
            break
        if any(math.hypot(c["x"] - d["x"], c["y"] - d["y"]) < max(c["r"], d["r"]) for d in dets):
            continue
        dets.append(c)
    # photo context: cone holes in one photo share a scale. A blob far smaller than a confident
    # large cone in the same photo is a bug hole / void, unless it is as black as a far cone hole.
    big = [d["r"] for d in dets if d["score"] >= 0.8 and d["r"] >= 25]
    if big:
        r_ref = max(big)
        for d in dets:
            s_rel = max(float(smoothstep(d["r"] / r_ref, 0.22, 0.38)),
                        float(smoothstep(abs(d["contrast"]), 15, 22)))
            d["score"] *= math.sqrt(s_rel)
        dets = [d for d in dets if d["score"] >= MIN_KEEP]
    for d in dets:
        d["photo"] = pid
    return pid, dets


# ----------------------------------------------------------------------------- 7 crop table
def disk_cell_overlap(x, y, r):
    """{(row, col): overlap area in half-res px^2} for the disk, by 4x supersampling."""
    out = {}
    x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r))
    y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, COLS * T), min(y1, ROWS * T)
    if x1 <= x0 or y1 <= y0:
        return out
    ss = 4
    xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss
    ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss
    inside = ((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2) <= r * r
    cc = (xs // T).astype(int)
    rr = (ys // T).astype(int)
    for ri in np.unique(rr):
        rows = inside[rr == ri]
        for ci in np.unique(cc):
            a = rows[:, cc == ci].sum() / (ss * ss)
            if a > 0:
                out[(int(ri), int(ci))] = float(a)
    return out


def list_crops():
    rels = []
    for sub in sorted(os.listdir(SDNET_W)):
        d = SDNET_W / sub
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".jpg"):
                rels.append(f"W/{sub}/{fn}")
    return rels


def draw_viz(pid, dets, path):
    rgb = np.load(MOSAIC / f"{pid}.npy")
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for c in range(1, COLS):
        cv2.line(vis, (c * T, 0), (c * T, ROWS * T), (200, 200, 200), 1)
    for r in range(1, ROWS):
        cv2.line(vis, (0, r * T), (COLS * T, r * T), (200, 200, 200), 1)
    for d in sorted(dets, key=lambda d: d["score"]):
        if d["score"] < 0.15:
            continue
        col = (0, 0, 255) if d["score"] >= DEFAULT_THRESHOLD else (0, 200, 255)
        cv2.circle(vis, (int(round(d["x"])), int(round(d["y"]))), int(round(d["r"])), col, 4)
        cv2.putText(vis, f"{d['score']:.2f}", (int(d["x"] + d["r"] + 4), int(d["y"] + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
    vis = cv2.resize(vis, (1152, 896), interpolation=cv2.INTER_AREA)
    cv2.putText(vis, pid, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])


VIZ_PHOTOS = ["7069", "7094", "7099", "7111", "7113", "7131", "7090", "7104", "7115", "7128",
              "7097", "7122", "7092", "7117", "7134", "7075"]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "viz").mkdir(exist_ok=True)
    pids = sorted(p.stem for p in MOSAIC.glob("*.npy") if not p.stem.endswith("_filled"))
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        results = dict(pool.map(detect_photo, pids))
    with open(OUT / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "x", "y", "r", "score"])
        for pid in pids:
            for d in sorted(results[pid], key=lambda d: -d["score"]):
                w.writerow([pid, f"{d['x']:.1f}", f"{d['y']:.1f}", f"{d['r']:.1f}", f"{d['score']:.4f}"])
    # per-cell best detection
    best = {}
    for pid in pids:
        for d in results[pid]:
            for cell, area in disk_cell_overlap(d["x"], d["y"], d["r"]).items():
                key = (pid,) + cell
                if key not in best or d["score"] > best[key][0]:
                    best[key] = (d["score"], area)
    rels = list_crops()
    n_pos = 0
    with open(OUT / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_path", "score", "overlap_px"])
        for rel in rels:
            m = NAME.match(Path(rel).stem)
            sc, ov = 0.0, 0.0
            if m:
                idx = int(m.group(2))
                r, c = divmod(idx - 1, COLS)
                sc, ov = best.get((m.group(1), r, c), (0.0, 0.0))
            n_pos += sc >= DEFAULT_THRESHOLD
            w.writerow([rel, f"{sc:.4f}", f"{ov:.1f}"])
    for pid in VIZ_PHOTOS:
        if pid in results:
            draw_viz(pid, results[pid], OUT / "viz" / f"{pid}.jpg")
    n_det = sum(d["score"] >= DEFAULT_THRESHOLD for v in results.values() for d in v)
    zero = sum(1 for v in results.values() if not any(d["score"] >= DEFAULT_THRESHOLD for d in v))
    print(f"photos={len(pids)} crops={len(rels)} dets>=thr={n_det} crops>=thr={n_pos} "
          f"photos_without_det={zero} time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Label-free P-cone mark (Pコン跡) detector: multi-scale LoG dark-blob search + per-blob verification.

Run:  .venv/bin/python pcone/detect_log.py     (no arguments; ~70 s for the 72 photos on an M4)

One fixed algorithm, applied identically to every photo (no learning, no labels, no per-photo tuning):
  1. grey = luminance of the half-res mosaic (1792 x 2304); missing (black) cells are filled from
     their neighbourhood and detections touching them are dropped.
  2. Illumination normalisation: N = grey / background, background = 31-tap median at 1/8 scale.
  3. Scale-normalised LoG (sigma^2 * Laplacian, positive = dark blob) on N at quarter res for radii
     10..96 half-res px (x1.14 steps); 3-D (x, y, scale) maxima -> NMS -> 150 strongest per photo.
  4. Verification of every candidate in polar coordinates (64 rays, 32 samples per radius):
       - rim fit: per-ray maxima of the radial step |N(rho+0.1r) - N(rho-0.1r)| in [0.5, 1.7] r,
         weighted RANSAC circle (support = sum over rays), then narrow per-ray search + weighted
         least squares (x2). The hole's mouth usually wins over the darker cone bottom.
       - radial contrast profile: ring (1.25-1.9 r) median minus inside (<0.8 r) mean, also / ring MAD;
       - edge: 70th-percentile rim step (/ texture), angular coverage of a strong rim, rim-radius
         consistency (circle) or 2nd-order Fourier residual (ellipse = oblique view), sharpness;
       - thresholded blob (at the mid level between inside and ring): circularity, solidity, IoU with
         its fitted ellipse, area / disk area; fraction of rays whose ring is as dark as the inside
         (elongated dark bars, joints, pipes);
       - colour: Lab b* inside minus ring (marks are grey/bluish; rust, pipes, brown spalls are warm).
     Only samples inside the photo are used; circles cut by the border need >~half the rim visible
     unless clearly dark inside.
  5. score = weighted geometric mean of soft ramps (a soft AND) x size gate (r < ~20 px must be a
     near-perfect ellipse: pores / bug holes) x border factor x colour factor, in [0, 1].
     Default decision threshold THRESH = 0.45.
  6. crops.csv: per crop file, max score over detections whose disk intersects its 128x128 half-res
     square, and that detection's visible disk area (overlap_px).

Outputs (data/detect/log/): detections.csv (photo,x,y,r,score; half-res px, r = rim radius),
crops.csv (rel_path,score,overlap_px), features.csv (all verified candidates with features, debug),
viz/<photo>.jpg (quarter-res overlays; red >= THRESH, orange >= 0.3, blue >= 0.15).

Known failure modes: shallow / very oblique marks whose interior is as light as the wall (only a small
dark sub-spot is found -> low score); tiny far-away holes (r < ~10 half-res px) are indistinguishable
from pores; faint mortar-filled "ghost" circles score ~0.3-0.5; pipe ends and dark round spalls can
pass; partial marks at the photo border with < ~40% of the rim visible are suppressed.
"""
import csv
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent  # project root (relative to this file)
MOSAIC = ROOT / "data/mosaic_half"
SDNET = ROOT / "data/sdnet"
OUT = ROOT / "data/detect/log"

COLS, ROWS, T = 18, 14, 128          # grid and half-res cell size
NAME = re.compile(r"^(\d+)-(\d+)(?:_(\d+))?$")

# ---- parameters (fixed for all photos) -------------------------------------------------------
R_MIN, R_MAX = 10.0, 96.0            # half-res radius search range
R_STEP = 1.14                        # geometric scale step
LOG_MIN = 0.012                      # min scale-normalised LoG response for a candidate
MAX_CAND = 150                       # candidates verified per photo (strongest LoG first)
NA, SPR = 64, 32                     # polar sampling: angles, samples per radius
KEEP_MIN_SCORE = 0.05                # detections written to the CSV
THRESH = 0.45                        # default decision threshold (reported)

cv2.setNumThreads(4)


def ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi, linear in between (works with lo > hi too)."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ---- preprocessing ----------------------------------------------------------------------------
def load(photo):
    M = np.load(MOSAIC / f"{photo}.npy")
    fpath = MOSAIC / f"{photo}_filled.npy"
    filled = np.load(fpath) if fpath.exists() else np.ones((ROWS, COLS), bool)
    grey = (M[..., 0] * 0.299 + M[..., 1] * 0.587 + M[..., 2] * 0.114).astype(np.float32)
    bstar = cv2.cvtColor(M, cv2.COLOR_RGB2LAB)[..., 2].astype(np.float32)   # yellow(+)/blue(-)
    valid = np.kron(filled, np.ones((T, T), bool))
    return grey, bstar, valid


def warmth(bstar, x, y, r):
    """Lab b* of the disk interior minus the surrounding ring (open / filled P-cone marks are grey
    or bluish in shadow; rust, wood, soil and brown spalls are warm)."""
    H, W = bstar.shape
    h = int(2 * r) + 2
    xa, xb, ya, yb = max(int(x) - h, 0), min(int(x) + h, W), max(int(y) - h, 0), min(int(y) + h, H)
    if xb - xa < 3 or yb - ya < 3:
        return 0.0
    yy, xx = np.mgrid[ya:yb, xa:xb]
    d2 = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r)
    B = bstar[ya:yb, xa:xb]
    ins, ring = d2 < 0.64, (d2 > 1.56) & (d2 < 3.6)
    if ins.sum() < 5 or ring.sum() < 5:
        return 0.0
    return float(B[ins].mean() - np.median(B[ring]))


def normalise(grey, valid):
    """Fill missing cells, divide by a robust large-scale background. Returns N at half res."""
    g = grey.copy()
    if not valid.all():
        v = valid.astype(np.float32)
        num = cv2.GaussianBlur(g * v, (0, 0), 40)
        den = cv2.GaussianBlur(v, (0, 0), 40)
        fill = num / np.maximum(den, 1e-3)
        g[~valid] = fill[~valid]
    H, W = g.shape
    small = cv2.resize(g, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), 31).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    return g / np.maximum(bg, 1.0)


def log_candidates(N1, scale):
    """Scale-space maxima of sigma^2*Laplacian(G_sigma * N1) (positive = dark blob).
    N1 is the image at 1/scale of half res. Returns array (x, y, r, resp) in half-res units."""
    radii = []
    r = R_MIN
    while r <= R_MAX * 1.001:
        radii.append(r)
        r *= R_STEP
    stack = []
    for rh in radii:
        s = rh / scale / math.sqrt(2.0)
        G = cv2.GaussianBlur(N1, (0, 0), s)
        L = cv2.Laplacian(G, cv2.CV_32F, ksize=1) * (s * s)
        stack.append(L)
    S = np.stack(stack)                                           # (K, H, W)
    mx = ndi.maximum_filter(S, size=(3, 5, 5), mode="nearest")
    peaks = (S == mx) & (S > LOG_MIN)
    k, y, x = np.nonzero(peaks)
    resp = S[k, y, x]
    rr = np.array(radii)[k]
    cand = np.stack([x * scale + (scale - 1) / 2, y * scale + (scale - 1) / 2, rr, resp], 1)
    return cand[np.argsort(-resp)]


def nms(c, frac=0.8, key=3, cap=3000):
    """Greedy suppression: drop a candidate whose centre lies within frac*max(r) of a kept one."""
    c = c[np.argsort(-c[:, key])][:cap]
    kept = []
    kx = np.empty(0); ky = np.empty(0); kr = np.empty(0)
    for i in range(len(c)):
        x, y, r = c[i, :3]
        if len(kept) and (np.hypot(kx - x, ky - y) < frac * np.maximum(kr, r)).any():
            continue
        kept.append(i)
        kx, ky, kr = np.append(kx, x), np.append(ky, y), np.append(kr, r)
    return c[kept]


# ---- per-candidate verification ---------------------------------------------------------------
class Pyramid:
    def __init__(self, N0):
        self.lv = [N0]
        for _ in range(3):
            a = self.lv[-1]
            self.lv.append(cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2),
                                      interpolation=cv2.INTER_AREA))
        ang = (np.arange(NA) + 0.5) * 2 * np.pi / NA
        self.cos, self.sin = np.cos(ang)[:, None], np.sin(ang)[:, None]

    @staticmethod
    def level(r):
        """Pyramid level so that the radius is >= ~12 px there."""
        return int(np.clip(math.floor(math.log2(max(r, 1.0) / 12.0)), 0, 3))

    def polar(self, x, y, r, ext=2.0):
        """Polar unwrap around half-res (x, y), radius unit r -> P (NA, SPR*ext) and validity V
        (sample inside the photo); sample j sits at rho = (j + 0.5) / SPR * r."""
        L = self.level(r)
        s = 2 ** L
        img = self.lv[L]
        H, W = img.shape
        rho = ((np.arange(int(SPR * ext)) + 0.5) / SPR)[None, :]
        cx, cy, rl = (x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5, r / s
        mx = (cx + rl * rho * self.cos).astype(np.float32)
        my = (cy + rl * rho * self.sin).astype(np.float32)
        V = (mx > -0.5) & (mx < W - 0.5) & (my > -0.5) & (my < H - 0.5)
        P = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return P, V


def radial_step(P, V):
    """Smoothed polar image, signed step (outside - inside) over +-0.1 r along each ray and the
    local derivative; steps that touch samples outside the photo are zeroed."""
    pad = 4                                                     # angles wrap around
    Pw = np.concatenate([P[-pad:], P, P[:pad]], 0)
    Ps = cv2.GaussianBlur(Pw, (0, 0), sigmaX=1.2, sigmaY=0.8,
                          borderType=cv2.BORDER_REPLICATE)[pad:-pad]
    d = 3                                                       # ~0.1 r
    step = np.zeros_like(Ps)
    step[:, d:-d] = Ps[:, 2 * d:] - Ps[:, :-2 * d]
    sv = np.zeros_like(V)
    sv[:, d:-d] = V[:, 2 * d:] & V[:, :-2 * d]
    step[~sv] = 0.0
    deriv = np.zeros_like(Ps)
    deriv[:, 1:-1] = (Ps[:, 2:] - Ps[:, :-2]) * 0.5
    return Ps, step, deriv, sv


def rim(P, V, lo=0.7, hi=1.3):
    """Per-ray rim (strongest |step| within [lo, hi] r): sample index, signed step, sharpness and
    ray validity (rim inside the photo)."""
    Ps, step, deriv, sv = radial_step(P, V)
    a, b = int(SPR * lo), int(SPR * hi)
    idx = np.abs(step[:, a:b]).argmax(1) + a
    ar = np.arange(NA)
    st = step[ar, idx]
    rv = sv[ar, idx] & sv[:, a:b].any(1)
    win = np.clip(idx[:, None] + np.arange(-2, 3), 0, P.shape[1] - 1)
    dmax = np.abs(deriv[ar[:, None], win]).max(1)
    sharp = dmax * 6 / np.maximum(np.abs(st), 1e-4)            # ~1 for a linear ramp over 0.2 r
    return idx, st, sharp, rv


def fit_circle(px, py, w):
    """Weighted algebraic (Kasa) circle fit."""
    A = np.stack([px, py, np.ones_like(px)], 1) * w[:, None]
    b = (px * px + py * py) * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    rad = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-6))
    return cx, cy, rad


def blob_shape(pyr, x, y, r, thr):
    """Threshold the local patch at thr; circularity, solidity and area ratio (vs the part of the
    fitted disk inside the photo) of the dark connected blob at the centre."""
    L = pyr.level(r)
    s = 2 ** L
    img = pyr.lv[L]
    cx, cy, rl = (x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5, r / s
    h = int(math.ceil(rl * 1.8)) + 2
    x0, y0 = int(round(cx)) - h, int(round(cy)) - h
    H, W = img.shape
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * h + 1, W), min(y0 + 2 * h + 1, H)
    if xb - xa < 5 or yb - ya < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    patch = cv2.GaussianBlur(img[ya:yb, xa:xb], (0, 0), max(0.6, rl * 0.06))
    bw = (patch < thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(bw, connectivity=8)
    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    yy, xx = np.mgrid[ya:yb, xa:xb]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inner = d2 <= (0.6 * rl) ** 2
    disk_area = float((d2 <= rl * rl).sum())
    counts = np.bincount(lab[inner].ravel(), minlength=n)
    counts[0] = 0
    k = counts.argmax()
    if counts[k] == 0 or disk_area < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    comp = ndi.binary_fill_holes(lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    if area < 4 or per <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    circ = 4 * math.pi * area / (per * per)
    hull = cv2.contourArea(cv2.convexHull(c))
    solid = area / max(hull, 1e-6)
    ell, axr = 0.0, 0.0
    if len(c) >= 6:                                     # oblique views turn the mark into an ellipse
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(c)
        if min(ea, eb) > 1:
            em = np.zeros_like(comp)
            cv2.ellipse(em, ((ex, ey), (ea, eb), ang), 1, -1)
            inter = float((em & comp).sum())
            ell = inter / max(float((em | comp).sum()), 1.0)
            axr = min(ea, eb) / max(ea, eb)
    return min(circ, 1.0), solid, float(comp.sum()) / disk_area, ell, axr


def rim_points(pyr, x, y, r, lo, hi, k=2):
    """Up to k strongest local maxima of |radial step| per ray within [lo, hi] r.
    Returns px, py, w, ray (flattened, half-res coordinates)."""
    P, V = pyr.polar(x, y, r, ext=hi + 0.2)
    _, step, _, _ = radial_step(P, V)
    a, b = int(SPR * lo), int(SPR * hi)
    A = np.abs(step[:, a - 1:b + 1])
    loc = (A[:, 1:-1] >= A[:, :-2]) & (A[:, 1:-1] >= A[:, 2:])
    Vm = np.where(loc, A[:, 1:-1], 0.0)
    order = np.argsort(-Vm, 1)[:, :k]
    ray = np.repeat(np.arange(NA), k)
    j = order.ravel()
    w = Vm[ray, j]
    rho = (j + a + 0.5) / SPR * r
    px = x + rho * pyr.cos[ray, 0]
    py = y + rho * pyr.sin[ray, 0]
    m = w > 0
    return px[m], py[m], w[m], ray[m]


def ransac_circle(px, py, w, ray, x, y, r, rmin, rmax, max_off, n_hyp=160, seed=0):
    """Weighted RANSAC circle; support = sum over rays of the best inlier weight.
    Returns (cx, cy, R, support, ray coverage) or None."""
    if len(px) < 8:
        return None
    rng = np.random.default_rng(seed)
    I = rng.choice(len(px), size=(n_hyp, 3), p=w / w.sum())
    x1, y1 = px[I[:, 0]], py[I[:, 0]]
    x2, y2 = px[I[:, 1]], py[I[:, 1]]
    x3, y3 = px[I[:, 2]], py[I[:, 2]]
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    ok = np.abs(d) > 1e-6
    d = np.where(ok, d, 1.0)
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    R = np.hypot(x1 - ux, y1 - uy)
    ok &= (R > rmin) & (R < rmax) & (np.hypot(ux - x, uy - y) < max_off)
    if not ok.any():
        return None
    ux, uy, R = ux[ok], uy[ok], R[ok]
    dist = np.abs(np.hypot(px[None] - ux[:, None], py[None] - uy[:, None]) - R[:, None])
    W = np.where(dist < np.maximum(1.5, 0.07 * R)[:, None], w[None], 0.0)
    best = np.zeros((len(R), NA))
    hi_ = np.repeat(np.arange(len(R)), len(ray))
    np.maximum.at(best, (hi_, np.tile(ray, len(R))), W.ravel())
    sup = best.sum(1)
    h = sup.argmax()
    return ux[h], uy[h], R[h], sup[h], float((best[h] > 0).mean())


def fit_rim(pyr, x, y, r, lo=0.5, hi=1.7, iters=2):
    """Robust rim fit around a LoG blob: RANSAC on per-ray edge maxima, then narrow-window
    per-ray search + weighted least squares. Returns (x, y, r, rim points)."""
    px, py, w, ray = rim_points(pyr, x, y, r, lo, hi)
    hyp = ransac_circle(px, py, w, ray, x, y, r, lo * r, hi * r, 0.6 * r)
    if hyp is not None:
        x, y, r = hyp[:3]
    pts = (px, py, w)
    for _ in range(iters):
        P, V = pyr.polar(x, y, r)
        idx, st, _, rv = rim(P, V, 0.85, 1.15)
        if rv.sum() < 12:
            break
        wa = np.abs(st)
        rho_px = (idx + 0.5) / SPR * r
        qx = x + rho_px * pyr.cos[:, 0]
        qy = y + rho_px * pyr.sin[:, 0]
        pts = (qx[rv], qy[rv], wa[rv])
        good = rv & (wa >= np.quantile(wa[rv], 0.3))
        cx, cy, rad = fit_circle(qx[good], qy[good], wa[good])
        if not (0.85 * r < rad < 1.18 * r) or math.hypot(cx - x, cy - y) > 0.2 * r:
            break
        x, y, r = cx, cy, rad
    return x, y, r, pts


def outer_rim(pyr, x, y, r, sup_main):
    """Look for a concentric larger rim (cone mouth around the dark bottom). Returns radius or r."""
    px, py, w, ray = rim_points(pyr, x, y, r, 1.15, 2.3)
    hyp = ransac_circle(px, py, w, ray, x, y, r, 1.2 * r, 2.2 * r, 0.35 * r, seed=1)
    if hyp is None:
        return r
    ux, uy, R, sup, cov = hyp
    return R if (cov >= 0.5 and sup >= 0.5 * sup_main) else r


def rim_ellipse(P, V, vis):
    """Rim radius per ray searched over [0.6, 1.5] r, fitted by a 2nd-order Fourier series
    rho(theta) (covers an off-centre ellipse = oblique view of a round mark). Returns the robust
    residual (fraction of r) and the fraction of rays whose rim agrees with the fit."""
    idx, st, _, rv = rim(P, V, 0.6, 1.5)
    if rv.sum() < 12:
        return 1.0, 0.0
    th = (np.arange(NA) + 0.5) * 2 * np.pi / NA
    A = np.stack([np.ones(NA), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], 1)
    rho = (idx + 0.5) / SPR
    w = np.abs(st) * rv
    for _ in range(3):                                          # IRLS with Cauchy-like weights
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(A * sw[:, None], rho * sw, rcond=None)
        res = np.abs(rho - A @ c)
        w = np.abs(st) * rv / (1.0 + (res / 0.05) ** 2)
    res = np.abs(rho - A @ c)[rv]
    ww = np.abs(st)[rv]
    o = np.argsort(res)
    cw = np.cumsum(ww[o]) / ww.sum()
    med = float(res[o][np.searchsorted(cw, 0.5)])               # weighted median residual
    return med, float((res < 0.06).mean() * vis + (1 - vis) * 0.0)


def circle_features(pyr, x, y, r, noise):
    """Features of the fitted circle, using only samples inside the photo. None if < 12 rays."""
    P, V = pyr.polar(x, y, r)
    idx, st, sharp, rv = rim(P, V)
    if rv.sum() < 12:
        return None
    rr = (np.arange(P.shape[1]) + 0.5) / SPR
    mi, mr = (rr < 0.8)[None] & V, ((rr > 1.25) & (rr < 1.9))[None] & V
    if mi.sum() < 20 or mr.sum() < 20:
        return None
    in_mean = float(P[mi].mean())
    ring = P[mr]
    ring_med = float(np.median(ring))
    ring_mad = float(np.median(np.abs(ring - ring_med))) * 1.4826
    contrast = ring_med - in_mean
    # rays whose outer ring is as dark as the inside (elongated dark structures, dark halos)
    cnt = mr.sum(1)
    ray_ring = np.where(mr, P, 0.0).sum(1) / np.maximum(cnt, 1)
    okr = cnt > 0
    ring_dark = float((ray_ring[okr] < in_mean + 0.5 * contrast).mean()) if contrast > 0 else 1.0
    a = np.abs(st[rv])
    tex = max(ring_mad, noise)
    strong = a >= np.median(a)
    step_q = float(np.quantile(a, 0.7))
    coverage = float((a > max(2.0 * tex, 0.025)).mean())
    rho_rel = (idx[rv] + 0.5) / SPR
    rad_cons = float(np.median(np.abs(rho_rel[strong] - np.median(rho_rel[strong]))))
    ell_res, ecov = rim_ellipse(P, V, rv.mean())
    sharp_med = float(np.median(sharp[rv][strong]))
    thr = in_mean + 0.5 * contrast
    circ, solid, aratio, ell, axr = blob_shape(pyr, x, y, r, thr) if contrast > 0 else (0.0,) * 5
    return dict(x=x, y=y, r=r, contrast=contrast, tex=tex, ring_dark=ring_dark, step=step_q,
                coverage=coverage, rad_cons=rad_cons, sharp=sharp_med, circ=circ, solid=solid,
                aratio=aratio, ell=ell, axr=axr, ell_res=ell_res, ecov=ecov, zc=contrast / tex, zs=step_q / tex, vis=float(rv.mean()),
                sup=float(a.sum() / rv.mean()))


def verify(pyr, x, y, r, noise):
    """Fit the rim around the LoG blob and compute features (None if mostly outside the photo)."""
    x, y, r, pts = fit_rim(pyr, x, y, r)
    f = circle_features(pyr, x, y, r, noise)
    if f is None:
        return None
    f["pts"] = pts
    f["r_in"] = r
    f["r_out"] = outer_rim(pyr, x, y, r, f["sup"])
    return f


def score(f, logr):
    """Combine features into a [0,1] score: weighted geometric mean of soft ramps (a soft AND),
    times a size gate. Blob-shape terms are skipped when the circle is cut by the photo border."""
    whole = f["vis"] >= 0.95
    terms = [
        (ramp(logr, 0.03, 0.15), 1.0),                 # LoG dark-blob response
        (ramp(f["contrast"], 0.04, 0.18), 1.5),        # radial contrast: inside darker than ring
        (ramp(f["zc"], 2.0, 6.0), 1.0),                # ... relative to the ring texture
        (ramp(f["coverage"], 0.3, 0.75), 1.5),         # rim edge present all around
        (max(ramp(f["rad_cons"], 0.12, 0.04),          # rim at a consistent radius (circle)
             ramp(f["ell_res"], 0.06, 0.025)), 1.0),   # ... or on a smooth ellipse (oblique)
        (ramp(f["zs"], 2.0, 5.5), 1.0),                # rim step relative to texture
        (ramp(f["sharp"], 1.0, 1.3), 0.5),             # edge sharpness (peak slope vs step)
        (ramp(abs(math.log(max(f["aratio"], 1e-3))), 0.8, 0.35), 1.5),  # blob fills the circle
        (ramp(f["ring_dark"], 0.35, 0.1), 1.0),        # no dark band leaving the circle
    ]
    if whole:
        terms += [
            (ramp(f["circ"], 0.4, 0.7), 1.0),          # thresholded blob circularity
            (ramp(f["solid"], 0.75, 0.9), 0.5),        # ... solidity
            (ramp(f["ell"], 0.7, 0.85), 1.0),          # ... matches an ellipse (oblique view)
        ]
    ws = sum(w for _, w in terms)
    gm = math.exp(sum(w * math.log(max(v, 0.02)) for v, w in terms) / ws)
    # size gate: below ~r=20 (half-res) only near-perfect ellipses survive (pores / bug holes)
    big = ramp(f["r"], 12.0, 20.0)
    small_ok = ramp(f["ell"], 0.86, 0.95) * ramp(f["solid"], 0.9, 0.96) if whole else 0.0
    gate = ramp(f["r"], 8.0, 11.0) * max(big, small_ok)
    # circles cut by the photo border: need >~half the rim inside, unless clearly dark inside
    # (photo edges are often darker -> half-blobs that the LoG mirror padding completes)
    border = max(ramp(f["vis"], 0.4, 0.65), ramp(f["contrast"], 0.25, 0.4) * ramp(f["vis"], 0.2, 0.35))
    colour = ramp(f["warm"], 7.5, 4.5)                 # warm-coloured blobs: rust, pipes, spalls
    return float(gm * gate * border * colour)


# ---- per photo --------------------------------------------------------------------------------
def detect(photo, debug=False):
    grey, bstar, valid = load(photo)
    N0 = normalise(grey, valid)
    pyr = Pyramid(N0)
    N1 = pyr.lv[1]                                             # quarter res
    hp = N1 - cv2.GaussianBlur(N1, (0, 0), 2.0)
    noise = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826
    cand = log_candidates(N1, 2)
    cand = nms(cand, 0.8)[:MAX_CAND]
    vmask = valid
    out = []
    for x, y, r, resp in cand:
        f = verify(pyr, x, y, r, noise)
        if f is None:
            continue
        # reject circles that reach into missing (black) cells
        yi, xi = int(np.clip(f["y"], 0, vmask.shape[0] - 1)), int(np.clip(f["x"], 0, vmask.shape[1] - 1))
        rr = int(f["r"] * 1.3)
        sub = vmask[max(yi - rr, 0):yi + rr + 1, max(xi - rr, 0):xi + rr + 1]
        if sub.size and sub.mean() < 0.98:
            continue
        f["log"] = float(resp)
        f["warm"] = warmth(bstar, f["x"], f["y"], f["r"])
        f["score"] = score(f, resp)
        # reported radius: cone mouth if found; a dark blob larger than the fitted rim (oblique /
        # small marks whose rim fit locked on the bottom) widens it a little
        f["r"] = max(f["r_out"], f["r_in"] * min(math.sqrt(max(f["aratio"], 1.0)), 1.3))
        out.append(f)
    if not out:
        return []
    arr = np.array([[f["x"], f["y"], f["r"], f["score"]] for f in out])
    keep = nms(arr, 0.7)
    keyset = {(round(a[0], 3), round(a[1], 3)) for a in keep}
    out = [f for f in out if (round(f["x"], 3), round(f["y"], 3)) in keyset]
    out.sort(key=lambda f: -f["score"])
    return out


# ---- crop table -------------------------------------------------------------------------------
def disk_square_area(x, y, r, sx, sy, size=T):
    """Area (px^2) of the disk inside the square [sx, sx+size) x [sy, sy+size), pixel-centre count."""
    xa, xb = max(sx, int(math.floor(x - r))), min(sx + size, int(math.ceil(x + r)) + 1)
    ya, yb = max(sy, int(math.floor(y - r))), min(sy + size, int(math.ceil(y + r)) + 1)
    if xa >= xb or ya >= yb:
        return 0
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return int((((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) <= r * r).sum())


def crop_scores(dets):
    """photo -> (ROWS, COLS, 2) array of (best score, overlap px)."""
    res = {}
    for photo, lst in dets.items():
        A = np.zeros((ROWS, COLS, 2))
        for f in lst:
            x, y, r, s = f["x"], f["y"], f["r"], f["score"]
            c0, c1 = int(max((x - r) // T, 0)), int(min((x + r) // T, COLS - 1))
            r0, r1 = int(max((y - r) // T, 0)), int(min((y + r) // T, ROWS - 1))
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    ov = disk_square_area(x, y, r, cc * T, rr * T)
                    if ov > 0 and s > A[rr, cc, 0]:
                        A[rr, cc] = (s, ov)
        res[photo] = A
    return res


def draw(photo, lst, path):
    M = np.load(MOSAIC / f"{photo}.npy")
    img = cv2.cvtColor(cv2.resize(M, (1152, 896), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    for f in sorted(lst, key=lambda f: f["score"]):
        if f["score"] < 0.15:
            continue
        col = (0, 0, 255) if f["score"] >= THRESH else (0, 200, 255) if f["score"] >= 0.3 else (255, 160, 0)
        c = (int(round(f["x"] / 2)), int(round(f["y"] / 2)))
        cv2.circle(img, c, max(int(round(f["r"] / 2)), 2), col, 2)
        cv2.putText(img, f"{f['score']:.2f}", (c[0] + int(f["r"] / 2) + 2, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "viz").mkdir(exist_ok=True)
    photos = sorted(p.stem for p in MOSAIC.glob("*.npy") if not p.stem.endswith("_filled"))
    dets = {}
    for i, p in enumerate(photos):
        dets[p] = detect(p)
    with open(OUT / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "x", "y", "r", "score"])
        for p in photos:
            for f in dets[p]:
                if f["score"] >= KEEP_MIN_SCORE:
                    w.writerow([p, f"{f['x']:.1f}", f"{f['y']:.1f}", f"{f['r']:.1f}", f"{f['score']:.4f}"])
    # debug feature dump
    with open(OUT / "features.csv", "w", newline="") as fh:
        keys = ["x", "y", "r", "r_in", "log", "contrast", "tex", "zc", "step", "zs", "coverage",
                "rad_cons", "sharp", "circ", "solid", "aratio", "ell", "axr", "ell_res", "ecov", "ring_dark", "vis", "warm", "score"]
        w = csv.writer(fh)
        w.writerow(["photo"] + keys)
        for p in photos:
            for f in dets[p]:
                w.writerow([p] + [f"{f[k]:.4f}" for k in keys])
    cs = crop_scores({p: [f for f in dets[p] if f["score"] >= KEEP_MIN_SCORE] for p in photos})
    rows = []
    for f in sorted((SDNET / "W").glob("*/*.jpg")):
        m = NAME.match(f.stem)
        if not m:
            continue
        p, idx = m.group(1), int(m.group(2))
        rr, cc = divmod(idx - 1, COLS)
        s, ov = cs[p][rr, cc] if p in cs and 0 <= rr < ROWS else (0.0, 0)
        rows.append((f"W/{f.parent.name}/{f.name}", s, int(ov)))
    with open(OUT / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_path", "score", "overlap_px"])
        for rp, s, ov in rows:
            w.writerow([rp, f"{s:.4f}", ov])
    for p in photos:
        draw(p, dets[p], OUT / "viz" / f"{p}.jpg")
    n_pos = sum(1 for _, s, _ in rows if s >= THRESH)
    n_det = sum(1 for p in photos for f in dets[p] if f["score"] >= THRESH)
    zero = sum(1 for p in photos if not any(f["score"] >= THRESH for f in dets[p]))
    print(f"photos={len(photos)} crops={len(rows)} det>=thr={n_det} crops>=thr={n_pos} "
          f"photos_without_det={zero} time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

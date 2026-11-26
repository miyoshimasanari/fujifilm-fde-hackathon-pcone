"""Label-free P-cone mark detector: Hough circle candidates + edge-support verification.

Run:  .venv/bin/python pcone/detect_hough.py          (all 72 mosaics, ~1.5 min on 10 cores)
      .venv/bin/python pcone/detect_hough.py --one 7069 7131   (debug print for some photos)

Pipeline (identical for every photo, no per-photo tuning, no labels, no pretrained weights):
  1. Grey level of the half-res mosaic minus a masked large-scale background (sigma 40 px), in units of
     the photo's robust pixel noise (MAD). Missing crop cells are masked out.
  2. Candidates from cv2.HoughCircles on a 3-level pyramid (1/1, 1/2, 1/4 of half-res) so each level
     only searches a narrow radius band (marks span r ~ 8..90 half-res px: camera distance/obliquity vary).
       - step channel: denoised normalised grey (open holes = dark recess with a sharp rim)
       - ring channel: black-hat image (mortar-filled marks = thin dark joint ring, weak darkness)
     Images are zero-padded so marks cut by the photo border can still get a centre.
  3. Each candidate is verified in polar coordinates and refined to an ellipse (oblique views make marks
     elliptic) by iterated per-angle edge picking + cv2.fitEllipse (circle fit for partly visible arcs):
       - step mode: radial derivative of the smoothed image (~ step height in noise sigmas)
       - ring mode: radial valley response, normalised by the photo's own level on random circles
     Edge support (cov) = fraction of the visible circumference whose strongest edge lies on the fitted
     ellipse (step: within max(1px, 6% r); ring: within 1.5 px). Interior/exterior contrast is measured.
  4. Scores (0..1):
       open hole   = ramp(cov) * ramp(edge strength) * ramp(interior darker than surround) * visibility
       filled mark = 0.85 * ramp(cov) * ramp(ring strength) * ramp(r >= 14..22) * (no long gap) * visibility
       (ring mode needs no interior darkness: that is how mortar-filled marks are caught)
     x size prior ramp(r, 9..16 px) against bug holes; relaxed for detections that look like the photo's
     confident marks (same foreshortening, >= ~0.5x their size) because far/oblique photos have small marks.
  5. Greedy NMS (marks never overlap), then crop scores: max score over detections whose ellipse
     intersects the crop's half-res square, plus that detection's visible area inside the square.

Outputs (data/detect/hough/): detections.csv (photo,x,y,r,score; r = sqrt(a*b) of the ellipse),
detections_full.csv (all fitted ellipse parameters and score components), crops.csv
(rel_path,score,overlap_px; one row per crop jpg on disk), viz/<photo>.jpg overlays.
"""
import csv
import glob
import os
import re
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MOSAIC = os.path.join(ROOT, "data", "mosaic_half")
SDNET = os.path.join(ROOT, "data", "sdnet")
OUT = os.path.join(ROOT, "data", "detect", "hough")
VIZ = os.path.join(OUT, "viz")

COLS, ROWS, T = 18, 14, 128          # grid and half-res tile size
DEFAULT_THRESHOLD = 0.5              # chosen by inspecting score-ranked galleries (no labels)
PAD = 64                             # zero padding so centres may lie just outside the photo
NTH = 72                             # angular samples for verification
TH = np.linspace(0, 2 * np.pi, NTH, endpoint=False).astype(np.float32)
COS, SIN = np.cos(TH), np.sin(TH)
AR = np.arange(NTH)
VIZ_PHOTOS = ["7069", "7099", "7104", "7111", "7115", "7122", "7131", "7136", "7140", "7128", "7090", "7102"]


# ----------------------------------------------------------------------------- preprocessing
def load(pid):
    rgb = np.load(os.path.join(MOSAIC, f"{pid}.npy"))
    filled = np.load(os.path.join(MOSAIC, f"{pid}_filled.npy"))
    valid = np.kron(filled.astype(np.float32), np.ones((T, T), np.float32))
    return rgb, valid


def normalise(rgb, valid):
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    # masked large-scale background (sigma 40 px), computed at 1/8 resolution for speed
    H, W = g.shape
    gs = cv2.resize(g * valid, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(gs, (0, 0), 5)
    den = cv2.GaussianBlur(vs, (0, 0), 5)
    bg = cv2.resize(num / np.maximum(den, 1e-3), (W, H), interpolation=cv2.INTER_LINEAR)
    hp = (g - bg) * valid
    v = hp[valid > 0]
    noise = 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-3
    return (hp / noise).astype(np.float32), noise


# ----------------------------------------------------------------------------- candidates
# Step channel: (downscale f, rmin, rmax at that level, [(method, dp, param1, param2), ...]).
# Classic HOUGH_GRADIENT with a high Canny threshold copes with oblique (elliptic) marks; the ALT
# variant adds accurate seeds for round marks at the coarser levels where it is cheap.
G, A = cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT_ALT
STEP_LEVELS = [(1, 7, 24, [(G, 1.5, 300, 10), (G, 1, 300, 10)]),
               (2, 5, 26, [(G, 1, 150, 10), (A, 1.5, 150, 0.5)]),
               (4, 10, 26, [(G, 1, 150, 10), (A, 1.5, 150, 0.5)])]
# Ring channel (black-hat image, thin dark joint rings of mortar-filled marks): r 12..68 half-res.
RING_LEVELS = [(1, 12, 24, [(G, 1, 80, 22)]),
               (2, 12, 26, [(G, 1, 80, 18), (A, 1.5, 120, 0.4)]),
               (4, 13, 17, [(G, 1, 80, 18), (A, 1.5, 120, 0.4)])]


def _hough(u8, f, rmin, rmax, specs, min_dist=None):
    out = []
    for meth, dp, p1, p2 in specs:
        c = cv2.HoughCircles(u8, meth, dp=dp, minDist=min_dist or max(2, rmin // 2), param1=p1, param2=p2,
                             minRadius=rmin, maxRadius=rmax)
        if c is not None:
            for x, y, r in c[0]:
                out.append((x * f - PAD, y * f - PAD, r * f))
    return out


def dedupe(seeds):
    """One seed per (x, y, log r) cell: Hough returns many near-identical circles."""
    seen, out = set(), []
    for x, y, r in seeds:
        q = 0.3 * r + 2.0
        k = (int(round(x / q)), int(round(y / q)), int(round(np.log(max(r, 1.0)) / 0.25)))
        if k not in seen:
            seen.add(k)
            out.append((x, y, r))
    return np.array(out, np.float32).reshape(-1, 3)


def candidates(n):
    npad = cv2.copyMakeBorder(n, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    pyr = {f: npad if f == 1 else cv2.resize(npad, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
           for f in {lv[0] for lv in STEP_LEVELS + RING_LEVELS}}
    step, ring = [], []
    for f, rmin, rmax, specs in STEP_LEVELS:
        img = pyr[f]
        s = cv2.GaussianBlur(img, (0, 0), 1.5)                       # denoise
        u8 = np.clip(128 + 20 * s, 0, 255).astype(np.uint8)          # 20 grey levels per noise sigma
        step += _hough(u8, f, rmin, rmax, specs)
    for f, rmin, rmax, specs in RING_LEVELS:
        img = pyr[f]
        s = cv2.GaussianBlur(img, (0, 0), 1.0)
        bh = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - s
        bh = cv2.GaussianBlur(bh, (0, 0), 1.2)
        u8 = np.clip(bh / (np.median(bh) + 1e-3) * 25, 0, 255).astype(np.uint8)
        ring += _hough(u8, f, rmin, rmax, specs, min_dist=rmin)
    return dedupe(step), dedupe(ring)


# ----------------------------------------------------------------------------- verification
def ell_rho(a, b, phi):
    c = np.cos(TH - phi)
    s = np.sin(TH - phi)
    return 1.0 / np.sqrt((c / a) ** 2 + (s / b) ** 2)


def fit_circle(px, py):
    A = np.stack([px, py, np.ones_like(px)], 1)
    rhs = px ** 2 + py ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx ** 2 + cy ** 2
    return cx, cy, np.sqrt(max(r2, 1e-6))


class Verifier:
    SIG = [1.0, 1.8, 3.0]

    def __init__(self, n, valid, seed=0):
        self.H, self.W = n.shape
        self.sm = [cv2.GaussianBlur(n, (0, 0), s) for s in self.SIG]
        self.sv = cv2.GaussianBlur(n, (0, 0), 1.2)          # for the valley (ring) response
        self.valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        self.v0 = 1.0
        self.v0 = self._valley_null(seed)

    def polar(self, img, cx, cy, rho):
        X = (cx + rho[None, :] * COS[:, None]).astype(np.float32)
        Y = (cy + rho[None, :] * SIN[:, None]).astype(np.float32)
        return cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def evidence(self, mode, cx, cy, r, rmax, rmin=0.0):
        dr = 0.5 if rmax < 40 else 0.75
        rmin = max(0.0, rmin)
        rho = (rmin + np.arange(int((rmax - rmin) / dr) + 1) * dr).astype(np.float32)
        V = self.polar(self.valid, cx, cy, rho)
        if mode == "step":
            si = 0 if r < 14 else (1 if r < 36 else 2)
            sig = self.SIG[si]
            P = self.polar(self.sm[si], cx, cy, rho)
            S = np.zeros_like(P)
            S[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * dr) * (2.5 * sig)   # ~ step height in noise units
            E = np.abs(S)
        else:
            P = self.polar(self.sv, cx, cy, rho)
            w = max(2, int(round(max(2.5, 0.05 * r) / dr)))
            S = np.zeros_like(P)
            S[:, w:-w] = 0.5 * (P[:, :-2 * w] + P[:, 2 * w:]) - P[:, w:-w]   # dark thin line > 0
            E = np.maximum(S, 0) / self.v0
        E[V < 0.99] = 0
        return rho, V, S, E

    def _valley_null(self, seed):
        """Typical per-angle valley maximum on random circles -> normalises the ring evidence."""
        rs = np.random.RandomState(seed)
        vals = []
        for _ in range(120):
            r = float(rs.uniform(15, 50))
            cx, cy = rs.uniform(r, self.W - r), rs.uniform(r, self.H - r)
            rho, V, S, E = self.evidence("ring", cx, cy, r, 1.3 * r + 4, 0.7 * r - 4)
            m = (np.abs(rho - r) <= 1.5)
            ok = V[:, m].min(1) > 0.99
            if ok.sum() > NTH // 2:
                vals.append(np.median(np.maximum(S[ok][:, m], 0).max(1)))
        return float(np.median(vals)) if vals else 1.0

    def verify(self, mode, x0, y0, r0):
        T_ = 1.5 if mode == "step" else 2.0
        cx, cy, a, b, phi = float(x0), float(y0), float(r0), float(r0), 0.0
        band = 0.35
        for it in range(5):
            rho, V, S, E = self.evidence(mode, cx, cy, np.sqrt(a * b), (1 + band) * max(a, b) + 6,
                                         (1 - band) * min(a, b) - 6)
            rhoe = ell_rho(a, b, phi)
            lo, hi = rhoe * (1 - band) - 1, rhoe * (1 + band) + 1
            m = (rho[None, :] >= lo[:, None]) & (rho[None, :] <= hi[:, None])
            m[:, :2] = False
            m[:, -2:] = False
            j = np.where(m, E, -1.0).argmax(1)
            e = E[AR, j]
            okang = m.any(1) & (V[AR, j] > 0.99)
            jm = np.clip(j, 1, len(rho) - 2)
            e0, e1, e2 = E[AR, jm - 1], E[AR, jm], E[AR, jm + 1]
            den = e0 - 2 * e1 + e2
            with np.errstate(divide="ignore", invalid="ignore"):
                off = np.where(np.abs(den) > 1e-6, 0.5 * (e0 - e2) / den, 0.0)
            rs = rho[jm] + np.clip(np.nan_to_num(off), -1, 1) * (rho[1] - rho[0])
            strong = okang & (e >= T_)
            vfrac = okang.sum() / NTH
            if strong.sum() < 0.25 * NTH:     # hopeless: not enough edge evidence around the circle
                return None
            if it == 4:
                break
            px = (cx + rs[strong] * COS[strong]).astype(np.float64)
            py = (cy + rs[strong] * SIN[strong]).astype(np.float64)
            if vfrac >= 0.6 and strong.sum() >= 12:
                try:
                    (ecx, ecy), (d1, d2), ang = cv2.fitEllipse(np.stack([px, py], 1).astype(np.float32))
                except cv2.error:
                    break
                ea, eb, ephi = d1 / 2, d2 / 2, np.deg2rad(ang)
            else:
                ecx, ecy, ea = fit_circle(px, py)
                eb, ephi = ea, 0.0
            if not (np.isfinite(ea) and np.isfinite(eb) and np.isfinite(ecx) and np.isfinite(ecy)) \
                    or min(ea, eb) < 3.5 or max(ea, eb) / max(min(ea, eb), 1e-3) > 3.0 \
                    or max(ea, eb) > 1.6 * max(a, b) or min(ea, eb) < 0.5 * min(a, b) \
                    or np.hypot(ecx - cx, ecy - cy) > 0.6 * max(a, b):
                break
            cx, cy, a, b, phi = ecx, ecy, ea, eb, ephi
            band = max(0.10, band * 0.6)
        rhoe = ell_rho(a, b, phi)
        # step edges: 6% of r; thin rings must sit on the fitted ellipse much more precisely
        tol = np.maximum(1.0, 0.06 * rhoe) if mode == "step" else np.full(NTH, 1.5, np.float32)
        inl = okang & (np.abs(rs - rhoe) <= tol) & (e >= T_)
        nvalid = int(okang.sum())
        cov = inl.sum() / max(nvalid, 1)
        mis = (okang & (np.abs(rs - rhoe) > tol) & (e >= T_)).sum() / max(nvalid, 1)   # strong edge, off the ellipse
        strength = float(np.median(e[inl])) if inl.any() else 0.0
        gap = longest_gap(inl[okang]) / NTH
        # interior / exterior contrast (light smoothing)
        rho = (np.arange(int(1.75 * max(a, b) / 0.75) + 2) * 0.75).astype(np.float32)
        V = self.polar(self.valid, cx, cy, rho)
        P0 = self.polar(self.sm[0], cx, cy, rho)
        ratio = rho[None, :] / rhoe[:, None]
        vin = (ratio < 0.8) & (V > 0.99)
        vout = (ratio > 1.25) & (ratio < 1.7) & (V > 0.99)
        vann = (ratio > 1.12) & (ratio < 1.5) & (V > 0.99)
        if vin.sum() > 5 and vout.sum() > 5 and vann.sum() > 5:
            ext = float(np.median(P0[vout]))
            pin = P0[vin]
            dark = ext - float(np.percentile(pin, 25))           # shadow / dark part of the interior
            tone = ext - float(np.median(pin))                    # whole-interior tone difference
            fin = float(np.mean(pin < ext - 3.0))                 # dark fraction inside
            fout = float(np.mean(P0[vann] < ext - 3.0))           # dark fraction just outside (irregular voids, cracks)
        else:
            dark = tone = fin = fout = 0.0
        return dict(mode=mode, x=cx, y=cy, a=float(a), b=float(b), phi=float(phi), r=float(np.sqrt(a * b)),
                    cov=float(cov), mis=float(mis), strength=strength, gap=float(gap), dark=dark, tone=tone, fin=fin, fout=fout,
                    vfrac=float(vfrac))


def longest_gap(inl):
    """Longest circular run of False in a boolean array."""
    if inl.size == 0:
        return 0
    if not inl.any():
        return inl.size
    k = int(np.argmax(inl))
    x = np.roll(inl, -k)                      # starts with True
    idx = np.flatnonzero(x)
    gaps = np.diff(np.concatenate([idx, [x.size]])) - 1
    return int(gaps.max()) if gaps.size else 0


# ----------------------------------------------------------------------------- scoring
def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def base_score(d):
    """Evidence that the fitted ellipse is a P-cone mark, before the size prior (0..1)."""
    a, b = max(d["a"], d["b"]), min(d["a"], d["b"])
    if d["vfrac"] < 0.3 or a / max(b, 1e-3) > 3.0:
        return 0.0
    f_vis = ramp(d["vfrac"], 0.45, 0.85)          # marks cut by photo border / missing cells need a long visible arc
    f_vis *= 1.0 - ramp(a / max(b, 1e-3), 2.3, 3.0)  # even oblique marks stay below ~2.3:1; thinner = joints/strips
    if d["mode"] == "step":
        # open hole: well supported elliptical step edge AND a darker interior (shadow / recess);
        # the interior test rejects ellipses wrapped around cracks or panel joints.
        q_cov = ramp(d["cov"], 0.45, 0.9)
        q_str = ramp(d["strength"], 1.5, 6.0)
        q_con = ramp(d["tone"], 1.0, 4.0) * ramp(d["fin"], 0.2, 0.5)
        return q_cov * q_str * q_con * f_vis
    # mortar-filled mark: thin dark joint ring, little darkness inside -> demand a near-complete, compact,
    # precisely elliptical ring of reasonable size; capped below the open-hole maximum.
    if a / max(b, 1e-3) > 1.8:
        return 0.0
    return 0.85 * ramp(d["cov"], 0.6, 0.85) * ramp(d["strength"], 2.5, 5.0) * ramp(d["r"], 14, 22) \
        * (1.0 - ramp(d["gap"], 0.12, 0.3)) * f_vis


def shape(d):
    """(axis ratio >= 1, major-axis orientation in [0, pi))."""
    if d["a"] >= d["b"]:
        return d["a"] / max(d["b"], 1e-3), d["phi"] % np.pi
    return d["b"] / max(d["a"], 1e-3), (d["phi"] + np.pi / 2) % np.pi


def size_prior(d, ref):
    """Bug holes are small: ramp on the mean radius. Far / oblique photos have small marks, so the ramp is
    relaxed for clean detections that look like the photo's confident marks (same foreshortening: similar
    axis ratio and orientation) and are at least ~half the size of the smallest of them."""
    f = ramp(d["r"], 9.0, 16.0)
    if ref is not None and d["base"] >= 0.6 and f < 1.0:
        r_ref, q_ref, t_ref = ref
        q, t = shape(d)
        similar = abs(np.log(q / q_ref)) <= 0.45
        if similar and q_ref >= 1.3 and q >= 1.3:
            dt = abs(t - t_ref) % np.pi
            similar = min(dt, np.pi - dt) <= np.deg2rad(40)   # perspective rotates the ellipses across the frame
        if similar:
            f = max(f, ramp(d["r"] / r_ref, 0.42, 0.55))
    if d["r"] > 110:
        f = 0.0
    return f


def score_photo(dets):
    for d in dets:
        d["base"] = base_score(d)
    pre = nms([dict(d, score=d["base"] * ramp(d["r"], 9.0, 16.0)) for d in dets if d["base"] > 0.02])
    conf = [d for d in pre if d["mode"] == "step" and d["score"] >= 0.8 and d["r"] >= 14]
    ref = None
    if conf:
        q = np.array([shape(d)[0] for d in conf])
        t = np.array([shape(d)[1] for d in conf])
        t_ref = (np.angle(np.sum((q - 1) * np.exp(2j * t))) / 2) % np.pi   # eccentricity-weighted axial mean
        ref = (float(min(d["r"] for d in conf)), float(np.median(q)), float(t_ref))
    for d in dets:
        d["size_f"] = size_prior(d, ref)
        d["score"] = d["base"] * d["size_f"]
    return nms([d for d in dets if d["score"] > 0.02]), ref


def nms(dets):
    """Greedy NMS: real marks never overlap, so a detection whose disk overlaps a better one is dropped.
    If a dropped fit is nearly as good, larger and contains the kept centre (rim fit vs. inner shadow fit of
    the same hole), the kept detection takes over its geometry so that the reported disk covers the whole mark."""
    dets = sorted(dets, key=lambda d: -d["score"])
    keep, kx, ky, kr = [], [], [], []
    for d in dets:
        if kx:
            dd = np.hypot(np.asarray(kx) - d["x"], np.asarray(ky) - d["y"])
            hit = np.flatnonzero(dd < 0.9 * (np.asarray(kr) + d["r"]))
            if hit.size:
                i = int(hit[0])
                k = keep[i]
                if (not k.get("grown") and d["score"] >= 0.85 * k["score"] and k["r"] < d["r"] <= 1.6 * k["r"]
                        and dd[i] < d["r"] and d["mode"] == k["mode"]):
                    for f in ("x", "y", "a", "b", "phi", "r"):
                        k[f] = d[f]
                    k["grown"] = True
                    kx[i], ky[i], kr[i] = d["x"], d["y"], d["r"]
                continue
        keep.append(d); kx.append(d["x"]); ky.append(d["y"]); kr.append(d["r"])
    return keep


def detect(pid):
    cv2.setNumThreads(1)
    rgb, valid = load(pid)
    n, noise = normalise(rgb, valid)
    cs, cr = candidates(n)
    ver = Verifier(n, valid, seed=int(pid) if pid.isdigit() else 0)
    dets = []
    for mode, C in (("step", cs), ("ring", cr)):
        for x, y, r in C:
            d = ver.verify(mode, x, y, r)
            if d is not None and d["cov"] > 0.3:
                dets.append(d)
    keep, r_ref = score_photo(dets)
    for d in keep:
        d["photo"] = pid
    return pid, keep, len(cs), len(cr), r_ref


# ----------------------------------------------------------------------------- crops
NAME = re.compile(r"^(\d+)-(\d+)(?:_(\d+))?$")


def ellipse_mask_area(d, x0, y0, size=T):
    """Area (px^2) of detection ellipse inside the square [x0,x0+size) x [y0,y0+size)."""
    ys, xs = np.mgrid[y0:y0 + size, x0:x0 + size].astype(np.float32) + 0.5
    dx, dy = xs - d["x"], ys - d["y"]
    c, s = np.cos(d["phi"]), np.sin(d["phi"])
    u = (dx * c + dy * s) / d["a"]
    v = (-dx * s + dy * c) / d["b"]
    return int(((u * u + v * v) <= 1.0).sum())


def crop_scores(dets_by_photo):
    rows = []
    files = sorted(glob.glob(os.path.join(SDNET, "W", "CW", "*.jpg")) + glob.glob(os.path.join(SDNET, "W", "UW", "*.jpg")))
    cache = {}
    for f in files:
        rel = os.path.relpath(f, SDNET).replace(os.sep, "/")
        m = NAME.match(os.path.splitext(os.path.basename(f))[0])
        best, area = 0.0, 0
        if m:
            pid, idx = m.group(1), int(m.group(2))
            if 1 <= idx <= COLS * ROWS:
                key = (pid, idx)
                if key not in cache:
                    r_, c_ = divmod(idx - 1, COLS)
                    x0, y0 = c_ * T, r_ * T
                    b_, a_ = 0.0, 0
                    for d in dets_by_photo.get(pid, []):
                        R = max(d["a"], d["b"])
                        if d["x"] + R < x0 or d["x"] - R >= x0 + T or d["y"] + R < y0 or d["y"] - R >= y0 + T:
                            continue
                        if d["score"] <= b_:
                            continue
                        ar = ellipse_mask_area(d, x0, y0)
                        if ar > 0:
                            b_, a_ = d["score"], ar
                    cache[key] = (b_, a_)
                best, area = cache[key]
        rows.append((rel, best, area))
    return rows


# ----------------------------------------------------------------------------- viz
def draw(pid, dets, thr):
    rgb, valid = load(pid)
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for r in range(1, ROWS):
        cv2.line(vis, (0, r * T), (COLS * T, r * T), (200, 200, 200), 1)
    for c in range(1, COLS):
        cv2.line(vis, (c * T, 0), (c * T, ROWS * T), (200, 200, 200), 1)
    for d in sorted(dets, key=lambda d: d["score"]):
        if d["score"] < 0.15:
            continue
        col = (0, 0, 255) if d["score"] >= thr else (0, 190, 255)
        if d["mode"] == "ring":
            col = (255, 0, 255) if d["score"] >= thr else (255, 160, 0)
        cv2.ellipse(vis, (int(round(d["x"])), int(round(d["y"]))), (max(1, int(round(d["a"]))), max(1, int(round(d["b"])))),
                    np.rad2deg(d["phi"]), 0, 360, col, 3)
        tx, ty = int(d["x"] + max(d["a"], d["b"]) + 4), int(d["y"] + 8)
        cv2.putText(vis, f"{d['score']:.2f}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
        cv2.putText(vis, f"{d['score']:.2f}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3)
    vis = cv2.resize(vis, (1152, 896), interpolation=cv2.INTER_AREA)
    cv2.putText(vis, f"{pid}  thr={thr:.2f}  red/magenta>=thr (step/ring), orange/blue below", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(VIZ, f"{pid}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(VIZ, exist_ok=True)
    pids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(MOSAIC, "*.npy")) if not f.endswith("_filled.npy"))
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        res = pool.map(detect, pids, chunksize=1)
    dets_by_photo = {pid: keep for pid, keep, _, _, _ in res}
    ncs = sum(r[2] for r in res)
    ncr = sum(r[3] for r in res)
    with open(os.path.join(OUT, "detections.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "x", "y", "r", "score"])
        for pid in pids:
            for d in sorted(dets_by_photo[pid], key=lambda d: -d["score"]):
                w.writerow([pid, f"{d['x']:.1f}", f"{d['y']:.1f}", f"{d['r']:.1f}", f"{d['score']:.4f}"])
    with open(os.path.join(OUT, "detections_full.csv"), "w", newline="") as fh:
        keys = ["photo", "mode", "x", "y", "a", "b", "phi", "r", "score", "base", "size_f", "cov", "mis", "strength", "gap", "dark", "tone", "fin", "fout", "vfrac"]
        w = csv.writer(fh)
        w.writerow(keys)
        for pid in pids:
            for d in sorted(dets_by_photo[pid], key=lambda d: -d["score"]):
                w.writerow([d[k] if isinstance(d[k], str) else f"{d[k]:.4f}" for k in keys])
    rows = crop_scores(dets_by_photo)
    with open(os.path.join(OUT, "crops.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_path", "score", "overlap_px"])
        for rel, s, a in rows:
            w.writerow([rel, f"{s:.4f}", a])
    for pid in VIZ_PHOTOS:
        if pid in dets_by_photo:
            draw(pid, dets_by_photo[pid], DEFAULT_THRESHOLD)
    thr = DEFAULT_THRESHOLD
    ndet = sum(1 for v in dets_by_photo.values() for d in v if d["score"] >= thr)
    zero = sum(1 for v in dets_by_photo.values() if not any(d["score"] >= thr for d in v))
    npos = sum(1 for _, s, _ in rows if s >= thr)
    print(f"photos={len(pids)} candidates step={ncs} ring={ncr} detections(all)={sum(len(v) for v in dets_by_photo.values())}")
    print(f"threshold={thr}: detections={ndet} photos_with_zero={zero} positive_crops={npos}/{len(rows)}")
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        for pid in sys.argv[2:]:
            t = time.time()
            _, keep, a, b, rref = detect(pid)
            print(pid, f"{time.time() - t:.2f}s cand {a}/{b} r_ref={rref}")
            for d in keep[:12]:
                print("  %-4s %5.0f %5.0f r=%5.1f ab=%.0f/%.0f sc=%.2f base=%.2f sf=%.2f cov=%.2f str=%.1f gap=%.2f dark=%.1f vf=%.2f" % (
                    d["mode"], d["x"], d["y"], d["r"], d["a"], d["b"], d["score"], d["base"], d["size_f"], d["cov"], d["strength"], d["gap"], d["dark"], d["vfrac"]))
    else:
        main()

"""Re-assemble SDNET2018 wall photos from their 256px crops.

Crops are numbered 1..252 row-major on an 18 x 14 grid (verified by edge continuity), so
'W/CW/7069-20.jpg' is row 1, col 1 (0-based) of photo 7069. Mosaics let the classifier see the
neighbourhood of each crop, so a P-cone hole cut by the crop border is still seen whole.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

COLS, ROWS, TILE = 18, 14, 256
NAME = re.compile(r"^(\d+)-(\d+)(?:_(\d+))?$")


def parse(rel_path):
    """-> (photo_id, index, variant) or None."""
    m = NAME.match(Path(rel_path).stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def cell(index):
    r, c = divmod(index - 1, COLS)
    return r, c


def build(root, out_dir, scale=0.5):
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_photo = {}
    for f in sorted(root.glob("W/*/*.jpg")):
        p = parse(f.name)
        if not p:
            continue
        pid, idx, var = p
        if not 1 <= idx <= COLS * ROWS:
            continue
        slot = by_photo.setdefault(pid, {})
        if idx not in slot or var is None:  # prefer the plain name over '_2' variants
            slot[idx] = f
    t = int(round(TILE * scale))
    for pid, slot in by_photo.items():
        M = np.zeros((ROWS * t, COLS * t, 3), np.uint8)
        filled = np.zeros((ROWS, COLS), bool)
        for idx, f in slot.items():
            r, c = cell(idx)
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            if scale != 1:
                img = cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)
            M[r * t:(r + 1) * t, c * t:(c + 1) * t] = img
            filled[r, c] = True
        np.save(out_dir / f"{pid}.npy", M)
        np.save(out_dir / f"{pid}_filled.npy", filled)
    return sorted(by_photo)


if __name__ == "__main__":
    pids = build(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
    print(f"{len(pids)} photos")

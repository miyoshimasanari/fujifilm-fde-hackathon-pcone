"""Load SDNET2018 crops listed in train_label.csv / test.txt into uint8 arrays (cached as .npy)."""
import re
from pathlib import Path

import cv2
import numpy as np


def read_list(path):
    """Rows of a header-less CSV (path[,label]). Tolerates a header line and blank lines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not re.search(r"\.(jpe?g|png)$", parts[0], re.I):
            continue  # header
        rows.append(parts)
    return rows


def photo_id(rel_path):
    """'W/CW/7069-3.jpg' -> 'W/7069' (crops of one source photo share the number before '-')."""
    p = Path(rel_path)
    return f"{p.parts[0]}/{p.stem.split('-')[0]}"


def load_images(root, rel_paths, size, cache=None):
    if cache and Path(cache).exists():
        arr = np.load(cache, mmap_mode=None)
        if len(arr) == len(rel_paths) and arr.shape[1] == size:
            return arr
    out = np.zeros((len(rel_paths), size, size, 3), np.uint8)
    for i, rp in enumerate(rel_paths):
        img = cv2.imread(str(Path(root) / rp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(Path(root) / rp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = img
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out

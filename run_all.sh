#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
"$PY" pcone/mosaic.py data/sdnet data/mosaic_half 0.5
"$PY" pcone/detect_log.py
"$PY" pcone/detect_hough.py
"$PY" pcone/detect_contour.py
"$PY" pcone/detect_combined.py
"$PY" pcone/make_final.py --out submission.csv

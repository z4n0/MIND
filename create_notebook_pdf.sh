#!/usr/bin/env bash

# Export a Jupyter notebook to single-file HTML (images embedded)
# and to PDF. We first render the notebook to self-contained HTML via
# nbconvert, then ask Playwright/Chromium to print that HTML to PDF so
# page layout and colors match the browser view. If Playwright is not
# available, we fall back to nbconvert's WebPDF exporter.

set -euo pipefail

note() { printf '[export] %s\n' "$*"; }
die()  { printf '[export] ERROR: %s\n' "$*" >&2; exit 1; }

NB_PATH=${1:-clinica/explanation.ipynb}

[[ -f "$NB_PATH" ]] || die "Notebook not found: $NB_PATH"

NB_DIR=$(cd "$(dirname "$NB_PATH")" && pwd)
NB_FILE=$(basename "$NB_PATH")
NB_BASE=${NB_FILE%.ipynb}

# Choose nbconvert and python binaries (prefer repo venv if present)
if [[ -x ".venv/bin/jupyter-nbconvert" ]]; then
  NBCONVERT=".venv/bin/jupyter-nbconvert"
else
  NBCONVERT=$(command -v jupyter-nbconvert) || die "jupyter-nbconvert not found"
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN=$(command -v python3) || die "python3 not found"
fi

note "Exporting to HTML with embedded images…"
"$NBCONVERT" \
  --to html \
  --no-input \
  --HTMLExporter.embed_images=True \
  "$NB_PATH" \
  --output "${NB_BASE}.html" \
  --output-dir "$NB_DIR"

OUT_HTML="$NB_DIR/${NB_BASE}.html"
[[ -f "$OUT_HTML" ]] || die "HTML export failed: $OUT_HTML not created"

OUT_PDF="$NB_DIR/${NB_BASE}.pdf"
rm -f "$OUT_PDF"

# Absolute path to HTML (for file:// URL)
if command -v realpath >/dev/null 2>&1; then
  HTML_ABS=$(realpath "$OUT_HTML")
elif command -v readlink >/dev/null 2>&1; then
  HTML_ABS=$(readlink -f "$OUT_HTML")
else
  HTML_ABS="$OUT_HTML"
fi

# Helper: does this python have playwright installed?
if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
import sys
sys.exit(0 if importlib.util.find_spec('playwright') else 1)
PY
then
  note "Rendering PDF via Playwright (Chromium)…"
  if ! "$PYTHON_BIN" - <<'PY' "file://$HTML_ABS" "$OUT_PDF"
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

url, dest = sys.argv[1:3]
dest_path = Path(dest)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")
    page.pdf(
        path=str(dest_path),
        print_background=True,
        format="A4",
        margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        display_header_footer=False,
    )
    browser.close()
PY
  then
    die "Playwright PDF export failed"
  fi
else
  note "Playwright not available; trying nbconvert webpdf (requires playwright install)…"
  set +e
  "$NBCONVERT" \
    --to webpdf \
    --no-input \
    "$NB_PATH" \
    --output "$NB_BASE" \
    --WebPDFExporter.embed_images=True \
    --output-dir "$NB_DIR" >/dev/null 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    die "PDF generation failed. Install playwright (pip install playwright && python -m playwright install chromium)."
  fi
fi

[[ -f "$OUT_PDF" ]] || die "PDF not created: $OUT_PDF"

note "Done.\nHTML: $OUT_HTML\nPDF:  $OUT_PDF"

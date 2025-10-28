#!/usr/bin/env bash

# Export a Jupyter notebook to single-file HTML with embedded images
#
# Usage:
#   ./create_notebook_html.sh [path/to/notebook.ipynb]
#
# If no argument is provided, defaults to clinica/explanation.ipynb.

set -euo pipefail

note() { printf "[HTML export] %s\n" "$*"; }
die()  { printf "[HTML export] ERROR: %s\n" "$*" >&2; exit 1; }

# Default notebook path
NB_PATH=${1:-clinica/explanation.ipynb}

# Validate input
[[ -f "$NB_PATH" ]] || die "Notebook not found: $NB_PATH"

# Resolve paths
NB_DIR=$(cd "$(dirname "$NB_PATH")" && pwd)
NB_FILE=$(basename "$NB_PATH")
NB_BASE=${NB_FILE%.ipynb}

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
  note "Activating virtual environment..."
  # shellcheck disable=SC1091
  source .venv/bin/activate || die "Failed to activate .venv"
fi

# Locate nbconvert (prefer venv-provided jupyter-nbconvert if available)
if command -v jupyter >/dev/null 2>&1; then
  NBCMD=(jupyter nbconvert)
elif command -v jupyter-nbconvert >/dev/null 2>&1; then
  NBCMD=(jupyter-nbconvert)
else
  die "jupyter not found. Activate your venv or install nbconvert."
fi

note "Exporting '$NB_PATH' to HTML with embedded images..."

# Export to HTML with embedded images
"${NBCMD[@]}" \
  --to html \
  --no-input \
  --HTMLExporter.embed_images=True \
  "$NB_PATH" \
  --output "${NB_BASE}.html" \
  --output-dir "$NB_DIR"

OUT_HTML="$NB_DIR/${NB_BASE}.html"

# Verify output
[[ -f "$OUT_HTML" ]] || die "HTML export failed: $OUT_HTML not created"

# Report file size
if command -v du >/dev/null 2>&1; then
  SIZE=$(du -h "$OUT_HTML" | cut -f1)
else
  SIZE="unknown size"
fi

note "✅ Done!"
note "Output: $OUT_HTML ($SIZE)"
note "The HTML file is self-contained with all images embedded."



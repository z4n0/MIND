#!/usr/bin/env python3
"""
Embed local Markdown-linked images into a Jupyter notebook as attachments.

Usage:
    python embed_images_in_notebook.py path/to/notebook.ipynb \
        --output path/to/notebook_embedded.ipynb

If --output is omitted, saves alongside the input as *_embedded.ipynb.
"""

import argparse
import base64
import mimetypes
import re
from pathlib import Path
import nbformat


IMG_MD_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)"
)


def guess_mime(path: Path) -> str | None:
    """Guess MIME type from suffix; return None if unsupported."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    # Accept a few common fallbacks explicitly.
    ext = path.suffix.lower()
    if ext in {".png"}:
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext in {".gif"}:
        return "image/gif"
    return None


def b64encode_file(path: Path) -> str:
    """Read a file and return base64-encoded string."""
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def unique_key(base: str, existing: set[str]) -> str:
    """Generate a unique attachment key avoiding collisions."""
    if base not in existing:
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in existing:
            return candidate
        i += 1


def embed_images(nb_path: Path, out_path: Path | None = None) -> Path:
    """
    Convert Markdown image links that point to local files into attachments.
    Returns the output notebook path.
    """
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)
    nb_dir = nb_path.parent

    total_found = 0
    total_embedded = 0
    total_missing = 0
    total_skipped = 0

    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "markdown":
            continue

        src = cell.get("source", "")
        if not src:
            continue

        # Ensure we preserve existing attachments.
        attachments = dict(cell.get("attachments") or {})
        existing_keys = set(attachments.keys())

        # Collect replacements for this cell.
        replacements: list[tuple[str, str]] = []

        for match in IMG_MD_PATTERN.finditer(src):
            total_found += 1
            alt_text, link = match.group(1), match.group(2)

            # Skip already-embedded attachments or remote URLs.
            if link.startswith("attachment:"):
                total_skipped += 1
                continue
            if re.match(r"^https?://", link, flags=re.IGNORECASE):
                total_skipped += 1
                continue

            img_path = (nb_dir / link).resolve()
            if not img_path.exists():
                total_missing += 1
                print(f"⚠️  Missing image for cell {idx}: {link} -> {img_path}")
                continue

            mime = guess_mime(img_path)
            if not mime:
                total_skipped += 1
                print(f"⏭️  Skipping unsupported type: {img_path}")
                continue

            b64 = b64encode_file(img_path)
            key = unique_key(img_path.name, existing_keys)
            existing_keys.add(key)
            attachments[key] = {mime: b64}

            # Replace ONLY the path portion with attachment:key
            original = match.group(0)
            replaced = original.replace(link, f"attachment:{key}", 1)
            replacements.append((original, replaced))
            total_embedded += 1
            print(f"✅  Embedded: {link}  →  attachment:{key} (cell {idx})")

        # Apply replacements if any, and write attachments back.
        if replacements:
            for original, replaced in replacements:
                src = src.replace(original, replaced)
            cell["source"] = src
            cell["attachments"] = attachments

    if out_path is None:
        out_path = nb_path.with_name(nb_path.stem + "_embedded.ipynb")

    nbformat.write(nb, out_path)

    print("\n=== Summary ===")
    print(f"Notebook:         {nb_path}")
    print(f"Output:           {out_path}")
    print(f"Images found:     {total_found}")
    print(f"Embedded:         {total_embedded}")
    print(f"Missing files:    {total_missing}")
    print(f"Skipped (url/ext):{total_skipped}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed local Markdown-linked images into a notebook."
    )
    parser.add_argument("notebook", type=Path, help="Path to .ipynb file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output .ipynb path (default: *_embedded.ipynb)"
    )
    args = parser.parse_args()
    embed_images(args.notebook, args.output)


if __name__ == "__main__":
    main()

---
applyTo: 'none'
---
description: >
  The rule enforces a unified, thesis-grade plotting style for all
  Matplotlib, Seaborn and MONAI visualizations used in the project.
  It standardizes fonts, palette, DPI, linewidths and file saving
  so figures are consistent across the thesis experiments.
alwaysApply: false

rules:
  - id: thesis-plot-style
    description: "Unified thesis plotting style for Matplotlib/Seaborn/MONAI"
    when: "Generating any visualization plot (Matplotlib, Seaborn, ecc )"
    actions: |
      - Import and activate the StyleManager defined in `style/style_manager.py`:
        from style.style_manager import StyleManager
        sm = StyleManager()
        sm.activate()

      - Assume the Matplotlib style file is at: style/thesis_style.mplstyle

      - When saving figures, call:
        sm.savefig("figures/<descriptive_name>.png")
        Defaults: dpi=300, bbox_inches='tight', transparent=True

      - Visual properties to enforce:
        * Font family: DejaVu Sans / Arial
        * Font sizes: title=14, labels=12, ticks=10
        * Line width: 2.0
        * Marker size: 6
        * Palette: Okabe–Ito (colorblind-safe) with hex:
          #0072B2, #D55E00, #009E73, #CC79A7, #F0E442, #56B4E9, #E69F00, #999999

      - Seaborn default:
        sns.set_theme(style="whitegrid", palette=sm.palette)

      - General plotting rules:
        * Label axes with units where applicable
        * Use sentence-case titles
        * Include legends only when multiple categories exist
        * Remove top/right spines
        * Prefer 'viridis' for heatmaps and intensity maps
        * Avoid 3D effects, shadows, gradients, and excessive gridlines

      - Statistical plots:
        * Maintain class order: ["MSA-P", "MSA-C", "PD"]
        * Show mean/median indicators where meaningful

      - Model-comparison plots:
        * Display mean ± std with labels
        * Group bars side-by-side for different models/settings
        * Add horizontal gridlines for readability

      - Save figures in the `images/` folder with descriptive names:
        01_dataset_distribution.png, 02_corr_heatmap.png, etc.

stylePrompt: |
  Use my unified thesis plotting style:
  - Load and activate `style/style_manager.py` with StyleManager.activate()
  - Seaborn theme: whitegrid, Okabe–Ito palette
  - Matplotlib style: style/thesis_style.mplstyle (default 6×4 in, 300 DPI)
  - Sans-serif fonts, 2.0 pt lines, clear labels/titles
  - Save with `sm.savefig(..., dpi=300, bbox_inches='tight', transparent=True)`
  - No 3D, no shadows, no color gradients
  - Maintain consistent PD/MSA class order and colors
---
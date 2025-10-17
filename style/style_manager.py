"""
style_manager.py
----------------
Utilities to enforce a consistent, thesis-grade plotting style across
Matplotlib and Seaborn.

Usage
-----
from style.style_manager import StyleManager

sm = StyleManager()  # or pass a custom palette, font, etc.
sm.activate()        # sets Matplotlib rcParams and Seaborn theme

# ... your plotting code (Matplotlib or Seaborn) ...

sm.savefig("figures/01_dataset_distribution.png")  # tight/transparent by default

# Or use as a context manager:
with StyleManager().context():
    # plots here automatically use the thesis style
    ...
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # Seaborn optional
    sns = None


DEFAULT_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple/magenta
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#999999",  # grey
]


@dataclass
class StyleManager:
    """
    Manage a unified plotting style for Matplotlib and Seaborn.

    Parameters
    ----------
    mplstyle_path : str
        Path to a .mplstyle file to load.
    palette : List[str]
        List of hex colors to use for categorical data.
    font_family : Optional[str]
        Override the default font family (e.g., "DejaVu Sans").
    dpi : int
        Default dpi for figures and saved files.
    transparent : bool
        Save figures with transparent background by default.
    """

    mplstyle_path: str = "style/thesis_style.mplstyle"
    palette: List[str] = field(default_factory=lambda: DEFAULT_PALETTE.copy())
    font_family: Optional[str] = None
    dpi: int = 300
    transparent: bool = True

    def activate(self) -> None:
        """
        Apply the style globally. Safe to call multiple times.
        """
        # Load Matplotlib style
        try:
            plt.style.use(self.mplstyle_path)
        except OSError as exc:
            raise FileNotFoundError(
                f"Cannot load mplstyle at '{self.mplstyle_path}'."
            ) from exc

        # Optional font override
        if self.font_family:
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [self.font_family]

        # Seaborn integration (if available)
        if sns is not None:
            # Use current Matplotlib rcParams but set seaborn palette and defaults
            sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
            sns.set_palette(self.palette)

        # Ensure default DPI aligns with saving DPI
        mpl.rcParams["figure.dpi"] = self.dpi
        mpl.rcParams["savefig.dpi"] = self.dpi

    @contextmanager
    def context(self):
        """
        Temporary context that activates the style and restores state on exit.
        """
        with mpl.rc_context():
            prev_params = mpl.rcParams.copy()
            self.activate()
            try:
                yield
            finally:
                mpl.rcParams.update(prev_params)

    def savefig(
        self,
        path: str,
        *,
        dpi: Optional[int] = None,
        transparent: Optional[bool] = None,
        bbox_inches: str = "tight",
        pad_inches: float = 0.02,
    ) -> None:
        """
        Save the current figure with thesis defaults.

        Parameters
        ----------
        path : str
            Output file path (e.g., 'figures/my_plot.png').
        dpi : Optional[int]
            Override DPI for this save only.
        transparent : Optional[bool]
            Whether to use transparent background.
        bbox_inches : str
            Bounding box setting (default 'tight').
        pad_inches : float
            Padding around the figure.
        """
        dpi = dpi if dpi is not None else self.dpi
        transparent = (
            transparent if transparent is not None else self.transparent
        )
        plt.savefig(
            path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )

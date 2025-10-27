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
from cycler import cycler

try:
    import seaborn as sns
except Exception:  # Seaborn optional
    sns = None

# Okabe-Ito palette
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
    palette : 
        List[str] = DEFAULT_PALETTE.copy() if None else List[str] List of hex colors to use for categorical data.
    
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
        # 1) Load .mplstyle
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

        # 2) Sync palette from the active style cycle
        cycle = mpl.rcParams.get("axes.prop_cycle")
        if cycle:
            self.palette = cycle.by_key().get("color", self.palette)

        # 3) Set Seaborn theme but KEEP the cycle from .mplstyle
        if sns is not None:
            sns.set_theme(style="whitegrid", context="notebook",
                        rc={"axes.prop_cycle": cycler(color=self.palette)})
            # Optional: also set seaborn's categorical palette for functions that read it
            sns.set_palette(self.palette)

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
        path: str = "images/",
        *,
        dpi: Optional[int] = None,
        transparent: Optional[bool] = None,
        bbox_inches: str = "tight",
        pad_inches: float = 0.12,
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
            Padding around the figure (default 0.12 for a bit more right margin
            in exported images).
        """
        dpi = dpi if dpi is not None else self.dpi
        
        transparent = (
            transparent if transparent is not None else self.transparent
        )
        # path = os.path.join(path, filename)
        plt.savefig(
            path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        
        print(f"Figure saved to {path}")

#!/usr/bin/env python
"""Shared matplotlib style for the Science Advances figures (0 API cost).

Style follows docs/画图代码.md: Arial, font 9/10/11/8, thin spines,
PDF/SVG (Type-42 fonts), 300/600 dpi, no seaborn, restrained palette:
  gray      -> low effort (baseline)
  dark blue -> high effort
  dark red  -> max effort (potential harm)
  dark gold -> router / single accent
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "data" / "figures"

# palette (kept deliberately small)
C_LOW = "#4d4d4d"      # gray: baseline / low effort
C_HIGH = "#1f4e79"     # dark blue: high effort
C_MAX = "#a62c2c"      # dark red: max effort / harm
C_ROUTER = "#b07a1d"   # single accent: router / allocation policy
C_FLOOR = "#8a8a8a"    # muted gray for floor controls

SETTING_ORDER = ["low", "high", "max"]
SETTING_LABELS = ["low", "high", "max"]
SETTING_COLORS = {"low": C_LOW, "high": C_HIGH, "max": C_MAX}


def set_science_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(
    name: str,
    figdir: Path | str = FIGDIR,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    close: bool = True,
) -> list[str]:
    """Save the current figure as publication PDF/SVG (+ PNG preview)."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for ext in formats:
        p = figdir / f"{name}.{ext}"
        plt.savefig(p, bbox_inches="tight")
        out.append(str(p))
    if close:
        plt.close()
    return out


def panel_label(fig, ax, label: str, y_pad: float = 0.014, fontsize: int = 13) -> None:
    """Bold panel label placed above the axes in figure coordinates.

    Figure-coordinate placement avoids colliding with titles or neighboring
    panels (the common failure of transform=ax.transAxes labels).
    """
    bb = ax.get_position()
    fig.text(
        bb.x0,
        bb.y1 + y_pad,
        label,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

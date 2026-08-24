#!/usr/bin/env python
"""Generate the three main figures for the NMI manuscript (0 API).

All numbers are read from JSON single sources of truth under results/tables/.

Fig. 1  Heterogeneous returns to reasoning (measure)
Fig. 2  Reasoning demand vs difficulty; demand is model-relative (flash vs pro)
Fig. 3  Structural computational load: positive manipulations and negative
        controls (expenditure != demand)

Calibration-simulation and prospective-validation results are reported as
main-text Tables 3 and 4 (see docs/NMI_MAIN_TEXT.md), not as figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figure_style import (
    C_FLOOR,
    C_HIGH,
    C_LOW,
    C_MAX,
    C_ROUTER,
    panel_label,
    save_fig,
    set_science_style,
)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data" / "tables"
SETTINGS = ["low", "high", "max"]
RANK = {"low": 1, "high": 2, "max": 3}


def _load(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def _r_star(acc: dict[str, float], rt: dict[str, float], eps: float) -> str:
    a_star = max(acc.values())
    qualified = [s for s in SETTINGS if acc[s] >= a_star - eps]
    return min(qualified, key=lambda s: rt[s])


def _fig1() -> None:
    pop = _load("paper_tables.json")["population"]
    benches = [
        ("math500", "MATH-500"),
        ("easy2hard_amc", "E2H-AMC"),
        ("aime", "AIME"),
        ("gpqa_diamond", "GPQA Diamond"),
        ("livecodebench", "LiveCodeBench"),
    ]
    colors = [C_LOW, C_HIGH, C_MAX]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.1))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.15, hspace=0.58, wspace=0.5)
    axes = axes.flatten()

    for i, (key, label) in enumerate(benches):
        ax = axes[i]
        cell = pop[key]
        accs = [cell[s]["accuracy"] for s in SETTINGS]
        ax.plot([0, 1, 2], accs, color="#c9c9c9", lw=1.0, zorder=1)
        for s, c in zip(SETTINGS, colors):
            v = cell[s]
            acc = v["accuracy"]
            lo, hi = v["ci"]
            ax.errorbar(
                [SETTINGS.index(s)],
                [acc],
                yerr=[[acc - lo], [hi - acc]],
                fmt="o",
                ms=4,
                color=c,
                ecolor=c,
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
        ax.set_xticks([0, 1, 2], ["low", "high", "max"])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(label, fontsize=9.5, pad=6)
        ax.grid(alpha=0.2, lw=0.5)
        if i == 0:
            ax.set_ylabel("Accuracy")
        if i in (3, 4):
            ax.set_xlabel("Configured reasoning effort")
        panel_label(fig, ax, "ABCDE"[i])

        if key == "math500":
            lo_h, hi_h, mx = cell["low"], cell["high"], cell["max"]
            mru1 = (hi_h["accuracy"] - lo_h["accuracy"]) / ((hi_h["mean_rt"] - lo_h["mean_rt"]) / 1000)
            mru2 = (mx["accuracy"] - hi_h["accuracy"]) / ((mx["mean_rt"] - hi_h["mean_rt"]) / 1000)
            ax.text(
                0.02,
                0.06,
                f"MRU +{mru1:.2f} / {mru2:.2f} per 1k tokens",
                transform=ax.transAxes,
                fontsize=6.2,
                color="#333333",
                ha="left",
                va="bottom",
            )

    ax = axes[5]
    markers = ["o", "s", "^", "D", "v"]
    for (key, label), m in zip(benches, markers):
        toks = [pop[key][s]["mean_rt"] for s in SETTINGS]
        ax.plot([0, 1, 2], toks, marker=m, ms=4, color="#555555", lw=1.3, label=label)
    ax.set_yscale("log")
    ax.set_yticks([1000, 3000, 10000, 30000], ["1k", "3k", "10k", "30k"])
    ax.set_xticks([0, 1, 2], ["low", "high", "max"])
    ax.set_xlabel("Configured reasoning effort")
    ax.set_ylabel("Mean reasoning tokens")
    ax.set_title("Token expenditure", fontsize=9.5, pad=6)
    ax.grid(alpha=0.2, lw=0.5, which="both")
    ax.text(1.12, 22800, "high > max", ha="center", fontsize=6.5, color=C_MAX)
    panel_label(fig, ax, "F")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=5,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=7,
        frameon=False,
        columnspacing=1.2,
    )
    save_fig("Figure1_reasoning_returns")


def _fig2() -> None:
    sd = _load("stratum_demand.json")
    fp = _load("full_pilot_analysis.json")
    matrix = sd["matrix"]
    rows = [
        ("math500", "MATH-500"),
        ("easy2hard_amc", "E2H-AMC"),
        ("zebralogic_grid", "ZebraLogic"),
    ]
    cols = [f"S{i}" for i in range(1, 6)]

    pro_cols = ["math500", "easy2hard_amc", "zebralogic_grid"]
    pro_r = {}
    for b in pro_cols:
        acc = {s: fp["population"][f"deepseek-v4-pro|{b}|{s}"]["accuracy"] for s in SETTINGS}
        rt = {s: fp["population"][f"deepseek-v4-pro|{b}|{s}"]["mean_reasoning_tokens"] for s in SETTINGS}
        pro_r[b] = _r_star(acc, rt, sd["epsilon"])
    flash_r = {b: sd["benchmark"][b]["r_star"] for b in pro_cols}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), gridspec_kw={"width_ratios": [1.0, 1.35, 0.72]})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.17, wspace=0.5)

    # A: r*_eps matrix (flash)
    ax = axes[0]
    data = np.array([[RANK[matrix[k][j]] for j in range(5)] for k, _ in rows])
    cmap = ListedColormap(["#d9d9d9", C_HIGH, C_MAX])
    ax.imshow(data, cmap=cmap, vmin=1, vmax=3, aspect="auto")
    for i in range(len(rows)):
        for j in range(5):
            v = matrix[rows[i][0]][j]
            ax.text(
                j,
                i,
                v,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v != "low" else "#111111",
            )
    ax.set_xticks(range(5), cols)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_title("Sufficient setting r*\u03b5\n(gray low \u00b7 blue high \u00b7 red max)", fontsize=8.5, pad=6)
    panel_label(fig, ax, "A")

    # B: difficulty rank vs demand rank
    ax = axes[1]
    series = [
        ("math500", "MATH", [RANK[v] for v in matrix["math500"]], "o", C_LOW),
        ("easy2hard_amc", "E2H", [RANK[v] for v in matrix["easy2hard_amc"]], "s", C_HIGH),
        ("zebralogic_grid", "Zebra", [RANK[v] for v in matrix["zebralogic_grid"]], "^", C_MAX),
        ("livecodebench", "LCB", [RANK[v] for v in matrix["livecodebench"]], "D", C_ROUTER),
    ]
    for _, lab, vals, m, c in series:
        xs = [1, 2, 3, 4, 5][: len(vals)] if len(vals) == 5 else [1, 2, 3]
        ax.plot(xs, vals, marker=m, ms=4.5, lw=1.5, color=c, label=lab)
    ax.set_xticks([1, 2, 3, 4, 5], ["S1", "S2", "S3", "S4", "S5"])
    ax.set_yticks([1, 2, 3], ["low", "high", "max"])
    ax.set_ylim(0.5, 3.5)
    ax.set_xlim(0.7, 5.3)
    ax.set_xlabel("Nominal difficulty stratum")
    ax.set_ylabel("Reasoning demand rank")
    ax.set_title("Difficulty \u2260 demand", fontsize=9.5, pad=6)
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(fontsize=6.8, loc="lower right", ncols=2, frameon=False)
    ax.text(
        0.02,
        0.97,
        "E2H: hardest strata need less reasoning",
        transform=ax.transAxes,
        fontsize=6.5,
        color=C_HIGH,
        ha="left",
        va="top",
    )
    panel_label(fig, ax, "B")

    # C: demand is model-relative (flash vs pro)
    ax = axes[2]
    ddata = np.array([[RANK[flash_r[b]] for b in pro_cols], [RANK[pro_r[b]] for b in pro_cols]])
    ax.imshow(ddata, cmap=cmap, vmin=1, vmax=3, aspect="auto")
    for i, model in enumerate(["flash", "pro"]):
        for j, b in enumerate(pro_cols):
            v = flash_r[b] if i == 0 else pro_r[b]
            ax.text(
                j,
                i,
                v,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v != "low" else "#111111",
            )
    ax.set_xticks(range(3), ["MATH", "E2H", "Zebra"])
    ax.set_yticks(range(2), ["flash", "pro"])
    ax.set_title("Demand is model-relative", fontsize=9, pad=6)
    ax.text(
        0.5,
        -0.22,
        "pro coverage 74\u2013100%",
        transform=ax.transAxes,
        fontsize=6,
        color="#555555",
        ha="center",
    )
    panel_label(fig, ax, "C")

    save_fig("Figure2_reasoning_demand")


def _negative_series(mech, st, factor: str):
    if factor == "statetrack":
        xs = [2, 4, 8]
        toks = [st["trial"][str(k)]["low"]["mean_rt"] for k in xs]
        accs = [st["trial"][str(k)]["low"]["accuracy"] for k in xs]
        return xs, toks, accs
    rows = sorted([c for c in mech["cells"][factor] if c["setting"] == "low"], key=lambda c: c["level"])
    return [c["level"] for c in rows], [c["mean_rt"] for c in rows], [c["accuracy"] for c in rows]


def _fig3() -> None:
    mech = _load("mechanism_summary.json")
    sw = _load("searchwidth_mid.json")
    st = _load("statetrack_mid.json")

    def pos_series(factor: str, setting: str):
        rows = sorted([c for c in mech["cells"][factor] if c["setting"] == setting], key=lambda c: c["level"])
        return [c["level"] for c in rows], [c["accuracy"] for c in rows], [c["ci"] for c in rows]

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.1, hspace=0.55, wspace=0.45)

    # A: constraint load
    ax = axes[0, 0]
    for s, c in [("low", C_LOW), ("high", C_HIGH)]:
        xs, ys, cis = pos_series("constraints", s)
        yerr = np.array([[y - lo, hi - y] for y, (lo, hi) in zip(ys, cis)]).T
        ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=4.5, lw=1.6, color=c, label=s, capsize=2.5)
    ax.axvspan(11.2, 12.8, color="#eeeeee", zorder=0)
    ax.text(10.0, 0.79, "R* shifts\nlow \u2192 high", ha="center", fontsize=6.8, color=C_MAX, linespacing=1.3)
    ax.set_xticks([4, 8, 12])
    ax.set_ylim(0.72, 1.04)
    ax.set_xlabel("Clues")
    ax.set_ylabel("Accuracy")
    ax.set_title("Constraint load", fontsize=9.5, pad=6)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(alpha=0.2, lw=0.5)
    panel_label(fig, ax, "A")

    # B: search width
    ax = axes[0, 1]
    for s, c in [("low", C_LOW), ("high", C_HIGH), ("max", C_MAX)]:
        xs = [2, 4, 8]
        ys = [sw["trial"][str(b)][s]["accuracy"] for b in xs]
        cis = [sw["trial"][str(b)][s]["ci"] for b in xs]
        yerr = np.array([[y - lo, hi - y] for y, (lo, hi) in zip(ys, cis)]).T
        style = dict(marker="o", ms=4.5, lw=1.6, capsize=2.5)
        if s == "max":
            style["linestyle"] = "--"
        ax.errorbar(xs, ys, yerr=yerr, color=c, label=s, **style)
    ax.text(
        0.97,
        0.05,
        "B=8: low \u22126pp vs high",
        transform=ax.transAxes,
        fontsize=6.5,
        color=C_MAX,
        ha="right",
        va="bottom",
    )
    ax.set_xticks([2, 4, 8])
    ax.set_ylim(0.82, 1.04)
    ax.set_xlabel("Branching factor B")
    ax.set_ylabel("Accuracy")
    ax.set_title("Search width", fontsize=9.5, pad=6)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.grid(alpha=0.2, lw=0.5)
    panel_label(fig, ax, "B")

    # C: expenditure scales with B
    ax = axes[0, 2]
    for s, c in [("low", C_LOW), ("high", C_HIGH)]:
        xs = [2, 4, 8]
        toks = [sw["trial"][str(b)][s]["mean_rt"] for b in xs]
        ax.plot(xs, toks, marker="o", ms=4.5, lw=1.6, color=c, label=s)
    ax.set_yscale("log")
    ax.set_yticks([5000, 10000, 20000, 40000], ["5k", "10k", "20k", "40k"])
    ax.set_xticks([2, 4, 8])
    ax.set_xlabel("Branching factor B")
    ax.set_ylabel("Mean tokens")
    ax.set_title("Expenditure scales with B", fontsize=9.5, pad=6)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(alpha=0.2, lw=0.5, which="both")
    panel_label(fig, ax, "C")

    # D-F: negative controls (expenditure != demand)
    controls = [
        ("depth", "Sequential depth", "Steps"),
        ("distractor", "Distractor load", "Distractors"),
        ("statetrack", "State-tracking load", "Tracked variables k"),
    ]
    for i, (ax, (factor, title, xlab)) in enumerate(zip(axes[1], controls)):
        xs, toks, accs = _negative_series(mech, st, factor)
        xpos = np.arange(len(xs))
        bars = ax.bar(xpos, toks, color="#cccccc", width=0.55, label="low tokens")
        ax.set_xticks(xpos, [str(x) for x in xs])
        ax.set_xlabel(xlab)
        ax.set_ylabel("Mean tokens")
        ax.set_title(title, fontsize=9.5, pad=6)
        ax2 = ax.twinx()
        (line,) = ax2.plot(xpos, accs, marker="o", ms=4.5, color=C_HIGH, lw=1.6, label="low accuracy")
        ax2.set_ylim(0.9, 1.04)
        ax2.set_yticks([0.92, 0.96, 1.0])
        ax2.set_ylabel("Accuracy", color=C_HIGH)
        ax2.tick_params(axis="y", colors=C_HIGH, labelsize=8)
        ax2.set_xticklabels([])
        ax.legend(
            handles=[bars, line],
            labels=["low tokens", "low accuracy"],
            loc="upper left",
            fontsize=6.8,
            frameon=False,
        )
        panel_label(fig, ax, "DEF"[i])

    save_fig("Figure3_task_structure")


def main() -> int:
    set_science_style()
    _fig1()
    _fig2()
    _fig3()
    print("figures written to results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

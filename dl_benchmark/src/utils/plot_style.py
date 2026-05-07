import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm


CAT_PALETTE = "colorblind"
CAT_PALETTE_20 = "tab20"
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "crest"
BG_GREY = "0.75"
BG_ALPHA = 0.15
RASTERIZE_THRESHOLD = 1000


def make_color_map(categories, palette=CAT_PALETTE):
    cats = list(dict.fromkeys([str(c) for c in categories]))
    pal = sns.color_palette(palette, n_colors=len(cats))
    return {c: pal[i] for i, c in enumerate(cats)}


def method_color_map(categories):
    cats = list(dict.fromkeys([str(c) for c in categories]))
    palette = CAT_PALETTE if len(cats) <= 10 else CAT_PALETTE_20
    cmap = make_color_map(cats, palette=palette)
    # Keep a stable highlighted mapping for the main benchmark figures.
    if "mofaflex" in cmap and "concerto" in cmap:
        cmap["mofaflex"], cmap["concerto"] = cmap["concerto"], cmap["mofaflex"]
    return cmap


def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")


def clean_ax(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_zero_lines(ax, alpha=0.6):
    ax.axhline(0, ls="--", lw=0.8, color="0.6", alpha=alpha, zorder=0)
    ax.axvline(0, ls="--", lw=0.8, color="0.6", alpha=alpha, zorder=0)


def place_legend(
    ax,
    mode="inside",
    title=None,
    loc="upper right",
    anchor=(0.98, 0.98),
    ncol=1,
    fontsize=None,
):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if mode == "outside":
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
            ncol=ncol,
            fontsize=fontsize,
        )
    elif mode == "inside":
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            loc=loc,
            ncol=ncol,
            fontsize=fontsize,
        )
    elif mode == "manual":
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            loc="center",
            bbox_to_anchor=anchor,
            ncol=ncol,
            fontsize=fontsize,
        )


def centered_norm(values, center=0.0, q=0.995):
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return TwoSlopeNorm(vmin=-1, vcenter=center, vmax=1)
    vmax = np.quantile(np.abs(v), q)
    vmax = max(vmax, 1e-6)
    return TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)


def clean_colorbar(cbar):
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=8)


def set_effect_colorbar_ticks(fig, vmin, vmax, include_zero=True):
    ticks = [float(vmin), float(vmax)]
    if include_zero:
        ticks.append(0.0)
    ticks = sorted(set(ticks))

    def _fmt(v):
        rv = float(np.round(v, 1))
        if abs(rv) < 1e-12:
            return "0"
        s = f"{rv:.1f}"
        if s.endswith(".0"):
            return s[:-2]
        return s

    labels = [_fmt(v) for v in ticks]
    for ax in fig.axes:
        title = ax.get_title().strip()
        xlabel = ax.get_xlabel().strip()
        ylabel = ax.get_ylabel().strip()

        if "|effect size|" in f"{title} {xlabel} {ylabel}".lower():
            continue
        is_effect_cbar = (
            title == "Effect size" or xlabel == "Effect size" or ylabel == "Effect size"
        )
        if not is_effect_cbar:
            continue

        pos = ax.get_position()
        if pos.width >= pos.height:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks([])
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_xticks([])
        break


def set_house_style(dpi=300, font="DejaVu Sans"):
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.figsize": (4.5, 3.5),
            "font.family": font,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.labelpad": 2.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="white", context="paper")
    sns.set_style({"axes.grid": False})


set_house_style()

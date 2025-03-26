import matplotlib as mpl
import plotnine
from functools import partial
import numpy as np
from plotnine import *
from mizani.palettes import brewer_pal

# Define constants
FONT_FAMILY = "Helvetica"
FONT_SIZE = 10
DISCRETE_COLORS = [
    "#1f78b4",
    "#33a02c",
    "#e31a1c",
    "#ff7f00",
    "#6a3d9a",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
    "#ffff99",
]

# Configure matplotlib settings
def setup_matplotlib():
    """Configure matplotlib global settings."""
    settings = {
        "svg.fonttype": "none",
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
    }
    mpl.rcParams.update(settings)

# Configure plotnine theme
def setup_plotnine_theme():
    """Configure and set the default plotnine theme."""
    plotnine.options.base_family = FONT_FAMILY
    th = theme_bw(base_size=FONT_SIZE, base_family=FONT_FAMILY) + theme(
        line=element_line(size=0.5),
        rect=element_rect(size=0.5),
        panel_grid_minor=element_blank(),
        panel_border=element_line(),
        axis_ticks=element_line(color="black"),
        axis_ticks_minor=element_blank(),
        axis_text=element_text(color="black", size=FONT_SIZE),
        strip_background=element_blank(),
        strip_text=element_text(color="black", size=FONT_SIZE),
        legend_text=element_text(size=FONT_SIZE),
        legend_key=element_blank(),
        plot_title=element_text(ha="center"),
        aspect_ratio=1,
    )
    theme_set(th)


def _rescale_zerosymmetric(x, to: tuple[float, float] = (0, 1), _from: tuple[float, float] | None = None):
    _from = _from or (np.min(x), np.max(x))
    return np.interp(x, (_from[0], 0, _from[1]), (0, 0.5, 1))

_scale_fill_zerosymmetric_diverging = partial(
    scale_fill_gradientn,
    colors=brewer_pal(type="div", palette="RdBu", direction=-1)(11),
    rescaler=_rescale_zerosymmetric,
    expand=(0, 0),
)

_weights_inferred_color_scale = scale_color_manual(
    values=("red", "black"), breaks=(True, False), labels=("Inferred", "Annotated")
)

discrete_scale_fill = scale_fill_manual(values=DISCRETE_COLORS)
discrete_scale_color = scale_color_manual(values=DISCRETE_COLORS)

setup_matplotlib()
setup_plotnine_theme()
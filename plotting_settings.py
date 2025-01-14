import matplotlib as mpl
import plotnine
from functools import partial
import numpy as np
from plotnine import *
from mizani.palettes import brewer_pal


mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "Helvetica"
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["axes.titlesize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["figure.titlesize"] = 7

plotnine.options.base_family = "Helvetica"
th = theme_bw(base_size=7, base_family="Helvetica") + theme(
    line=element_line(size=0.5),
    rect=element_rect(size=0.5),
    panel_grid_minor=element_blank(),
    panel_border=element_line(),
    axis_ticks=element_line(color="black"),
    axis_ticks_minor=element_blank(),
    axis_text=element_text(color="black", size=7),
    strip_background=element_blank(),
    strip_text=element_text(color="black", size=7),
    legend_text=element_text(size=7),
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

discrete_scale_fill = scale_fill_manual(values=DISCRETE_COLORS)
discrete_scale_color = scale_color_manual(values=DISCRETE_COLORS)
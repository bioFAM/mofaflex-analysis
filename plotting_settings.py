from plotnine import *
import plotnine as p9
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = mpl.rcParams["axes.labelsize"] = mpl.rcParams[
    "axes.titlesize"
] = mpl.rcParams["xtick.labelsize"] = mpl.rcParams["ytick.labelsize"] = mpl.rcParams[
    "legend.fontsize"
] = mpl.rcParams["figure.titlesize"] = 7

p9.theme_set(p9.theme_bw())

DISCRETE_COLORS = [
    "#FF9999",
    "#66B2FF",
    "#99FF99",
    "#FFCC99",
    "#FF99CC",
    "#99CCFF",
    "#FF6666",
    "#66CC00",
]
discrete_scale_fill = scale_fill_manual(values=DISCRETE_COLORS)
discrete_scale_color = scale_color_manual(values=DISCRETE_COLORS)

import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

#  theme colours
theme_colours: dict = {
    "dark_grey": "#435562",
    "light_grey": "#C6CFD3",
    "rust": "#f14f30",
    "white": "#FFFFFF",
}

#  Categorical colours
theme_categorical_bright: list = [
    "#E14B4B",  # Red
    "#E1964B",  # Orange
    "#ca4be1",  # Purple
    "#4b62e1",  # Blue 
    "#2fb0dc",  # Cyan
    "#3cda8c",  # Aqua
    "#edc81e",  # Yellow
    "#a66e58",  # Brown
]  

theme_categorical_pastel: list = [
    "#95D0A9",  # Light Green
    "#DAC4F7",  # Light Purple
    "#F4989C",  # Light Red
    "#EBD2B4",  # Light Orange
    "#ACECF7",  # Light Blue
    "#59594A",  # Dark Grey
    "#7588A3",  # Blue Grey
    "#B9314F",  # Red
]


def setup_matplotlib_environment() -> None:
    """Set up the matplotlib plotting environment
    to use a default style.
    """

    plt.rc("lines", linewidth=4)
    # plt.rc('axes', prop_cycle=(plt.cycler('linestyle', ['-', '--', ':', '-.'])))

    # Font
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fontsize"] = 18

    # Ticks
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 5.0
    plt.rcParams["xtick.minor.size"] = 3.0
    plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["ytick.minor.size"] = 3.0
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True

    # Axes
    plt.rcParams["axes.edgecolor"] = theme_colours["dark_grey"]
    plt.rcParams["axes.titlecolor"] = theme_colours["dark_grey"]
    plt.rcParams["axes.labelcolor"] = theme_colours["dark_grey"]
    plt.rcParams["xtick.color"] = theme_colours["dark_grey"]
    plt.rcParams["ytick.color"] = theme_colours["dark_grey"]

    # Line width
    plt.rcParams["xtick.major.width"] = 3.0
    plt.rcParams["xtick.minor.width"] = 3.0
    plt.rcParams["ytick.major.width"] = 3.0
    plt.rcParams["ytick.minor.width"] = 3.0
    plt.rcParams["axes.linewidth"] = 3.0

    # Line Colours
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", theme_categorical_pastel)

    # Markers
    plt.rcParams["lines.markersize"] = 10
    plt.rcParams["lines.markeredgewidth"] = 2

    # Legend
    plt.rcParams["legend.handlelength"] = 3.0

    # use svg 
    backend_inline.set_matplotlib_formats('svg')
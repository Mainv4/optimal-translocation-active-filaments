import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    plt.style.use("ggplot")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 25
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["legend.labelspacing"] = 0.1
    plt.rcParams["legend.handlelength"] = 0.7
    plt.rcParams["legend.handletextpad"] = 0.25
    plt.rcParams["legend.borderpad"] = 0.1
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.columnspacing"] = 0.5
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.minor.width"] = 2
    plt.rcParams["ytick.minor.width"] = 2
    plt.rcParams["xtick.major.size"] = 10
    plt.rcParams["ytick.major.size"] = 10
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def read_mass_center_head_tail_x(path):
    path_array = path.split("/")[:-1]
    massCenter_file_name = "__".join(path_array[:]) + "__CenterOfMassOverTime.dat"
    massCenter_data = read_data_x("DATA_CenterOfMass/" + massCenter_file_name)

    head_file_name = "__".join(path_array[:]) + "__HeadOverTime.dat"
    head_data = read_data_x("DATA_Head/" + head_file_name)

    tail_file_name = "__".join(path_array[:]) + "__TailOverTime.dat"
    tail_data = read_data_x("DATA_Tail/" + tail_file_name)
    return massCenter_data, head_data, tail_data


def read_data_x(path):
    data = np.loadtxt(path, skiprows=1)
    if data.size == 0:
        raise ValueError(f"No data found in file: {path}")
    N = int((data.shape[1] - 1) / 2)
    time = data[:, 0]
    x_data = data[:, 1::2]
    y_data = data[:, 2::2]
    return time, x_data, y_data, N

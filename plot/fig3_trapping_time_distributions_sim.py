import matplotlib.pyplot as plt

from common import log_bins, save, set_style_framed, sim_trapping_events
from fig3_trapping_time_distributions_exp import MAX_TRAP_TIME, N_BINS, draw_distribution, finish, scale_fonts
from fig3_trapping_time_vs_length import CANDIDATE, N_TO_LENGTH

SIM_COLORS = {40: "#e8a87c", 50: "#CE7E5A", 60: "#a0522d"}
SIM_MARKERS = {40: "o", 50: "s", 60: "^"}


def main():
    set_style_framed()
    scale_fonts()
    bins = log_bins(MAX_TRAP_TIME, N_BINS)
    fig, ax = plt.subplots()
    for N, lc in N_TO_LENGTH.items():
        times = sim_trapping_events(N, *CANDIDATE, MAX_TRAP_TIME)
        draw_distribution(ax, times, bins, rf"$\ell_c \simeq {lc}$ mm ($N={N}$)", SIM_COLORS[N], SIM_MARKERS[N])
    finish(ax, MAX_TRAP_TIME)
    fig.tight_layout()
    save(fig, "fig3_trapping_time_distributions_sim")


if __name__ == "__main__":
    main()

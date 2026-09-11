import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import LAB_LC, LAB_TTRAP, exp_trapping_events, record, save, set_style_framed, sim_trapping_events

CANDIDATE = (0.4, 0.05, 0.4)
N_TO_LENGTH = {40: 20, 50: 25, 60: 30}
MAX_TRAP_TIME = 10.0
LENGTH_THRESHOLD = 2.0
MIN_EVENTS = 3
C_EXP = "#73A6A3"
C_SIM = "#CE7E5A"


def group_by_length(df, threshold=LENGTH_THRESHOLD, min_events=MIN_EVENTS):
    order = np.argsort(df["length_mm"].values)
    lengths, times = df["length_mm"].values[order], df["time_min"].values[order]
    groups, i = [], 0
    while i < len(lengths):
        j = i + 1
        while j < len(lengths) and abs(lengths[j] - lengths[i]) <= threshold:
            j += 1
        if j - i >= min_events:
            groups.append((float(np.mean(lengths[i:j])), list(times[i:j])))
        i = j
    return groups


def style_violin(parts, face, edge):
    for body in parts["bodies"]:
        body.set_facecolor(face)
        body.set_alpha(0.5 if face == C_EXP else 0.6)
        body.set_edgecolor(edge)
        body.set_linewidth(1)
    parts["cmeans"].set_edgecolor("darkred")
    parts["cmeans"].set_linewidth(2)


def main():
    set_style_framed()
    exp = group_by_length(exp_trapping_events(20, MAX_TRAP_TIME))
    sim = {N: sim_trapping_events(N, *CANDIDATE, MAX_TRAP_TIME) for N in N_TO_LENGTH}
    for lc, times in exp:
        record("C", f"living worms lc = {lc:.1f} mm", contour_length_mm=lc, tau_tr_min=times)
    for N in N_TO_LENGTH:
        if len(sim[N]) >= MIN_EVENTS:
            record("C", f"model N={N}", contour_length_mm=N_TO_LENGTH[N] + 2.5, tau_tr_min=sim[N])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    style_violin(ax.violinplot([g[1] for g in exp], positions=[g[0] for g in exp], widths=1.5,
                               showmeans=True, showmedians=False, showextrema=False), C_EXP, "black")
    positions = [N_TO_LENGTH[N] + 2.5 for N in N_TO_LENGTH if len(sim[N]) >= MIN_EVENTS]
    datasets = [sim[N].tolist() for N in N_TO_LENGTH if len(sim[N]) >= MIN_EVENTS]
    style_violin(ax.violinplot(datasets, positions=positions, widths=1.5,
                               showmeans=True, showmedians=False, showextrema=False), C_SIM, "darkorange")
    ax.set_xlabel(LAB_LC)
    ax.set_ylabel(LAB_TTRAP)
    ax.set_xlim(12, 40)
    ax.set_ylim(-0.5, MAX_TRAP_TIME + 0.5)
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=C_EXP, markersize=10, label="Exp."),
                       Line2D([0], [0], marker="D", color="w", markerfacecolor=C_SIM, markersize=10, label="Sim."),
                       Line2D([0], [0], color="darkred", linewidth=2, label="Mean")],
              loc="upper right", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_trapping_time_vs_length")


if __name__ == "__main__":
    main()

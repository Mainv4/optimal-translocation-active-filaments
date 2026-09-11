from common import (CALIBRATED, DATA, TEMP_COLORS, TEMPERATURES, draw_confmap, filter_by_envelope, load_boundaries,
                    read_confmap_npy, save, set_style_framed)
from fig1_conformational_maps_exp import colorbar, row_figure


def main():
    set_style_framed()
    env = load_boundaries()
    fig, axes, cax = row_figure()
    for ax, T in zip(axes, TEMPERATURES):
        pe, t, k = CALIBRATED[T]
        re, rg = read_confmap_npy(DATA / "sim" / "free_space" / "conformations" / f"Pe_{pe}_T_{t}_k_{k}.npy")
        re, rg = filter_by_envelope(re, rg, env)
        mesh = draw_confmap(ax, re, rg, env, boundary_color=TEMP_COLORS[T],
                            label=rf"Sim $f^a\!=\!{pe},\,T\!=\!{t},\,\kappa\!=\!{k}$",
                            panel="C", series=f"model fa={pe} T={t} kappa={k}")
    colorbar(fig, mesh, cax)
    save(fig, "fig1_conformational_maps_sim")


if __name__ == "__main__":
    main()

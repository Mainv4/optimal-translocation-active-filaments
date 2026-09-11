import importlib
import zipfile

from common import ROOT, source_data_begin, source_data_frame

FIGURES = {
    "Figure_1": [("fig1_msd", {}),
                 ("fig1_conformational_maps_exp", {}),
                 ("fig1_conformational_maps_sim", {})],
    "Figure_2": [("fig2_com_trajectories", {}),
                 ("fig2_com_distribution", {}),
                 ("fig2_conformational_map_confined", {})],
    "Figure_3": [("fig3_trapping_time_distributions_exp", {}),
                 ("fig3_trapping_time_distributions_sim", {}),
                 ("fig3_trapping_time_vs_length", {})],
    "Figure_4": [("fig4_com_log_ratio", {}),
                 ("fig4_conformational_maps_temperature", {}),
                 ("fig4_trapping_time_distributions_temperature", {}),
                 ("fig4_mean_trapping_time_vs_temperature", {})],
    "Figure_5": [("fig5_trapping_time_vs_decorrelation_time", {}),
                 ("fig5_trapping_time_vs_entropy", {}),
                 ("fig5_normalized_trapping_time_vs_entropy", {})],
    "Figure_6": [("fig6_rotational_msd", {}),
                 ("fig6_translational_msd", {}),
                 ("fig6_head_position_pdf", {}),
                 ("fig6_timescales_vs_temperature", {}),
                 ("fig6_trapping_time_vs_rotational_time", {}),
                 ("fig6_trapping_time_vs_translational_time", {}),
                 ("fig6_normalized_trapping_time_vs_pe", {})],
    "Figure_S3": [("figS_plateau_identification", {})],
    "Figure_S4": [("figS_conformational_maps_channel_exp", {})],
    "Figure_S5": [("figS_conformational_maps_channel_model", {})],
    "Figure_S6": [("figS_trapping_time_vs_length_violin", {})],
    "Figure_S7": [("fig5_trapping_time_vs_decorrelation_time",
                   {"color_by": "kappa", "name": "figS_trapping_time_vs_decorrelation_time_kappa", "panel": "-"})],
    "Figure_S8": [("fig6_trapping_time_vs_translational_time",
                   {"color_by": "kappa", "decorations": False,
                    "name": "figS_trapping_time_vs_translational_time_kappa", "panel": "-"})],
    "Figure_S9": [("figS_entropy_vs_kappa", {})],
}

OUT = ROOT / "source_data"


def panel_file(figure, panel):
    if panel == "-":
        return f"{figure}.csv"
    return f"{figure}_{panel.replace(' ', '_')}.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*.csv"):
        path.unlink()
    written = []
    for figure, entries in FIGURES.items():
        source_data_begin()
        for module_name, kwargs in entries:
            importlib.import_module(module_name).main(**kwargs)
        frame = source_data_frame()
        for panel, block in frame.groupby("panel", sort=False):
            block = block.drop(columns="panel").dropna(axis=1, how="all")
            path = OUT / panel_file(figure, panel)
            block.to_csv(path, index=False, float_format="%.6g")
            written.append(path)
            print(f"{path.name}: {len(block)} rows, {len(block.columns)} columns, "
                  f"{path.stat().st_size / 1e6:.2f} MB")
    archive = ROOT / "source_data.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in written:
            z.write(path, path.name)
    print(f"\n{len(written)} files, source_data.zip: {archive.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

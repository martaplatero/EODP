import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from config.ismConfig import ismConfig
from config.globalConfig import globalConfig
from auxiliary.constants import constants

# File paths
dir_my_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-ISM\my_output"
dir_ref_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-ISM\output"

output_test_dir = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-ISM\my_output_test"
os.makedirs(output_test_dir, exist_ok=True)

threshold_pct = 1e-3      # Acceptance criterion: relative difference < 1e-3 %

# Cross-Validation
ref_files = glob.glob(os.path.join(dir_ref_output, "*.nc"))

if not ref_files:
    print(f"Error: no .nc files found in {dir_ref_output}")
    exit(1)

print("=" * 80)
print(f"{'FILE':<38} | {'MAX ABS ERR':<14} | {'MAX REL ERR (%)':<16} | {'STATUS'}")
print("=" * 80)

all_passed = True
total_files = 0
passed_files = 0

for path_ref in sorted(ref_files):
    filename = os.path.basename(path_ref)
    path_my = os.path.join(dir_my_output, filename)
    total_files += 1

    if not os.path.exists(path_my):
        print(f"{filename:<38} | {'MISSING':<14} | {'MISSING':<16} | [FAIL]")
        all_passed = False
        continue

    try:
        with Dataset(path_my, "r") as ds_my, Dataset(path_ref, "r") as ds_ref:
            var_my = list(ds_my.variables.keys())[-1]
            var_ref = list(ds_ref.variables.keys())[-1]

            data_my = np.array(ds_my.variables[var_my][:], dtype=float)
            data_ref = np.array(ds_ref.variables[var_ref][:], dtype=float)
    except Exception as e:
        print(f"{filename:<38} | {'READ ERR':<14} | {'READ ERR':<16} | [FAIL]")
        all_passed = False
        continue

    if data_my.shape != data_ref.shape:
        print(f"{filename:<38} | {'SHAPE MISMATCH':<14} | {'SHAPE MISMATCH':<16} | [FAIL]")
        all_passed = False
        continue

    diff_abs = np.abs(data_my - data_ref)
    max_abs_err = np.max(diff_abs)

    mask = np.abs(data_ref) > 1e-12
    rel_err_pct = np.zeros_like(diff_abs)
    rel_err_pct[mask] = (diff_abs[mask] / np.abs(data_ref[mask])) * 100.0
    max_rel_err_pct = np.max(rel_err_pct)

    status = "[PASS]" if max_rel_err_pct < threshold_pct else "[FAIL]"
    if status == "[PASS]":
        passed_files += 1
    else:
        all_passed = False

    print(f"{filename:<38} | {max_abs_err:<14.6e} | {max_rel_err_pct:<16.6e} | {status}")

print("=" * 80)
print(f"Summary: {passed_files}/{total_files} files passed.")
if all_passed:
    print("GLOBAL TEST VERDICT: ALL FILES PASSED (< 1e-3 %)")
else:
    print("GLOBAL TEST VERDICT: SOME FILES FAILED")
print("=" * 80)

# Conversion factors -> analytic TOA vs real TOA
cfg = ismConfig()
gcfg = globalConfig()
cte = constants()


def read_toa(directory, name):
    with Dataset(os.path.join(directory, name), "r") as ds:
        return np.array(ds.variables[list(ds.variables.keys())[-1]][:], dtype=float)


def print_row(stage, analytic, real, units, note=""):
    err = abs(real - analytic) / abs(analytic) * 100.0 if analytic != 0 else np.nan
    print(f"{stage:<22} | {analytic:<16.6e} | {real:<16.6e} | {err:<12.4e} | {units:<14} | {note}")
    return err


max_dn = 2 ** cfg.bit_depth - 1
profiles = {}   # analytic vs real TOA along the central ALT line, for the figures

print("\n" + "=" * 120)
print("CONVERSION FACTORS AND ANALYTIC TOA vs REAL TOA (central pixel)")
print("=" * 120)

for iband, band in enumerate(gcfg.bands):
    wv = cfg.wv[iband]

    # Conversion factors (lecture equations, ideal chain)
    K_rad2irr = cfg.Tr * np.pi / 4 * (cfg.D / cfg.f) ** 2                           # [sr]
    E_ph = cte.h_planck * cte.speed_light / wv                                      # [J/ph]
    K_irr2ph = cfg.pix_size ** 2 * cfg.t_int / E_ph / 1000                          # [ph/(mW/m2)]
    K_ph2e = cfg.QE                                                                 # [e-/ph]
    K_irr2e = K_irr2ph * K_ph2e                                                     # [e-/(mW/m2)]
    K_e2dn = cfg.OCF * cfg.ADC_gain * max_dn / (cfg.max_voltage - cfg.min_voltage)  # [DN/e-]
    K_total = K_rad2irr * K_irr2e * K_e2dn                                          # [DN/(mW/m2/sr)]

    # Saved outputs of the ISM
    toa_isrf = read_toa(dir_my_output, f"{gcfg.ism_toa_isrf}{band}.nc")
    toa_opt = read_toa(dir_my_output, f"{gcfg.ism_toa_optical}{band}.nc")
    toa_e = read_toa(dir_my_output, f"{gcfg.ism_toa_e}{band}.nc")
    toa_prnu = read_toa(dir_my_output, f"{gcfg.ism_toa_prnu}{band}.nc")
    toa_ds = read_toa(dir_my_output, f"{gcfg.ism_toa_ds}{band}.nc")
    toa_det = read_toa(dir_my_output, f"{gcfg.ism_toa_detection}{band}.nc")
    toa_dn = read_toa(dir_my_output, f"{gcfg.ism_toa}{band}.nc")

    # Central pixel (and, if it is saturated, the closest unsaturated pixel of the central ALT line)
    nalt, nact = toa_isrf.shape
    ia, ic = nalt // 2, nact // 2
    pixels = [ic]
    if toa_dn[ia, ic] >= max_dn:
        cand = [j for j in range(6, nact - 5) if toa_dn[ia, j] < max_dn]
        if cand:
            pixels.append(min(cand, key=lambda j: abs(j - ic)))

    print(f"\nBAND {band} (lambda = {wv * 1e9:.0f} nm)")
    print("-" * 120)
    print(f"{'FACTOR':<22} | {'VALUE':<16} | UNITS")
    print("-" * 120)
    print(f"{'Rad -> Irrad':<22} | {K_rad2irr:<16.6e} | [sr]")
    print(f"{'Irrad -> Ph':<22} | {K_irr2ph:<16.6e} | [ph/(mW/m2)]")
    print(f"{'Ph -> e (QE)':<22} | {K_ph2e:<16.6e} | [e-/ph]")
    print(f"{'Irrad -> e':<22} | {K_irr2e:<16.6e} | [e-/(mW/m2)]")
    print(f"{'e -> DN':<22} | {K_e2dn:<16.6e} | [DN/e-]")
    print(f"{'Rad -> DN (total)':<22} | {K_total:<16.6e} | [DN/(mW/m2/sr)]")

    for jc in pixels:
        L = toa_isrf[ia, jc]                                     # input radiance [mW/m2/sr]

        # Analytic TOA: propagate the input radiance with the conversion factors only
        E_an = L * K_rad2irr                                     # [mW/m2]
        e_an = E_an * K_irr2e                                    # [e-]
        dn_an_raw = e_an * K_e2dn                                # [DN] before digitisation
        dn_an = float(np.clip(np.round(dn_an_raw), 0, max_dn))   # [DN] rounded and clipped (ADC)
        sat = toa_dn[ia, jc] >= max_dn

        label = "central pixel" if jc == ic else "closest unsaturated pixel"
        print("-" * 120)
        print(f"Pixel [ALT {ia}, ACT {jc}] ({label}) - input radiance L = {L:.4f} mW/m2/sr")
        print("-" * 120)
        print(f"{'STAGE':<22} | {'TOA ANALYTIC':<16} | {'TOA REAL':<16} | {'REL ERR (%)':<12} | {'UNITS':<14} | NOTE")
        print("-" * 120)
        print_row("Optical (irradiance)", E_an, toa_opt[ia, jc], "[mW/m2]", "ism_toa_optical (includes MTF)")
        print_row("Detection (electrons)", e_an, toa_det[ia, jc], "[e-]",
                  "ism_toa_detection (includes PRNU + DS + bad px)")
        print_row("Video chain (DN)", dn_an, toa_dn[ia, jc], "[DN]",
                  "ism_toa" + (" - SATURATED (4095 DN)" if sat else ""))
        if sat:
            print(f"{'':<22}   analytic DN before clipping = {dn_an_raw:.1f} DN > {max_dn} DN")

        # Breakdown of the differences at this pixel
        mtf_eff = toa_opt[ia, jc] / E_an - 1
        prnu_px = toa_prnu[ia, jc] / toa_e[ia, jc] - 1
        ds_px = toa_ds[ia, jc] - toa_prnu[ia, jc]
        print(f"  MTF       : optical / analytic irradiance - 1 = {100 * mtf_eff:+.4f} %")
        print(f"  PRNU      : column {jc} gain = {1 + prnu_px:.4f} ({100 * prnu_px:+.2f} %), kprnu = {cfg.kprnu}")
        print(f"  Dark sig. : {ds_px:.2f} e- added ({100 * ds_px / toa_prnu[ia, jc]:.2f} % of the signal after PRNU)")
        if sat:
            print(f"  Saturation: real detection = {toa_det[ia, jc]:.0f} e-, real DN clipped at {max_dn} DN")

    # Central ALT line for the figures
    E_line = toa_isrf[ia, :] * K_rad2irr
    e_line = E_line * K_irr2e
    dn_line = np.clip(np.round(e_line * K_e2dn), 0, max_dn)
    profiles[band] = {
        "Optical [mW/m2]": (E_line, toa_opt[ia, :]),
        "Detection [e-]": (e_line, toa_det[ia, :]),
        "Video chain [DN]": (dn_line, toa_dn[ia, :]),
    }

print("=" * 120)

# Figures: analytic vs real TOA along the central ALT line, one figure per band
for band, stages in profiles.items():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (stage, (an, real)) in zip(axes, stages.items()):
        ax.plot(real, color="black", linewidth=1.2, label="Real (saved output)")
        ax.plot(an, color="tab:blue", linestyle="--", linewidth=1.2, label="Analytic")
        ax.set_title(stage)
        ax.set_xlabel("ACT pixel [-]")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.suptitle(f"Analytic vs real TOA - central ALT line - {band}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_test_dir, f"fig_toa_analytic_vs_real_{band}.png"), dpi=300)
    plt.close(fig)

# MTF plots
mtf_names = ["Hdiff", "Hdefoc", "Hwfe", "Hdet", "Hsmear", "Hmotion"]
mtf_labels = {"Hdiff": "Diffraction MTF", "Hdefoc": "Defocus MTF", "Hwfe": "WFE Aberrations MTF",
              "Hdet": "Detector MTF", "Hsmear": "Smearing MTF", "Hmotion": "Motion blur MTF",
              "Hsys": "System MTF"}


def read_mat(directory, name):
    with Dataset(os.path.join(directory, name), "r") as ds:
        return np.array(ds.variables[list(ds.variables.keys())[-1]][:], dtype=float)


print("\n" + "=" * 110)
print("SYSTEM MTF AND CONTRIBUTORS AT THE NYQUIST FREQUENCY (f/(1/w) = 0.5)")
print("=" * 110)

for band in gcfg.bands:
    H = {name: read_mat(dir_my_output, f"{name}_{band}.nc") for name in mtf_names + ["Hsys"]}
    nlines, ncolumns = H["Hsys"].shape

    # Normalised frequencies f/(1/w), same definition as in mtf.freq2d
    fnAct = np.arange(-0.5, 0.5 - 1e-6, 1 / ncolumns)
    fnAlt = np.arange(-0.5, 0.5 - 1e-6, 1 / nlines)
    act_pos = fnAct >= 0.0
    alt_pos = fnAlt >= 0.0
    mid_alt = nlines // 2       # row with fnAlt = 0
    mid_act = ncolumns // 2     # column with fnAct = 0

    # ACT and ALT cuts
    cuts = {
        "ACT": (fnAct[act_pos], {n: H[n][mid_alt, act_pos] for n in H}),
        "ALT": (fnAlt[alt_pos], {n: H[n][alt_pos, mid_act] for n in H}),
    }

    print(f"\nBAND {band} (last samples: ACT f/(1/w) = {cuts['ACT'][0][-1]:.4f}, ALT f/(1/w) = {cuts['ALT'][0][-1]:.4f})")
    print("-" * 110)
    print(f"{'MTF':<22} | {'ACT @ Nyquist':<14} | {'ALT @ Nyquist':<14}")
    print("-" * 110)

    # MTF values at the last frequency sample (closest to Nyquist)
    nyq = {cut: {n: c[-1] for n, c in curves.items()} for cut, (fn, curves) in cuts.items()}
    for n in mtf_names + ["Hsys"]:
        print(f"{mtf_labels[n]:<22} | {nyq['ACT'][n]:<14.4f} | {nyq['ALT'][n]:<14.4f}")

    # Defining MTF: lowest contributor at Nyquist
    for cut in ["ACT", "ALT"]:
        dom = min(mtf_names, key=lambda n: nyq[cut][n])
        print(f"Defining MTF ({cut}): {mtf_labels[dom]} (MTF = {nyq[cut][dom]:.4f} at Nyquist)")

    for cut, (fn, curves) in cuts.items():
        plt.figure(figsize=(9, 5.5))
        for n in mtf_names:
            plt.plot(fn, curves[n], label=mtf_labels[n], linewidth=1.2)
        plt.plot(fn, curves["Hsys"], color="black", label="System MTF", linewidth=2.0)
        plt.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="f Nyquist")

        plt.title(f"System MTF - slice {cut} for {band}")
        plt.xlabel("Spatial frequencies f/(1/w) [-]")
        plt.ylabel("MTF")
        plt.xlim(0.0, 0.52)
        plt.ylim(0.0, 1.05)
        plt.grid(True)
        plt.legend(loc="lower left", fontsize=8)

        save_fig_path = os.path.join(output_test_dir, f"fig_system_mtf_{cut.lower()}_{band}.png")
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300)
        plt.close()

print("=" * 110)
print(f"\nAll MTF figures successfully saved in: {output_test_dir}")
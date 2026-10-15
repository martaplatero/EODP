import os
import glob
import numpy as np
from netCDF4 import Dataset

from config.ismConfig import ismConfig
from config.globalConfig import globalConfig
from auxiliary.constants import constants

# File paths
dir_my_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-ISM\my_output"
dir_ref_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-ISM\output"

threshold_pct = 1e-3      # Acceptance criterion: relative difference < 1e-3 %
threshold_dn_pct = 0.05   # Relaxed criterion for e- -> DN (quantisation error of +-0.5 DN)

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

# Conversion factors
cfg = ismConfig()
gcfg = globalConfig()
cte = constants()


def read_toa(directory, name):
    with Dataset(os.path.join(directory, name), "r") as ds:
        return np.array(ds.variables[list(ds.variables.keys())[-1]][:], dtype=float)


def print_factor(name, analytic, measured, units, limit=threshold_pct, note=""):
    if measured is None:
        print(f"{name:<12} | {analytic:<14.6e} | {'-':<14} | {'-':<12} | {units:<16} | [ANALYTIC]")
        return True
    err = abs(measured - analytic) / abs(analytic) * 100.0
    status = "[PASS]" if err < limit else "[FAIL]"
    print(f"{name:<12} | {analytic:<14.6e} | {measured:<14.6e} | {err:<12.4e} | {units:<16} | {status} {note}")
    return status == "[PASS]"


all_factors_passed = True
max_dn = 2 ** cfg.bit_depth - 1

print("\n" + "=" * 110)
print("CONVERSION FACTORS: ANALYTIC vs MEASURED FROM THE OUTPUTS")
print("=" * 110)

for iband, band in enumerate(gcfg.bands):
    wv = cfg.wv[iband]

    # Analytic factors
    K_rad2irr = cfg.Tr * np.pi / 4 * (cfg.D / cfg.f) ** 2                   # [sr]
    E_ph = cte.h_planck * cte.speed_light / wv                              # [J/ph]
    K_irr2ph = cfg.pix_size ** 2 * cfg.t_int / E_ph / 1000                  # [ph/(mW/m2)]
    K_ph2e = cfg.QE                                                         # [e-/ph]
    K_irr2e = K_irr2ph * K_ph2e                                             # [e-/(mW/m2)]
    K_e2dn = cfg.OCF * cfg.ADC_gain * max_dn / (cfg.max_voltage - cfg.min_voltage)  # [DN/e-]
    K_total = K_rad2irr * K_irr2e * K_e2dn                                  # [DN/(mW/m2/sr)]

    # Read outputs
    toa_isrf = read_toa(dir_my_output, f"{gcfg.ism_toa_isrf}{band}.nc")
    toa_opt = read_toa(dir_my_output, f"{gcfg.ism_toa_optical}{band}.nc")
    toa_e = read_toa(dir_my_output, f"{gcfg.ism_toa_e}{band}.nc")
    toa_det = read_toa(dir_my_output, f"{gcfg.ism_toa_detection}{band}.nc")
    toa_dn = read_toa(dir_my_output, f"{gcfg.ism_toa}{band}.nc")

    # Measured factors
    # Radiance -> irradiance: the MTF has H(0,0)=1, so the image mean is preserved
    K_rad2irr_m = toa_opt.mean() / toa_isrf.mean()

    # Irradiance -> electrons: pixel by pixel, excluding saturated pixels (FWC)
    mask_e = (toa_e < 0.999 * cfg.FWC) & (toa_opt > 0)
    K_irr2e_m = np.median(toa_e[mask_e] / toa_opt[mask_e])
    n_sat = np.sum(~mask_e)

    # Electrons -> DN: excluding clipped pixels and low DN (rounding error)
    mask_dn = (toa_dn > 100) & (toa_dn < max_dn)
    K_e2dn_m = np.sum(toa_dn[mask_dn]) / np.sum(toa_det[mask_dn])
    dn_pred = np.clip(np.round(toa_det * K_e2dn), 0, max_dn)
    max_dn_diff = np.max(np.abs(dn_pred - toa_dn))

    print(f"\nBAND {band} (lambda = {wv * 1e9:.0f} nm)")
    print("-" * 110)
    print(f"{'STEP':<12} | {'ANALYTIC':<14} | {'MEASURED':<14} | {'REL ERR (%)':<12} | {'UNITS':<16} | STATUS")
    print("-" * 110)

    ok = print_factor("Rad->Irrad", K_rad2irr, K_rad2irr_m, "[sr]", note="(optical/isrf)")
    print_factor("Irrad->Ph", K_irr2ph, None, "[ph/(mW/m2)]")
    print_factor("Ph->e (QE)", K_ph2e, None, "[e-/ph]")
    ok &= print_factor("Irrad->e", K_irr2e, K_irr2e_m, "[e-/(mW/m2)]",
                       note=f"(toa_e/optical, {n_sat} saturated px excluded)")
    ok &= print_factor("e->DN", K_e2dn, K_e2dn_m, "[DN/e-]", limit=threshold_dn_pct,
                       note=f"(ism_toa/detection, max |DN diff| = {max_dn_diff:.0f} DN)")
    ok &= max_dn_diff <= 1
    print_factor("Rad->DN", K_total, None, "[DN/(mW/m2/sr)]")

    ia, ic = toa_dn.shape[0] // 2, toa_dn.shape[1] // 2
    print(f"Pixel [{ia},{ic}]: {toa_det[ia, ic]:.2f} e- -> expected {dn_pred[ia, ic]:.0f} DN | "
          f"output {toa_dn[ia, ic]:.0f} DN{' (saturated)' if dn_pred[ia, ic] == max_dn else ''}")

    all_factors_passed &= ok

print("=" * 110)
if all_factors_passed:
    print("CONVERSION FACTORS VERDICT: ALL FACTORS MATCH")
else:
    print("CONVERSION FACTORS VERDICT: SOME FACTORS DO NOT MATCH")
print("=" * 110)
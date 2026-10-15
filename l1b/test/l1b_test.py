import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import glob

# File paths
dir_truth = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\input"
dir_my_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output"
dir_my_no_eq = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output_no_equalization"
dir_ref_output = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\output"

output_test_dir = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output_test"
os.makedirs(output_test_dir, exist_ok=True)

threshold_pct = 1e-3

# Cross-Validation
ref_files = glob.glob(os.path.join(dir_ref_output, "*.nc"))

print("=" * 75)
print(f"{'FILE':<35} | {'MAX ABS ERR':<14} | {'MAX REL ERR (%)':<15} | {'STATUS'}")
print("=" * 75)

all_passed = True

for path_ref in sorted(ref_files):
    filename = os.path.basename(path_ref)
    path_my = os.path.join(dir_my_output, filename)

    if not os.path.exists(path_my):
        print(f"{filename:<35} | {'MISSING':<14} | {'MISSING':<15} | [FAIL]")
        all_passed = False
        continue

    with Dataset(path_my, "r") as ds_my, Dataset(path_ref, "r") as ds_ref:
        var_my = list(ds_my.variables.keys())[-1]
        var_ref = list(ds_ref.variables.keys())[-1]

        data_my = ds_my.variables[var_my][:]
        data_ref = ds_ref.variables[var_ref][:]

    diff_abs = np.abs(data_my - data_ref)
    max_abs_err = np.max(diff_abs)

    mask = np.abs(data_ref) > 1e-12
    rel_err_pct = np.zeros_like(diff_abs)
    rel_err_pct[mask] = (diff_abs[mask] / np.abs(data_ref[mask])) * 100.0
    max_rel_err_pct = np.max(rel_err_pct)

    status = "[PASS]" if max_rel_err_pct < threshold_pct else "[FAIL]"
    if status == "[FAIL]":
        all_passed = False

    print(f"{filename:<35} | {max_abs_err:<14.6e} | {max_rel_err_pct:<15.6e} | {status}")

print("=" * 75)
if all_passed:
    print("GLOBAL TEST VERDICT: ALL FILES PASSED (< 1e-3 %)")
else:
    print("GLOBAL TEST VERDICT: SOME FILES FAILED")
print("=" * 75)

# Comparatives plots
bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

for band in bands:
    file_truth = os.path.join(dir_truth, f"ism_toa_isrf_{band}.nc")
    file_eq = os.path.join(dir_my_output, f"l1b_toa_{band}.nc")
    file_no_eq = os.path.join(dir_my_no_eq, f"l1b_toa_{band}.nc")

    if not (os.path.exists(file_truth) and os.path.exists(file_eq) and os.path.exists(file_no_eq)):
        continue

    with Dataset(file_truth, "r") as ds:
        toa_truth = ds.variables[list(ds.variables.keys())[-1]][:]

    with Dataset(file_eq, "r") as ds:
        toa_eq = ds.variables[list(ds.variables.keys())[-1]][:]

    with Dataset(file_no_eq, "r") as ds:
        toa_no_eq = ds.variables[list(ds.variables.keys())[-1]][:]


    mid_alt = toa_truth.shape[0] // 2
    profile_truth = toa_truth[mid_alt, :]
    profile_eq = toa_eq[mid_alt, :]
    profile_no_eq = toa_no_eq[mid_alt, :]


    plt.figure(figsize=(9, 5.5))
    plt.plot(profile_eq, color="black", label="TOA L1B with eq", linewidth=1.2)
    plt.plot(profile_no_eq, color="red", label="TOA L1B no eq", linewidth=1.2)
    plt.plot(profile_truth, color="blue", label="TOA after the ISRF", linewidth=1.2)

    plt.title(f"Effect of the Equalization for {band}")
    plt.xlabel("ACT pixel [-]")
    plt.ylabel("TOA [mW/m2/sr]")
    plt.grid(True)
    plt.legend(loc="upper left", fontsize=8)

    save_fig_path = os.path.join(output_test_dir, f"fig_8_3_equalization_{band}.png")
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.close()

print(f"\nAll comparison figures successfully saved in: {output_test_dir}")
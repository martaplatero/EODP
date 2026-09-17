import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# File paths
path_truth = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc"
path_l1b_eq = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output\l1b_toa_VNIR-0.nc"
path_l1b_no_eq = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output_no_equalization\l1b_toa_VNIR-0.nc"

output_dir = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1B\my_output_test"
os.makedirs(output_dir, exist_ok=True)
save_fig_path = os.path.join(output_dir, "fig_8_3_equalization_vnir0.png")

# Read netCDF files
with Dataset(path_truth, "r") as ds:
    var_name = list(ds.variables.keys())[-1]
    toa_truth = ds.variables[var_name][:]

with Dataset(path_l1b_eq, "r") as ds:
    var_name = list(ds.variables.keys())[-1]
    toa_eq = ds.variables[var_name][:]

with Dataset(path_l1b_no_eq, "r") as ds:
    var_name = list(ds.variables.keys())[-1]
    toa_no_eq = ds.variables[var_name][:]

# Take slice at central ALT line
mid_alt = toa_truth.shape[0] // 2
profile_truth = toa_truth[mid_alt, :]
profile_eq = toa_eq[mid_alt, :]
profile_no_eq = toa_no_eq[mid_alt, :]

# Plot
plt.figure(figsize=(9, 5.5))
plt.plot(profile_eq, color="black", label="TOA L1B with eq", linewidth=1.2)
plt.plot(profile_no_eq, color="red", label="TOA L1B no eq", linewidth=1.2)
plt.plot(profile_truth, color="blue", label="TOA after the ISRF", linewidth=1.2)

plt.title("Effect of the Equalization for VNIR-0")
plt.xlabel("ACT pixel [-]")
plt.ylabel("TOA [mW/m2/sr]")
plt.grid(True)
plt.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(save_fig_path, dpi=300)
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# File paths
base = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP-TS-E2E"
dir_my_l1b = os.path.join(base, "my_output_l1b")      # own L1B output of the E2E chain

output_test_dir = os.path.join(base, "my_output_test")
os.makedirs(output_test_dir, exist_ok=True)

bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
nbins = 100


def read_toa(directory, band):
    with Dataset(os.path.join(directory, f"l1b_toa_{band}.nc"), "r") as ds:
        return np.array(ds.variables["toa"][:], dtype=float)


# ----------------------------------------------------------------------------------------------
# 1) Statistics of the L1B TOA
# ----------------------------------------------------------------------------------------------
print("=" * 95)
print("E2E - L1B TOA STATISTICS [mW/m2/sr]")
print("=" * 95)
print(f"{'BAND':<8} | {'MIN':<8} | {'MAX':<8} | {'MEAN':<8} | {'MEDIAN':<8} | {'STD':<8} | {'P1':<8} | {'P99':<8}")
print("-" * 95)

toas = {}
for band in bands:
    toa = read_toa(dir_my_l1b, band)
    toas[band] = toa
    p1, p99 = np.percentile(toa, [1, 99])
    print(f"{band:<8} | {toa.min():<8.3f} | {toa.max():<8.3f} | {toa.mean():<8.3f} | {np.median(toa):<8.3f} | "
          f"{toa.std():<8.3f} | {p1:<8.3f} | {p99:<8.3f}")
print("=" * 95)

# ----------------------------------------------------------------------------------------------
# 2) TOA image and histogram side by side, one figure per band
# ----------------------------------------------------------------------------------------------
for band, toa in toas.items():
    nalt, nact = toa.shape
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.15, 1]})

    # TOA image (same orientation as Panoply: ALT lines in the vertical axis, origin at the bottom)
    im = ax_img.imshow(toa, origin="lower", cmap="jet", aspect="auto",
                       vmin=toa.min(), vmax=toa.max(), interpolation="nearest")
    ax_img.set_title(f"L1B TOA - {band}")
    ax_img.set_xlabel("ACT columns [-]")
    ax_img.set_ylabel("ALT lines [-]")
    fig.colorbar(im, ax=ax_img, label="TOA [mW/m2/sr]")

    # Histogram of the radiances (same horizontal range as the colour bar)
    counts, edges, patches = ax_hist.hist(toa.ravel(), bins=nbins, range=(toa.min(), toa.max()),
                                          edgecolor="black", linewidth=0.3)
    # Colour each bin with the colour it has in the image, so both plots can be read together
    cmap = plt.get_cmap("jet")
    centres = 0.5 * (edges[:-1] + edges[1:])
    for c, p in zip(centres, patches):
        p.set_facecolor(cmap((c - toa.min()) / (toa.max() - toa.min())))
    ax_hist.axvline(toa.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean = {toa.mean():.2f}")
    ax_hist.axvline(np.median(toa), color="gray", linestyle=":", linewidth=1.5, label=f"Median = {np.median(toa):.2f}")
    ax_hist.set_title(f"Histogram of the L1B TOA - {band} ({toa.size} pixels)")
    ax_hist.set_xlabel("TOA [mW/m2/sr]")
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.grid(True, alpha=0.5)
    ax_hist.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_test_dir, f"fig_e2e_l1b_image_hist_{band}.png"), dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 3) Histograms of the four bands in the same plot
# ----------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 5.5))
for band, toa in toas.items():
    plt.hist(toa.ravel(), bins=nbins, histtype="step", linewidth=1.5, label=band)
plt.title("Histogram of the L1B TOA - all bands")
plt.xlabel("TOA [mW/m2/sr]")
plt.ylabel("Number of pixels")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_test_dir, "fig_e2e_l1b_hist_all_bands.png"), dpi=300)
plt.close()

print(f"\nAll E2E figures successfully saved in: {output_test_dir}")

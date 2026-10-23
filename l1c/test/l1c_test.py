import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# File paths
base = r"C:\Users\marta\Desktop\MISE\2 YEAR\Earth Observation Data Processing\EODP_TER_2021\EODP-TS-L1C"
dir_gm = os.path.join(base, "input", "gm_alt100_act_150")
dir_l1b = os.path.join(base, "input", "l1b_output")
dir_my_output = os.path.join(base, "my_output")
dir_ref_output = os.path.join(base, "output")

output_test_dir = os.path.join(base, "my_output_test")
os.makedirs(output_test_dir, exist_ok=True)

bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
R_EARTH = 6371008.8          # [m] mean Earth radius (IUGG)
MGRS_GRID = 100.0            # [m] L1C grid size (MGRS precision 3)


# ----------------------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------------------
def read_geodetic(directory):
    with Dataset(os.path.join(directory, "geolocation.nc"), "r") as ds:
        lat = np.array(ds.groups["projection"].variables["latitude"][:], dtype=float)
        lon = np.array(ds.groups["projection"].variables["longitude"][:], dtype=float)
    return lat, lon


def read_l1b(directory, band):
    with Dataset(os.path.join(directory, f"l1b_toa_{band}.nc"), "r") as ds:
        return np.array(ds.variables["toa"][:], dtype=float)


def read_l1c(directory, band):
    with Dataset(os.path.join(directory, f"l1c_toa_{band}.nc"), "r") as ds:
        toa = np.array(ds.variables["toa"][:], dtype=float)
        lat = np.array(ds.variables["lat"][:], dtype=float)
        lon = np.array(ds.variables["lon"][:], dtype=float)
    return toa, lat, lon


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance [m] between points given in degrees (vectorised)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def stats(d):
    return f"mean {d.mean():8.2f} m | min {d.min():8.2f} m | max {d.max():8.2f} m | std {d.std():6.2f} m"


lat_l1b, lon_l1b = read_geodetic(dir_gm)
nalt, nact = lat_l1b.shape
ia, ic = nalt // 2, nact // 2

# ----------------------------------------------------------------------------------------------
# 1) GSD / SSD of the L1B (Haversine, central row and column)
# ----------------------------------------------------------------------------------------------
# ACT: distance between consecutive columns of the central row
d_act = haversine(lat_l1b[ia, :-1], lon_l1b[ia, :-1], lat_l1b[ia, 1:], lon_l1b[ia, 1:])
# ALT: distance between consecutive lines of the central column
d_alt = haversine(lat_l1b[:-1, ic], lon_l1b[:-1, ic], lat_l1b[1:, ic], lon_l1b[1:, ic])

print("=" * 100)
print(f"L1B GROUND SAMPLING DISTANCE (Haversine, R = {R_EARTH / 1000:.1f} km) - L1B grid {nalt} ALT x {nact} ACT")
print("=" * 100)
print(f"ACT (central row, ALT {ia})    : {stats(d_act)}")
print(f"ALT (central column, ACT {ic}) : {stats(d_alt)}")

# Cross-check with geopy (geodesic distance on the WGS84 ellipsoid), if installed
try:
    from geopy.distance import geodesic
    g_act = np.array([geodesic((lat_l1b[ia, j], lon_l1b[ia, j]), (lat_l1b[ia, j + 1], lon_l1b[ia, j + 1])).m
                      for j in range(nact - 1)])
    g_alt = np.array([geodesic((lat_l1b[i, ic], lon_l1b[i, ic]), (lat_l1b[i + 1, ic], lon_l1b[i + 1, ic])).m
                      for i in range(nalt - 1)])
    print("-" * 100)
    print("Cross-check with geopy (WGS84 geodesic):")
    print(f"ACT (central row)    : {stats(g_act)}")
    print(f"ALT (central column) : {stats(g_alt)}")
    print(f"Haversine vs geodesic: ACT {100 * (d_act.mean() / g_act.mean() - 1):+.3f} %, "
          f"ALT {100 * (d_alt.mean() / g_alt.mean() - 1):+.3f} %")
except ImportError:
    print("(geopy not installed: skipping the geodesic cross-check)")

# Footprint (swath and length of the scene)
swath = haversine(lat_l1b[ia, 0], lon_l1b[ia, 0], lat_l1b[ia, -1], lon_l1b[ia, -1])
length = haversine(lat_l1b[0, ic], lon_l1b[0, ic], lat_l1b[-1, ic], lon_l1b[-1, ic])
print("-" * 100)
print(f"Scene size: ACT swath {swath / 1000:.3f} km | ALT length {length / 1000:.3f} km")

# ----------------------------------------------------------------------------------------------
# 2) L1C grid: number of points and spacing
# ----------------------------------------------------------------------------------------------
toa_c0, lat_l1c, lon_l1c = read_l1c(dir_my_output, bands[0])
n_l1b, n_l1c = lat_l1b.size, lat_l1c.size

# Spacing of the L1C grid: distance from each point to its nearest neighbour
lat_r, lon_r = np.radians(lat_l1c), np.radians(lon_l1c)
x = R_EARTH * np.cos(np.radians(lat_l1c.mean())) * lon_r            # local equirectangular projection
y = R_EARTH * lat_r
nn = np.empty(n_l1c)
for k in range(n_l1c):
    d = np.hypot(x - x[k], y - y[k])
    d[k] = np.inf
    nn[k] = d.min()

expected_pts = (swath * length) / MGRS_GRID ** 2
print("\n" + "=" * 100)
print("L1B vs L1C GRIDS")
print("=" * 100)
print(f"Number of points       : L1B {n_l1b} | L1C {n_l1c} | ratio L1B/L1C = {n_l1b / n_l1c:.2f}")
print(f"L1C grid spacing       : nearest neighbour {stats(nn)} (nominal {MGRS_GRID:.0f} m)")
print(f"Pixel area ratio       : (GSD_ACT x GSD_ALT) / {MGRS_GRID:.0f}^2 = "
      f"{d_act.mean() * d_alt.mean() / MGRS_GRID ** 2:.3f}")
print(f"Expected L1C points    : footprint area / {MGRS_GRID:.0f}^2 = {expected_pts:.0f}")

# ----------------------------------------------------------------------------------------------
# 3) Plot of the L1B and L1C grids in the same figure
# ----------------------------------------------------------------------------------------------
corners = lambda m: [m[0, 0], m[0, -1], m[-1, -1], m[-1, 0], m[0, 0]]

fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.4, 1]})
for a in ax:
    a.plot(lon_l1b.ravel(), lat_l1b.ravel(), ".", color="tab:blue", markersize=1.5, label=f"L1B grid ({n_l1b} px)")
    a.plot(lon_l1c, lat_l1c, "o", color="tab:red", markersize=2.5, mfc="none", label=f"L1C MGRS grid ({n_l1c} pts)")
    a.plot(corners(lon_l1b), corners(lat_l1b), "k-", linewidth=1.2, label="L1B footprint")
    a.set_xlabel("Longitude [deg]")
    a.set_ylabel("Latitude [deg]")
    a.grid(True)
ax[0].set_title("L1B (irregular) and L1C (MGRS, 100 m) grids")
ax[0].legend(loc="upper right", fontsize=8)
# Zoom around the centre of the scene to see the individual points
zlat, zlon = lat_l1b[ia, ic], lon_l1b[ia, ic]
dz = 0.004
ax[1].set_xlim(zlon - dz, zlon + dz)
ax[1].set_ylim(zlat - dz, zlat + dz)
ax[1].set_title("Zoom at the centre of the scene")
plt.tight_layout()
plt.savefig(os.path.join(output_test_dir, "fig_l1b_l1c_grids.png"), dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 4) L1B vs L1C radiances (same colour scale), one figure per band
# ----------------------------------------------------------------------------------------------
for band in bands:
    toa_b = read_l1b(dir_l1b, band)
    toa_c, la_c, lo_c = read_l1c(dir_my_output, band)
    vmin, vmax = np.nanpercentile(toa_b, [1, 99])

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    im = ax[0].scatter(lon_l1b.ravel(), lat_l1b.ravel(), c=toa_b.ravel(), s=4, cmap="viridis", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"L1B TOA {band}")
    ax[1].scatter(lo_c, la_c, c=toa_c, s=10, cmap="viridis", vmin=vmin, vmax=vmax)
    nan = np.isnan(toa_c)
    if nan.any():
        ax[1].plot(lo_c[nan], la_c[nan], "x", color="red", markersize=4, label=f"NaN ({nan.sum()} pts)")
        ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title(f"L1C TOA {band}")
    for a in ax:
        a.plot(corners(lon_l1b), corners(lat_l1b), "k-", linewidth=1)
        a.set_xlabel("Longitude [deg]")
        a.grid(True)
    ax[0].set_ylabel("Latitude [deg]")
    fig.colorbar(im, ax=ax, label="TOA [mW/m2/sr]")
    plt.savefig(os.path.join(output_test_dir, f"fig_l1b_vs_l1c_toa_{band}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



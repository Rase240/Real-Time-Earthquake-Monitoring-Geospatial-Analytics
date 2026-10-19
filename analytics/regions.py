"""
The fixed geographic aggregation used by the forecasting and anomaly modules.

Regions are cells of a 10 x 10 degree latitude/longitude grid: 18 rows x 36
columns = 648 cells. A cell contains latitudes [lat_lo, lat_lo + 10) and
longitudes [lon_lo, lon_lo + 10); latitude 90 belongs to the top row and
longitude 180 is wrapped to -180. Cells are equal-angle, not equal-area: a cell
at 60 degrees latitude covers half the area of an equatorial cell. Counts are
therefore only compared within a cell over time, never across cells.

A 10-degree cell (~1100 km) is coarse enough that M4.5+ weekly counts are not
almost all zero, yet fine enough to separate the main subduction systems.

Weekly count matrices
---------------------
`count_matrix` returns C[r, w] = number of events with magnitude >= threshold
whose time t satisfies epoch + 7w days <= t < epoch + 7(w+1) days. Week bins are
half-open, so an event exactly on a boundary belongs to the later week.
"""
import math

import numpy as np

CELL_DEG = 10
N_LAT = 180 // CELL_DEG
N_LON = 360 // CELL_DEG
N_REGIONS = N_LAT * N_LON
WEEK_MS = 7 * 86_400_000
MAG_EPS = 1e-9   # catalogue magnitudes are decimals; 4.5 must satisfy ">= 4.5"


def region_index(lats, lons):
    """Region index for each point (vectorised)."""
    lats = np.asarray(lats, float)
    lons = (np.asarray(lons, float) + 180.0) % 360.0 - 180.0
    i = np.clip(np.floor((lats + 90.0) / CELL_DEG), 0, N_LAT - 1).astype(int)
    j = np.clip(np.floor((lons + 180.0) / CELL_DEG), 0, N_LON - 1).astype(int)
    return i * N_LON + j


def bounds(idx):
    i, j = divmod(int(idx), N_LON)
    lat_lo, lon_lo = -90 + i * CELL_DEG, -180 + j * CELL_DEG
    return {"lat_min": lat_lo, "lat_max": lat_lo + CELL_DEG, "lon_min": lon_lo, "lon_max": lon_lo + CELL_DEG}


def region_id(idx):
    """Stable text id, e.g. 'R+30+140' for the cell starting at 30N, 140E."""
    b = bounds(idx)
    return f"R{b['lat_min']:+03d}{b['lon_min']:+04d}"


def parse_region_id(text):
    """Inverse of region_id. Returns the index or None if the id is not a valid cell."""
    if not isinstance(text, str) or len(text) != 8 or text[0] != "R":
        return None
    try:
        lat, lon = int(text[1:4]), int(text[4:8])
    except ValueError:
        return None
    if lat % CELL_DEG or lon % CELL_DEG or not (-90 <= lat < 90) or not (-180 <= lon < 180):
        return None
    if text[1] not in "+-" or text[4] not in "+-":
        return None
    return int(region_index([lat + 0.5], [lon + 0.5])[0])


def label(idx):
    b = bounds(idx)

    def la(v):
        return f"{abs(v)}°{'N' if v >= 0 else 'S'}" if v else "0°"

    def lo(v):
        return f"{abs(v)}°{'E' if v >= 0 else 'W'}" if v not in (0, 180, -180) else f"{abs(v)}°"
    return f"{la(b['lat_min'])}–{la(b['lat_max'])}, {lo(b['lon_min'])}–{lo(b['lon_max'])}"


def neighbors(idx):
    """The up-to-8 surrounding cells. Longitude wraps at the antimeridian; latitude does not wrap at the poles."""
    i, j = divmod(int(idx), N_LON)
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0 or not (0 <= i + di < N_LAT):
                continue
            out.append((i + di) * N_LON + (j + dj) % N_LON)
    return out


def adjacency():
    """(N_REGIONS, N_REGIONS) 0/1 matrix, A[r, s] = 1 when s is a neighbour of r."""
    A = np.zeros((N_REGIONS, N_REGIONS), dtype=np.int32)
    for r in range(N_REGIONS):
        A[r, neighbors(r)] = 1
    return A


def count_matrix(times, mags, lats, lons, threshold, epoch_ms, n_weeks):
    """Weekly counts C[r, w] of events with mag >= threshold (see module docstring)."""
    times = np.asarray(times, dtype=np.int64)
    mags = np.asarray(mags, float)
    C = np.zeros((N_REGIONS, n_weeks), dtype=np.int64)
    with np.errstate(invalid="ignore"):
        keep = (mags >= threshold - MAG_EPS) & (times >= epoch_ms) & (times < epoch_ms + n_weeks * WEEK_MS)
    if not keep.any():
        return C
    w = (times[keep] - epoch_ms) // WEEK_MS
    r = region_index(np.asarray(lats, float)[keep], np.asarray(lons, float)[keep])
    np.add.at(C, (r, w), 1)
    return C


def week_epoch(origin_ms, n_weeks):
    """Epoch for a matrix whose last column ends exactly at origin_ms."""
    return int(origin_ms) - n_weeks * WEEK_MS


def cell_area_fraction(idx):
    """Area of the cell relative to an equatorial cell (sin-difference of its latitude bounds)."""
    b = bounds(idx)
    eq = math.sin(math.radians(CELL_DEG)) - math.sin(0.0)
    return (math.sin(math.radians(b["lat_max"])) - math.sin(math.radians(b["lat_min"]))) / eq

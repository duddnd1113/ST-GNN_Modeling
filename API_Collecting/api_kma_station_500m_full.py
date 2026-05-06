import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path.cwd()

STATION_PATH = BASE_DIR / "seoul_stations.csv"

OUT_DIR = BASE_DIR / "kma_station_500m_weather"
OUT_DIR.mkdir(exist_ok=True)

MONTHLY_DIR = OUT_DIR / "monthly"
MONTHLY_DIR.mkdir(exist_ok=True)

STATION_MONTHLY_DIR = OUT_DIR / "station_monthly"
STATION_MONTHLY_DIR.mkdir(exist_ok=True)

RAW_DIR = OUT_DIR / "raw_txt"
RAW_DIR.mkdir(exist_ok=True)

GRID_MAP_PATH = OUT_DIR / "station_to_kma_500m_grid.csv"
FINAL_PATH = OUT_DIR / "station_hourly_kma_weather_all.csv"
FAILED_PATH = OUT_DIR / "failed_requests.csv"
TIME_WARN_PATH = OUT_DIR / "time_warnings.csv"
LOG_PATH = OUT_DIR / "progress_log.csv"


# ============================================================
# 2. API settings
# ============================================================

URL = "https://apihub.kma.go.kr/api/typ01/url/sfc_nc_var.php"
AUTH_KEY = "AcGHCAQmTWeBhwgEJq1nuw"

OBS_COLS = [
    "ta",
    "hm",
    "td",
    "wd_10m",
    "ws_10m",
    "uu",
    "vv",
    "pa",
    "ps",
    "rn_ox",
    "rn_15m",
    "rn_60m",
    "rn_day",
    "vs",
    "ta_chi",
    "sd_tot",
    "sd_day",
    "sd_3hr",
    "sd_24h",
]

OBS = ",".join(OBS_COLS)
ITV = "60"

START = "2023-10-01 00:00"
END = "2024-03-31 23:00"


# ============================================================
# 3. Utility functions
# ============================================================

def make_month_chunks(start, end):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    month_starts = pd.date_range(
        start=start_ts.normalize().replace(day=1),
        end=end_ts,
        freq="MS"
    )

    chunks = []

    for ms in month_starts:
        s = max(ms, start_ts)
        e = min(ms + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1), end_ts)

        if s <= e:
            chunks.append((s, e))

    return chunks


def make_daily_chunks(month_start, month_end):
    days = pd.date_range(
        start=month_start.normalize(),
        end=month_end.normalize(),
        freq="D"
    )

    chunks = []

    for d in days:
        s = max(d, month_start)
        e = min(d + pd.Timedelta(hours=23), month_end)

        if s <= e:
            chunks.append((s, e))

    return chunks


def safe_filename(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def append_csv_row(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig"
    )


def fetch_with_retry(params, max_retries=5, timeout=90):
    last_error = None

    for attempt in range(max_retries):
        try:
            r = requests.get(URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.text, None

        except Exception as e:
            last_error = str(e)
            wait = 3 * (attempt + 1)
            print(f"    retry {attempt + 1}/{max_retries}: {last_error}")
            time.sleep(wait)

    return None, last_error


def parse_kma_response(text, station_name, kma_grid_id, kma_lon, kma_lat):
    rows = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if not line[0].isdigit():
            continue

        parts = line.replace(",", " ").split()

        if len(parts) < 1 + len(OBS_COLS):
            continue

        row = {
            "stationName": station_name,
            "kma_grid_id": kma_grid_id,
            "kma_lon": kma_lon,
            "kma_lat": kma_lat,
            "time": parts[0],
        }

        for col, value in zip(OBS_COLS, parts[1:1 + len(OBS_COLS)]):
            try:
                row[col] = float(value)
            except ValueError:
                row[col] = np.nan

        rows.append(row)

    return rows


def validate_expected_hours(rows, station_name, kma_grid_id, kma_lon, kma_lat, tm1, tm2):
    if not rows:
        return False, {
            "stationName": station_name,
            "kma_grid_id": kma_grid_id,
            "kma_lon": kma_lon,
            "kma_lat": kma_lat,
            "tm1": tm1,
            "tm2": tm2,
            "expected_count": np.nan,
            "actual_count": 0,
            "missing_count": np.nan,
            "extra_count": np.nan,
            "message": "zero parsed rows",
        }

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    expected = pd.date_range(
        start=pd.to_datetime(tm1, format="%Y%m%d%H%M"),
        end=pd.to_datetime(tm2, format="%Y%m%d%H%M"),
        freq="h"
    )

    actual = pd.DatetimeIndex(df["time"].dropna().sort_values().unique())

    missing = expected.difference(actual)
    extra = actual.difference(expected)

    ok = (len(missing) == 0) and (len(extra) == 0)

    warn_row = {
        "stationName": station_name,
        "kma_grid_id": kma_grid_id,
        "kma_lon": kma_lon,
        "kma_lat": kma_lat,
        "tm1": tm1,
        "tm2": tm2,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_times": ",".join([t.strftime("%Y%m%d%H%M") for t in missing]),
        "extra_times": ",".join([t.strftime("%Y%m%d%H%M") for t in extra]),
        "message": "ok" if ok else "time mismatch",
    }

    return ok, warn_row


# ============================================================
# 4. Load Seoul stations and build 500m query points
# ============================================================

stations = pd.read_csv(STATION_PATH, encoding="utf-8-sig")

stations = stations.rename(columns={
    "dmX": "lat",
    "dmY": "lon"
})

station_gdf = gpd.GeoDataFrame(
    stations.copy(),
    geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=5179)

station_gdf["x_m"] = station_gdf.geometry.x
station_gdf["y_m"] = station_gdf.geometry.y

station_gdf["x500"] = np.floor(station_gdf["x_m"] / 500).astype(int)
station_gdf["y500"] = np.floor(station_gdf["y_m"] / 500).astype(int)

station_gdf["kma_grid_id"] = (
    station_gdf["x500"].astype(str) + "_" + station_gdf["y500"].astype(str)
)

station_gdf["kma_center_x_m"] = (station_gdf["x500"] + 0.5) * 500
station_gdf["kma_center_y_m"] = (station_gdf["y500"] + 0.5) * 500

station_grid_map = station_gdf.drop(columns="geometry").copy()
station_grid_map.to_csv(GRID_MAP_PATH, index=False, encoding="utf-8-sig")

kma_points_gdf = station_gdf[
    [
        "stationName",
        "mangName",
        "addr",
        "lat",
        "lon",
        "kma_grid_id",
        "kma_center_x_m",
        "kma_center_y_m",
    ]
].copy()

kma_points_gdf = gpd.GeoDataFrame(
    kma_points_gdf,
    geometry=gpd.points_from_xy(
        kma_points_gdf["kma_center_x_m"],
        kma_points_gdf["kma_center_y_m"],
    ),
    crs="EPSG:5179",
)

kma_points_4326 = kma_points_gdf.to_crs(epsg=4326)

kma_points_gdf["kma_lon"] = kma_points_4326.geometry.x
kma_points_gdf["kma_lat"] = kma_points_4326.geometry.y

print("Stations:", len(kma_points_gdf))
print("Unique 500m grids:", kma_points_gdf["kma_grid_id"].nunique())
print("Station-grid map saved:", GRID_MAP_PATH)


# ============================================================
# 5. Monthly scraping with station-month + raw_txt checkpoints
# ============================================================

chunks = make_month_chunks(START, END)

for month_start, month_end in chunks:
    month_key = month_start.strftime("%Y_%m")

    monthly_path = MONTHLY_DIR / f"kma_station_weather_{month_key}.csv"
    month_station_dir = STATION_MONTHLY_DIR / month_key
    month_station_dir.mkdir(exist_ok=True)

    if monthly_path.exists():
        print(f"[SKIP MONTH] {month_key} already exists: {monthly_path}")
        continue

    print(f"\n[START MONTH] {month_key}: {month_start} ~ {month_end}")

    daily_chunks = make_daily_chunks(month_start, month_end)
    total_jobs = len(kma_points_gdf) * len(daily_chunks)
    job_count = 0
    failed_month_count = 0
    time_warn_month_count = 0

    for _, row in kma_points_gdf.iterrows():
        station_name = row["stationName"]
        kma_grid_id = row["kma_grid_id"]
        kma_lon = row["kma_lon"]
        kma_lat = row["kma_lat"]

        station_file = month_station_dir / f"{safe_filename(station_name)}_{safe_filename(kma_grid_id)}_{month_key}.csv"

        if station_file.exists():
            print(f"  [SKIP STATION] {station_name} already done for {month_key}")
            job_count += len(daily_chunks)
            continue

        station_rows = []

        print(f"  [START STATION] {station_name} ({month_key})")

        for day_start, day_end in daily_chunks:
            job_count += 1

            tm1 = day_start.strftime("%Y%m%d%H%M")
            tm2 = day_end.strftime("%Y%m%d%H%M")

            if job_count == 1 or job_count % 10 == 0:
                print(
                    f"  [{job_count}/{total_jobs}] "
                    f"{station_name} | {tm1} ~ {tm2} | "
                    f"lat={kma_lat:.6f}, lon={kma_lon:.6f}"
                )

            raw_name = safe_filename(
                f"{month_key}_{station_name}_{kma_grid_id}_{tm1}_{tm2}.txt"
            )
            raw_path = RAW_DIR / raw_name

            if raw_path.exists():
                text = raw_path.read_text(encoding="utf-8")
            else:
                params = {
                    "tm1": tm1,
                    "tm2": tm2,
                    "lon": kma_lon,
                    "lat": kma_lat,
                    "obs": OBS,
                    "itv": ITV,
                    "help": "0",
                    "authKey": AUTH_KEY,
                }

                text, error = fetch_with_retry(params)

                if text is None:
                    failed_month_count += 1

                    fail_row = {
                        "month": month_key,
                        "stationName": station_name,
                        "kma_grid_id": kma_grid_id,
                        "kma_lon": kma_lon,
                        "kma_lat": kma_lat,
                        "tm1": tm1,
                        "tm2": tm2,
                        "error": error,
                    }

                    append_csv_row(FAILED_PATH, fail_row)

                    print(
                        f"  [FAILED] {station_name} | {tm1} ~ {tm2} | "
                        f"lat={kma_lat:.6f}, lon={kma_lon:.6f}"
                    )
                    continue

                raw_path.write_text(text, encoding="utf-8")

            rows = parse_kma_response(
                text=text,
                station_name=station_name,
                kma_grid_id=kma_grid_id,
                kma_lon=kma_lon,
                kma_lat=kma_lat,
            )

            ok, warn_row = validate_expected_hours(
                rows=rows,
                station_name=station_name,
                kma_grid_id=kma_grid_id,
                kma_lon=kma_lon,
                kma_lat=kma_lat,
                tm1=tm1,
                tm2=tm2,
            )

            if not ok:
                time_warn_month_count += 1
                append_csv_row(TIME_WARN_PATH, warn_row)

                print(
                    f"  [TIME WARN] {station_name} | {tm1} ~ {tm2} | "
                    f"expected={warn_row['expected_count']}, actual={warn_row['actual_count']}, "
                    f"missing={warn_row['missing_count']}, extra={warn_row['extra_count']}"
                )

            station_rows.extend(rows)

            time.sleep(0.25)

        station_df = pd.DataFrame(station_rows)

        if not station_df.empty:
            station_df["time"] = pd.to_datetime(station_df["time"], errors="coerce")
            station_df = station_df.drop_duplicates(
                subset=["stationName", "kma_grid_id", "time"]
            ).sort_values(["stationName", "time"])

        station_df.to_csv(station_file, index=False, encoding="utf-8-sig")
        print(f"  [SAVED STATION] {station_file} rows={len(station_df)}")

    # ========================================================
    # Combine station-month files into monthly file
    # ========================================================

    station_files = sorted(month_station_dir.glob("*.csv"))
    month_dfs = []

    for p in station_files:
        temp = pd.read_csv(p, encoding="utf-8-sig")
        month_dfs.append(temp)

    if month_dfs:
        monthly_df = pd.concat(month_dfs, ignore_index=True)
        monthly_df["time"] = pd.to_datetime(monthly_df["time"], errors="coerce")
        monthly_df = monthly_df.drop_duplicates(
            subset=["stationName", "kma_grid_id", "time"]
        ).sort_values(["stationName", "time"])
    else:
        monthly_df = pd.DataFrame()

    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    progress_row = {
        "month": month_key,
        "start": month_start,
        "end": month_end,
        "rows": len(monthly_df),
        "failed": failed_month_count,
        "time_warnings": time_warn_month_count,
        "output": str(monthly_path),
    }

    append_csv_row(LOG_PATH, progress_row)

    print(f"[SAVED MONTH] {monthly_path}")
    print(
        f"[MONTH DONE] {month_key} | rows={len(monthly_df)}, "
        f"failed={failed_month_count}, time_warnings={time_warn_month_count}"
    )


# ============================================================
# 6. Combine monthly files into one final file
# ============================================================

monthly_files = sorted(MONTHLY_DIR.glob("kma_station_weather_*.csv"))
dfs = []

for path in monthly_files:
    temp = pd.read_csv(path, encoding="utf-8-sig")
    dfs.append(temp)

if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    final_df["time"] = pd.to_datetime(final_df["time"], errors="coerce")

    final_df = final_df.drop_duplicates(
        subset=["stationName", "kma_grid_id", "time"]
    ).sort_values(["stationName", "time"])

    final_df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")

    print("\n[FINAL SAVED]", FINAL_PATH)
    print("rows:", len(final_df))
    print("stations:", final_df["stationName"].nunique())
    print("time range:", final_df["time"].min(), "~", final_df["time"].max())

else:
    print("No monthly files found.")



# %%

import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd()
FINAL_PATH = BASE_DIR / "kma_station_500m_weather" / "station_hourly_kma_weather_all.csv"

df = pd.read_csv(FINAL_PATH, encoding="utf-8-sig")

df = df.rename(columns={
    "stationName": "측정소명",
    "kma_lat": "dmX",
    "kma_lon": "dmY"
})

# overwrite or save new file
SAVE_PATH = FINAL_PATH  # or change filename if you want backup
df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("Renaming complete.")

print(df["dmX"].min(), df["dmX"].max())  # should be ~36–38
print(df["dmY"].min(), df["dmY"].max())  # should be ~126–128

# %%

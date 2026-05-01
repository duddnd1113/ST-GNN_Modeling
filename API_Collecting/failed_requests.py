import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "kma_station_500m_weather"
MONTHLY_DIR = OUT_DIR / "monthly"
STATION_MONTHLY_DIR = OUT_DIR / "station_monthly"
RAW_DIR = OUT_DIR / "raw_txt"

FAILED_PATH = OUT_DIR / "failed_requests.csv"
RECOVERED_LOG_PATH = OUT_DIR / "recovered_failed_requests.csv"
STILL_FAILED_PATH = OUT_DIR / "still_failed_requests.csv"
FINAL_PATH = OUT_DIR / "station_hourly_kma_weather_all.csv"


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


# ============================================================
# 3. Utility functions
# ============================================================

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


def validate_expected_hours(rows, tm1, tm2):
    if not rows:
        return False, 0, np.nan, np.nan

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

    return ok, len(actual), len(missing), len(extra)


# ============================================================
# 4. Retry failed requests
# ============================================================

if not FAILED_PATH.exists():
    print("No failed_requests.csv found. Nothing to recover.")
    raise SystemExit

failed = pd.read_csv(FAILED_PATH, encoding="utf-8-sig")

if failed.empty:
    print("failed_requests.csv is empty. Nothing to recover.")
    raise SystemExit

failed = failed.drop_duplicates(
    subset=["month", "stationName", "kma_grid_id", "kma_lon", "kma_lat", "tm1", "tm2"]
)

print("Failed requests to retry:", len(failed))

recovered_count = 0
still_failed_count = 0

for idx, row in failed.iterrows():
    month_key = row["month"]
    station_name = row["stationName"]
    kma_grid_id = row["kma_grid_id"]
    kma_lon = float(row["kma_lon"])
    kma_lat = float(row["kma_lat"])
    tm1 = str(row["tm1"])
    tm2 = str(row["tm2"])

    print(
        f"\n[{idx + 1}/{len(failed)}] RETRY "
        f"{station_name} | {tm1} ~ {tm2} | "
        f"lat={kma_lat:.6f}, lon={kma_lon:.6f}"
    )

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
        still_failed_count += 1

        append_csv_row(STILL_FAILED_PATH, {
            "month": month_key,
            "stationName": station_name,
            "kma_grid_id": kma_grid_id,
            "kma_lon": kma_lon,
            "kma_lat": kma_lat,
            "tm1": tm1,
            "tm2": tm2,
            "error": error,
        })

        print("  STILL FAILED")
        continue

    rows = parse_kma_response(
        text=text,
        station_name=station_name,
        kma_grid_id=kma_grid_id,
        kma_lon=kma_lon,
        kma_lat=kma_lat,
    )

    ok, actual_count, missing_count, extra_count = validate_expected_hours(rows, tm1, tm2)

    if not rows:
        still_failed_count += 1

        append_csv_row(STILL_FAILED_PATH, {
            "month": month_key,
            "stationName": station_name,
            "kma_grid_id": kma_grid_id,
            "kma_lon": kma_lon,
            "kma_lat": kma_lat,
            "tm1": tm1,
            "tm2": tm2,
            "error": "response received but zero parsed rows",
        })

        print("  RESPONSE RECEIVED BUT ZERO PARSED ROWS")
        continue

    recovered_df = pd.DataFrame(rows)
    recovered_df["time"] = pd.to_datetime(recovered_df["time"], errors="coerce")

    month_station_dir = STATION_MONTHLY_DIR / month_key
    month_station_dir.mkdir(exist_ok=True)

    station_file = month_station_dir / (
        f"{safe_filename(station_name)}_{safe_filename(kma_grid_id)}_{month_key}.csv"
    )

    if station_file.exists():
        old_df = pd.read_csv(station_file, encoding="utf-8-sig")
        old_df["time"] = pd.to_datetime(old_df["time"], errors="coerce")

        combined = pd.concat([old_df, recovered_df], ignore_index=True)
    else:
        combined = recovered_df.copy()

    combined = combined.drop_duplicates(
        subset=["stationName", "kma_grid_id", "time"]
    ).sort_values(["stationName", "time"])

    combined.to_csv(station_file, index=False, encoding="utf-8-sig")

    raw_name = safe_filename(
        f"{month_key}_{station_name}_{kma_grid_id}_{tm1}_{tm2}_RECOVERED.txt"
    )
    raw_path = RAW_DIR / raw_name
    raw_path.write_text(text, encoding="utf-8")

    append_csv_row(RECOVERED_LOG_PATH, {
        "month": month_key,
        "stationName": station_name,
        "kma_grid_id": kma_grid_id,
        "kma_lon": kma_lon,
        "kma_lat": kma_lat,
        "tm1": tm1,
        "tm2": tm2,
        "rows_recovered": len(recovered_df),
        "time_check_ok": ok,
        "actual_count": actual_count,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "station_file": str(station_file),
    })

    recovered_count += 1

    print(
        f"  RECOVERED rows={len(recovered_df)} | "
        f"time_check_ok={ok} | saved into {station_file}"
    )

    time.sleep(0.25)


# ============================================================
# 5. Rebuild monthly files
# ============================================================

print("\nRebuilding monthly files...")

month_dirs = sorted([p for p in STATION_MONTHLY_DIR.iterdir() if p.is_dir()])

for month_dir in month_dirs:
    month_key = month_dir.name
    station_files = sorted(month_dir.glob("*.csv"))

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

    monthly_path = MONTHLY_DIR / f"kma_station_weather_{month_key}.csv"
    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    print(f"  rebuilt {monthly_path} rows={len(monthly_df)}")


# ============================================================
# 6. Rebuild final file
# ============================================================

print("\nRebuilding final file...")

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

    print("\nFINAL SAVED:", FINAL_PATH)
    print("rows:", len(final_df))
    print("stations:", final_df["stationName"].nunique())
    print("time range:", final_df["time"].min(), "~", final_df["time"].max())

else:
    print("No monthly files found.")


print("\nRecovery complete.")
print("Recovered requests:", recovered_count)
print("Still failed requests:", still_failed_count)
print("Recovered log:", RECOVERED_LOG_PATH)
print("Still failed log:", STILL_FAILED_PATH)

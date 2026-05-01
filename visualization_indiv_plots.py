import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "kma_station_500m_weather"
FINAL_PATH = OUT_DIR / "station_hourly_kma_weather_all.csv"
FAILED_PATH = OUT_DIR / "failed_requests.csv"

PLOT_DIR = OUT_DIR / "sanity_temperature_plots"
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load collected weather data
# ============================================================

df = pd.read_csv(FINAL_PATH, encoding="utf-8-sig")
df["time"] = pd.to_datetime(df["time"], errors="coerce")

df = df.dropna(subset=["time"])
df = df.sort_values(["stationName", "time"])

print("Data rows:", len(df))
print("Stations:", df["stationName"].nunique())
print("Time range:", df["time"].min(), "~", df["time"].max())


# ============================================================
# 3. Load failed requests
# ============================================================

if FAILED_PATH.exists():
    failed = pd.read_csv(FAILED_PATH, encoding="utf-8-sig")

    failed["tm1_dt"] = pd.to_datetime(
        failed["tm1"].astype(str),
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    failed["tm2_dt"] = pd.to_datetime(
        failed["tm2"].astype(str),
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    failed = failed.dropna(subset=["tm1_dt", "tm2_dt"])

    print("Failed requests:", len(failed))

else:
    failed = pd.DataFrame()
    print("No failed_requests.csv found.")


# ============================================================
# 4. Plot hourly temperature for each station
# ============================================================

stations = sorted(df["stationName"].dropna().unique())

for station in stations:
    station_df = df[df["stationName"] == station].copy()

    if station_df.empty:
        continue

    # Build complete hourly index for this station
    full_time = pd.date_range(
        start=station_df["time"].min(),
        end=station_df["time"].max(),
        freq="h"
    )

    station_hourly = (
        station_df
        .drop_duplicates(subset=["time"])
        .set_index("time")
        .reindex(full_time)
    )

    station_hourly.index.name = "time"

    # Failed request hours for this station
    failed_times = []

    if not failed.empty:
        station_failed = failed[failed["stationName"] == station].copy()

        for _, r in station_failed.iterrows():
            failed_range = pd.date_range(
                start=r["tm1_dt"],
                end=r["tm2_dt"],
                freq="h"
            )
            failed_times.extend(failed_range)

    failed_times = pd.DatetimeIndex(failed_times).drop_duplicates()

    # Keep failed times only within plotted range
    failed_times = failed_times[
        (failed_times >= station_hourly.index.min()) &
        (failed_times <= station_hourly.index.max())
    ]

    # y-position for red failed dots
    temp_min = station_hourly["ta"].min()
    temp_max = station_hourly["ta"].max()

    if pd.isna(temp_min) or pd.isna(temp_max):
        red_y = 0
    else:
        red_y = temp_min - (temp_max - temp_min) * 0.08

    plt.figure(figsize=(16, 5))

    plt.plot(
        station_hourly.index,
        station_hourly["ta"],
        linewidth=1,
        label="Temperature (ta)"
    )

    if len(failed_times) > 0:
        plt.scatter(
            failed_times,
            [red_y] * len(failed_times),
            color="red",
            s=12,
            label="Failed request hour"
        )

    plt.title(f"Hourly Temperature Sanity Check - {station}")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_name = f"temperature_{station}.png".replace("/", "_").replace("\\", "_")
    save_path = PLOT_DIR / save_name

    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved:", save_path)

print("\nDone. Plots saved to:", PLOT_DIR)
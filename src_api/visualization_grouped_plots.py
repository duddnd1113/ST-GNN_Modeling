import math
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

PLOT_DIR = OUT_DIR / "sanity_temperature_plots_grouped"
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Settings
# ============================================================

STATIONS_PER_FIG = 8
N_COLS = 2
N_ROWS = math.ceil(STATIONS_PER_FIG / N_COLS)


# ============================================================
# 3. Load data
# ============================================================

df = pd.read_csv(FINAL_PATH, encoding="utf-8-sig")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df.sort_values(["stationName", "time"])

print("Data rows:", len(df))
print("Stations:", df["stationName"].nunique())
print("Time range:", df["time"].min(), "~", df["time"].max())


# ============================================================
# 4. Load failed requests
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
# 5. Helper function
# ============================================================

def get_station_hourly_and_failed_times(station):
    station_df = df[df["stationName"] == station].copy()

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

    failed_times = failed_times[
        (failed_times >= station_hourly.index.min()) &
        (failed_times <= station_hourly.index.max())
    ]

    return station_hourly, failed_times


# ============================================================
# 6. Make grouped plots
# ============================================================

stations = sorted(df["stationName"].dropna().unique())

num_groups = math.ceil(len(stations) / STATIONS_PER_FIG)

for group_idx in range(num_groups):
    start_idx = group_idx * STATIONS_PER_FIG
    end_idx = start_idx + STATIONS_PER_FIG
    group_stations = stations[start_idx:end_idx]

    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(18, 14),
        sharex=True
    )

    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(group_stations):
            ax.axis("off")
            continue

        station = group_stations[ax_idx]

        station_hourly, failed_times = get_station_hourly_and_failed_times(station)

        temp_min = station_hourly["ta"].min()
        temp_max = station_hourly["ta"].max()

        if pd.isna(temp_min) or pd.isna(temp_max):
            red_y = 0
        else:
            red_y = temp_min - (temp_max - temp_min) * 0.08

        ax.plot(
            station_hourly.index,
            station_hourly["ta"],
            linewidth=0.8,
            label="Temperature"
        )

        if len(failed_times) > 0:
            ax.scatter(
                failed_times,
                [red_y] * len(failed_times),
                color="red",
                s=10,
                label="Failed request"
            )

        ax.set_title(station, fontsize=10)
        ax.set_ylabel("Temp (°C)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")

    fig.suptitle(
        f"Hourly Temperature Sanity Check "
        f"({start_idx + 1}–{min(end_idx, len(stations))} of {len(stations)} stations)",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = PLOT_DIR / f"temperature_group_{group_idx + 1:02d}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved:", save_path)

print("\nDone. Grouped plots saved to:", PLOT_DIR)
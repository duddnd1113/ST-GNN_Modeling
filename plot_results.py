"""
Visualise training results saved under checkpoints/.

Reads loss_history.csv and metrics.json from each completed experiment and
produces three plots saved under plots/:

  1. loss_curves.png   — train/val loss curves, one panel per scenario
  2. test_mae.png      — test MAE bar chart, grouped by graph mode
  3. test_rmse.png     — test RMSE bar chart, grouped by graph mode

Usage:
    python3 plot_results.py              # read everything under checkpoints/
    python3 plot_results.py --graph_mode static climatological
    python3 plot_results.py --scenarios S1_transport_pm10 S3_transport_pm10_pollutants
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CHECKPOINT_DIR = Path("checkpoints")
PLOT_DIR       = Path("plots")

GRAPH_MODES = ("static", "climatological", "soft_dynamic")
MODE_COLORS = {
    "static":         "#4a90d9",
    "climatological": "#e67e22",
    "soft_dynamic":   "#27ae60",
}
MODE_LABELS = {
    "static":         "Static",
    "climatological": "Climatological",
    "soft_dynamic":   "Soft-Dynamic",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def collect_results(filter_windows=None, filter_scenarios=None, filter_modes=None):
    """Walk checkpoints/window_{W}/{scenario}/{mode}/ and collect results."""
    metrics_list = []
    history_map  = {}   # (window, scenario, mode) → DataFrame

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"{CHECKPOINT_DIR} 폴더가 없습니다. 먼저 train.py를 실행하세요.")

    for window_dir in sorted(CHECKPOINT_DIR.iterdir()):
        if not window_dir.is_dir() or not window_dir.name.startswith("window_"):
            continue
        try:
            window = int(window_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if filter_windows and window not in filter_windows:
            continue

        for scenario_dir in sorted(window_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario = scenario_dir.name
            if filter_scenarios and scenario not in filter_scenarios:
                continue

            for mode_dir in sorted(scenario_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode = mode_dir.name
                if filter_modes and mode not in filter_modes:
                    continue

                metrics_path = mode_dir / "metrics.json"
                history_path = mode_dir / "loss_history.csv"

                if metrics_path.exists():
                    with open(metrics_path) as f:
                        m = json.load(f)
                        m["window"] = window
                        metrics_list.append(m)

                if history_path.exists():
                    df = pd.read_csv(history_path)
                    history_map[(window, scenario, mode)] = df

    return metrics_list, history_map


# ── Plot 1: Loss curves ───────────────────────────────────────────────────────

def plot_window_comparison(metrics_list, windows, modes):
    """MAE/RMSE line plot: x=window size, one line per graph mode, averaged over scenarios."""
    df = pd.DataFrame(metrics_list)
    if "window" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ["mae", "rmse"], ["MAE (µg/m³)", "RMSE (µg/m³)"]):
        for mode in modes:
            sub = df[df["graph_mode"] == mode].groupby("window")[metric].mean()
            ax.plot(sub.index, sub.values, marker="o", linewidth=2,
                    color=MODE_COLORS.get(mode, "gray"),
                    label=MODE_LABELS.get(mode, mode))
        ax.set_xlabel("Window size (hours)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(windows)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Window Size Ablation (averaged over scenarios)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = PLOT_DIR / "window_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_loss_curves(history_map, scenarios, modes, title_suffix="", suffix=""):
    n_scenarios = len(scenarios)
    if n_scenarios == 0:
        print("  loss curve: 데이터 없음")
        return

    ncols = min(3, n_scenarios)
    nrows = (n_scenarios + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        has_data = False

        for mode in modes:
            key = (scenario, mode)
            if key not in history_map:
                continue
            df = history_map[key]
            color = MODE_COLORS.get(mode, "gray")
            label = MODE_LABELS.get(mode, mode)
            ax.plot(df["epoch"], df["train_loss"],
                    color=color, linewidth=1.5, linestyle="--", alpha=0.6)
            ax.plot(df["epoch"], df["val_loss"],
                    color=color, linewidth=2.0, label=label)
            has_data = True

        short_name = scenario.replace("transport_", "").replace("_", "\n")
        ax.set_title(short_name, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Masked MSE", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if has_data:
            ax.legend(fontsize=7)

    # 빈 패널 숨기기
    for idx in range(n_scenarios, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # 공통 범례 (실선=val, 점선=train)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="-",  color="gray", linewidth=2, label="Val loss"),
        Line2D([0], [0], linestyle="--", color="gray", linewidth=1.5, alpha=0.6, label="Train loss"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=9, framealpha=0.9)

    fig.suptitle(f"Train / Val Loss Curves{title_suffix}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = PLOT_DIR / f"loss_curves{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Plot 2 & 3: MAE / RMSE bar charts ────────────────────────────────────────

def plot_metric_bar(metrics_list, metric_key, ylabel, title, out_filename):
    if not metrics_list:
        print(f"  {out_filename}: 데이터 없음")
        return

    df = pd.DataFrame(metrics_list)
    if metric_key not in df.columns:
        return

    scenarios = sorted(df["scenario"].unique())
    modes_present = [m for m in GRAPH_MODES if m in df["graph_mode"].unique()]

    x = np.arange(len(scenarios))
    n_modes = len(modes_present)
    bar_w = 0.7 / n_modes
    offsets = np.linspace(-(n_modes - 1) / 2, (n_modes - 1) / 2, n_modes) * bar_w

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 1.2), 5))

    for i, mode in enumerate(modes_present):
        sub = df[df["graph_mode"] == mode].set_index("scenario")
        vals = [sub.loc[s, metric_key] if s in sub.index else np.nan
                for s in scenarios]
        bars = ax.bar(
            x + offsets[i], vals,
            width=bar_w * 0.9,
            color=MODE_COLORS.get(mode, "gray"),
            label=MODE_LABELS.get(mode, mode),
            alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7)

    short_labels = [s.replace("transport_", "").replace("_", "\n")
                    for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="y", which="minor", alpha=0.1)

    fig.tight_layout()
    out_path = PLOT_DIR / out_filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(filter_windows=None, filter_scenarios=None, filter_modes=None):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("결과 수집 중...")
    metrics_list, history_map = collect_results(filter_windows, filter_scenarios, filter_modes)

    n_metrics = len(metrics_list)
    n_history = len(history_map)
    print(f"  metrics.json: {n_metrics}개  /  loss_history.csv: {n_history}개")

    if n_metrics == 0 and n_history == 0:
        print("완료된 실험이 없습니다. train.py를 먼저 실행하세요.")
        return

    modes     = filter_modes or list(GRAPH_MODES)
    windows   = filter_windows or sorted({k[0] for k in history_map} |
                                         {m.get("window") for m in metrics_list if "window" in m})
    scenarios = filter_scenarios or sorted({k[1] for k in history_map} |
                                           {m["scenario"] for m in metrics_list})

    print("플롯 생성 중...")
    # window별로 loss curve + bar chart 분리 생성
    for w in windows:
        h_map_w = {(s, mo): df for (win, s, mo), df in history_map.items() if win == w}
        m_list_w = [m for m in metrics_list if m.get("window") == w]
        suffix = f"_w{w}"

        plot_loss_curves(h_map_w, scenarios, modes, title_suffix=f" (window={w}h)", suffix=suffix)
        plot_metric_bar(m_list_w, "mae",  "MAE (µg/m³)",
                        f"Test MAE — window={w}h", f"test_mae{suffix}.png")
        plot_metric_bar(m_list_w, "rmse", "RMSE (µg/m³)",
                        f"Test RMSE — window={w}h", f"test_rmse{suffix}.png")

    # window 비교 플롯 (metric 있을 때만)
    if len(windows) > 1 and metrics_list:
        plot_window_comparison(metrics_list, windows, modes)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows",     nargs="+", type=int, default=None)
    parser.add_argument("--scenarios",   nargs="+", default=None)
    parser.add_argument("--graph_modes", nargs="+", default=None, choices=GRAPH_MODES)
    args = parser.parse_args()
    main(args.windows, args.scenarios, args.graph_modes)

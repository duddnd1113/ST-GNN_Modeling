"""
Appendix C4: ST-GNN Experimental Results Visualization

Generates 5 figures from all 78 checkpoint metrics.json files:
  fig1_heatmap_mae.png        — Scenario × Graph Mode MAE heatmap (best window)
  fig2_graph_mode_bar.png     — Graph Mode 성능 비교 (window별)
  fig3_window_effect.png      — Window size가 MAE에 미치는 영향
  fig4_best_per_scenario.png  — 시나리오별 Best 구성 MAE 랭킹
  fig5_scenario_group.png     — Feature Group 추가 효과 분석
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

# ── Korean font ───────────────────────────────────────────────────────────────
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT_DIR   = "Appendix/Appendix_C4"
CKPT_GLOB = "checkpoints/**/metrics.json"

SCENARIO_LABELS = {
    "S1_transport_pm10":                      "S1\n(Base)",
    "S2_transport_pm10_pm10mask":             "S2\n(+PM10 Mask)",
    "S3_transport_pm10_pollutants":           "S3\n(+Pollutants)",
    "S4_transport_pm10_pollutants_pm10mask":  "S4\n(+Poll+PM10M)",
    "S5_transport_pm10_pollutants_allmask":   "S5\n(+Poll+AllM)",
    "S6_transport_pm10_pollutants_summarymask":"S6\n(+Poll+SumM)",
    "S7_transport_pm10_weather":              "S7\n(+Weather)",
    "S8_transport_pm10_rain":                 "S8\n(+Rain)",
    "S9_transport_pm10_weather_rain":         "S9\n(+Weath+Rain)",
    "S10_transport_all_summarymask":          "S10\n(All+SumM)",
}

SCENARIO_SHORT = {k: f"S{i+1}" for i, k in enumerate(SCENARIO_LABELS)}

MODE_COLORS = {
    "static":        "#2B6CB0",
    "soft_dynamic":  "#C0185A",
    "climatological":"#2D9E4F",
}
MODE_LABELS = {
    "static":        "Static",
    "soft_dynamic":  "Soft-Dynamic",
    "climatological":"Climatological",
}
WINDOW_MARKERS = {12: "o", 24: "s", 48: "^"}


def load_results(glob_pattern: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(glob_pattern, recursive=True):
        with open(path) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    df["scenario_short"] = df["scenario"].map(SCENARIO_SHORT)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    return df.sort_values(["window", "scenario", "graph_mode"]).reset_index(drop=True)


# ── Fig 1: Heatmap (Scenario × Graph Mode, best window per cell) ──────────────

def fig1_heatmap(df: pd.DataFrame):
    modes    = ["static", "soft_dynamic", "climatological"]
    scenarios = list(SCENARIO_LABELS.keys())

    # Best MAE across windows
    best = df.groupby(["scenario", "graph_mode"])["mae"].min().reset_index()
    pivot = best.pivot(index="scenario", columns="graph_mode", values="mae")
    pivot = pivot.reindex(index=scenarios, columns=modes)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#2ecc71", "#f1c40f", "#e74c3c"], N=256
    )
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=2.6, vmax=3.7)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([SCENARIO_LABELS[s].replace("\n", " ") for s in scenarios],
                        fontsize=9)

    for i, s in enumerate(scenarios):
        for j, m in enumerate(modes):
            val = pivot.loc[s, m]
            if pd.notna(val):
                color = "white" if val > 3.2 else "#1a1a2e"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#aaaaaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Test MAE (µg/m³)", fontsize=9)

    ax.set_title("Fig 1. Scenario × Graph Mode Best MAE Heatmap\n"
                 "(각 셀: window 12/24/48 중 최소 MAE)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Graph Mode", fontsize=10)
    ax.set_ylabel("Feature Scenario", fontsize=10)

    plt.tight_layout()
    out = f"{OUT_DIR}/fig1_heatmap_mae.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close()


# ── Fig 2: Graph Mode bar chart (window별 평균 MAE) ───────────────────────────

def fig2_graph_mode_bar(df: pd.DataFrame):
    windows = [12, 24, 48]
    modes   = ["static", "soft_dynamic", "climatological"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Fig 2. Graph Mode별 평균 MAE (Window Size 별)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, w in zip(axes, windows):
        sub = df[df["window"] == w]
        means = [sub[sub["graph_mode"] == m]["mae"].mean() for m in modes]
        stds  = [sub[sub["graph_mode"] == m]["mae"].std()  for m in modes]
        colors = [MODE_COLORS[m] for m in modes]
        labels = [MODE_LABELS[m] for m in modes]

        bars = ax.bar(range(len(modes)), means, yerr=stds,
                      color=colors, alpha=0.85, width=0.55,
                      capsize=5, error_kw=dict(lw=1.5, capthick=1.5))

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#1a1a2e")

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"Window = {w}h", fontsize=10, fontweight="bold")
        ax.set_ylabel("MAE (µg/m³)" if ax == axes[0] else "", fontsize=9)
        ax.set_ylim(2.4, 3.8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/fig2_graph_mode_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close()


# ── Fig 3: Window size 효과 (line plot) ───────────────────────────────────────

def fig3_window_effect(df: pd.DataFrame):
    scenarios = list(SCENARIO_LABELS.keys())
    modes     = ["static", "soft_dynamic"]  # climatological 일부 누락으로 제외

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Fig 3. Window Size가 MAE에 미치는 영향\n"
                 "(시나리오별 선, 각 그래프 모드)",
                 fontsize=12, fontweight="bold", y=1.02)

    cmap_lines = plt.cm.get_cmap("tab10", len(scenarios))

    for ax, mode in zip(axes, modes):
        for i, s in enumerate(scenarios):
            sub = df[(df["scenario"] == s) & (df["graph_mode"] == mode)]
            sub = sub.sort_values("window")
            if len(sub) < 2:
                continue
            ax.plot(sub["window"], sub["mae"],
                    marker="o", markersize=5, linewidth=1.5,
                    color=cmap_lines(i), alpha=0.8,
                    label=SCENARIO_SHORT[s])

        ax.set_title(f"Graph Mode: {MODE_LABELS[mode]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Window Size (hours)", fontsize=9)
        ax.set_ylabel("Test MAE (µg/m³)" if ax == axes[0] else "", fontsize=9)
        ax.set_xticks([12, 24, 48])
        ax.set_ylim(2.4, 3.7)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Scenario", fontsize=8, title_fontsize=9,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.14),
               framealpha=0.9, edgecolor="#CBD5E0")

    plt.tight_layout()
    out = f"{OUT_DIR}/fig3_window_effect.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close()


# ── Fig 4: 시나리오별 Best MAE 랭킹 ─────────────────────────────────────────

def fig4_best_per_scenario(df: pd.DataFrame):
    best = (df.sort_values("mae")
              .groupby("scenario")
              .first()
              .reset_index()
              [["scenario", "mae", "rmse", "graph_mode", "window"]])
    best["label"] = best["scenario"].map(SCENARIO_LABELS).str.replace("\n", " ")
    best = best.sort_values("mae")

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [MODE_COLORS[m] for m in best["graph_mode"]]
    bars = ax.barh(range(len(best)), best["mae"], color=colors, alpha=0.85, height=0.6)

    for i, (_, row) in enumerate(best.iterrows()):
        ax.text(row["mae"] + 0.01, i,
                f'{row["mae"]:.3f}  [{MODE_LABELS[row["graph_mode"]]}, W={row["window"]}h]',
                va="center", fontsize=8.5, color="#2D3748")

    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best["label"], fontsize=9)
    ax.set_xlabel("Best Test MAE (µg/m³)", fontsize=10)
    ax.set_xlim(2.5, 3.3)
    ax.set_title("Fig 4. 시나리오별 Best Configuration MAE 랭킹\n"
                 "(최적 Graph Mode × Window 조합 기준)", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85, label=MODE_LABELS[m])
        for m, c in MODE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, title="Graph Mode", fontsize=8.5,
              title_fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = f"{OUT_DIR}/fig4_best_per_scenario.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close()


# ── Fig 5: Feature Group 추가 효과 (Static, Window 12 기준) ──────────────────

def fig5_feature_group_effect(df: pd.DataFrame):
    # Static + window=12 기준으로 feature group 추가 효과 비교
    sub = df[(df["graph_mode"] == "static") & (df["window"] == 12)].copy()
    sub["short"] = sub["scenario"].map(SCENARIO_SHORT)

    scenario_order = list(SCENARIO_LABELS.keys())
    sub = sub.set_index("scenario").reindex(scenario_order).reset_index()
    sub["short"] = sub["scenario"].map(SCENARIO_SHORT)

    # Baseline (S1) MAE
    baseline = sub[sub["scenario"] == "S1_transport_pm10"]["mae"].values[0]

    colors = []
    for _, row in sub.iterrows():
        delta = row["mae"] - baseline
        colors.append("#2ecc71" if delta <= 0 else "#e74c3c")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(len(sub)), sub["mae"], color=colors, alpha=0.85,
                  width=0.6, zorder=3)
    ax.axhline(baseline, color="#2B6CB0", linewidth=1.5, linestyle="--",
               label=f"S1 Baseline (MAE={baseline:.3f})", zorder=4)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        delta = row["mae"] - baseline
        sign = "+" if delta > 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, row["mae"] + 0.02,
                f'{row["mae"]:.3f}\n({sign}{delta:.3f})',
                ha="center", va="bottom", fontsize=7.5, color="#1a1a2e",
                fontweight="bold")

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [f"{row['short']}\n{SCENARIO_LABELS[row['scenario']].split(chr(10))[1]}"
         for _, row in sub.iterrows()],
        fontsize=8.5
    )
    ax.set_ylabel("Test MAE (µg/m³)", fontsize=10)
    ax.set_ylim(2.4, 3.2)
    ax.set_title("Fig 5. Feature Group 추가에 따른 MAE 변화\n"
                 "(Static Graph Mode, Window=12h 기준; 초록=개선, 빨강=악화)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/fig5_scenario_group.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    df = load_results(CKPT_GLOB)
    print(f"  Total runs: {len(df)}")

    print("Generating figures...")
    fig1_heatmap(df)
    fig2_graph_mode_bar(df)
    fig3_window_effect(df)
    fig4_best_per_scenario(df)
    fig5_feature_group_effect(df)

    # Save aggregated results CSV
    csv_path = f"{OUT_DIR}/all_results.csv"
    df.sort_values(["window", "scenario", "graph_mode"]).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()

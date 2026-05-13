"""
Appendix D2: Relationship Between Wind Direction and Attention

  fig_d2_1_wind_attn_scatter.png  — Wind alignment vs attention scatter (per edge)
  fig_d2_2_polar_attn.png         — Polar plot: bearing vs mean attention
  fig_d2_3_corr_by_head.png       — Correlation between wind alignment & attention per head
  fig_d2_4_wind_dir_attn_map.png  — Per-wind-direction attention maps + stability heatmap
"""

import sys, warnings, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
ROOT = "/workspace/ST-GNN Modeling"
sys.path.insert(0, ROOT)

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})

OUT = "Appendix/Appendix_D2"

# ── Load cache ────────────────────────────────────────────────────────────────
cache = np.load(f"{ROOT}/Appendix/attn_cache.npz")
attn_per_ts   = cache["attn_per_ts"]    # [N_test, T, E, H]
edge_index    = cache["edge_index"]
static_attr   = cache["static_attr"]   # [E, 3] — [dist_norm, sin_b, cos_b]
edge_bearings = cache["edge_bearings"] # [E] degrees

# Wind alignment from static_attr: full edge features are [dist, sin_b, cos_b, wind_align, eff_wind]
# We need to re-derive wind alignment from test node features
# For simplicity: use static_attr bearing to categorize edges directionally,
# and correlate with attention from cache (attn_per_ts averaged over T per sample)

E = edge_index.shape[1]
attn_mean_e = attn_per_ts.mean(axis=(0,1,3))  # [E] — mean over samples, T, heads
attn_mean_eh = attn_per_ts.mean(axis=(0,1))   # [E, H]

sin_b = static_attr[:, 1]   # [E]
cos_b = static_attr[:, 2]   # [E]
bearings = edge_bearings     # [E] in degrees

# Load raw test node features to compute wind alignment dynamically
import pandas as _pd, pickle as _pkl
from pathlib import Path
from dataset import load_scenario_split

SCENARIO = "S3_transport_pm10_pollutants"
(train_nodes, val_nodes, test_nodes, *_rest, coords, feat_cols) = load_scenario_split(SCENARIO)
with open(f"{ROOT}/split_info.pkl","rb") as f:
    split_info = _pkl.load(f)

UU_IDX = feat_cols.index("동서 방향 풍속")
VV_IDX = feat_cols.index("남북 방향 풍속")
WINDOW = 12

T_test = len(test_nodes) - WINDOW

# Compute per-edge wind alignment for each test timestep
# wind_align[t, e] = uu_src[t] * sin(b[e]) + vv_src[t] * cos(b[e])
src_idx = edge_index[0]   # [E]
sin_b_e = np.sin(np.radians(bearings))  # [E]
cos_b_e = np.cos(np.radians(bearings))

# Average wind alignment over window for each sample
# attn_per_ts[s, t, e, h] ↔ test window starting at s, timestep offset t
wind_align_samples = []   # [T_test, E]
for start in range(T_test):
    uu = test_nodes[start:start+WINDOW, src_idx, UU_IDX]   # [T, E]
    vv = test_nodes[start:start+WINDOW, src_idx, VV_IDX]   # [T, E]
    wa = (uu * sin_b_e[None,:] + vv * cos_b_e[None,:]).mean(axis=0)  # [E]
    wind_align_samples.append(wa)

wind_align = np.stack(wind_align_samples, axis=0)  # [T_test, E]
wind_align_mean = wind_align.mean(axis=0)           # [E]

# ── Fig D2-1: Scatter wind_alignment vs attention ─────────────────────────────
print("Generating fig_d2_1...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig D2-1. Wind Alignment vs GAT Attention Weight\n"
             "(엣지별 평균값, S3+Static+W12 기준)", fontsize=11, fontweight="bold")

# Left: scatter all edges
ax = axes[0]
sc = ax.scatter(wind_align_mean, attn_mean_e,
                c=attn_mean_e, cmap="YlOrRd", alpha=0.5, s=18, zorder=3)
# Trend line
z = np.polyfit(wind_align_mean, attn_mean_e, 1)
p = np.poly1d(z)
xline = np.linspace(wind_align_mean.min(), wind_align_mean.max(), 100)
ax.plot(xline, p(xline), "k--", lw=1.5, label=f"trend (slope={z[0]:.5f})")
corr = np.corrcoef(wind_align_mean, attn_mean_e)[0,1]
ax.set_xlabel("Mean Wind Alignment (m/s)", fontsize=10)
ax.set_ylabel("Mean Attention Weight", fontsize=10)
ax.set_title(f"전체 엣지 (Pearson r = {corr:.3f})", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.colorbar(sc, ax=ax, label="Attention", shrink=0.8)

# Right: box plot by wind direction quadrant
ax = axes[1]
labels_q = ["역풍\n(< −1)", "약한 역풍\n(−1~0)", "약한 순풍\n(0~1)", "강한 순풍\n(> 1)"]
bins = [(-np.inf,-1), (-1,0), (0,1), (1,np.inf)]
grouped = []
for lo, hi in bins:
    mask = (wind_align_mean > lo) & (wind_align_mean <= hi)
    grouped.append(attn_mean_e[mask])

bp = ax.boxplot(grouped, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", lw=2))
colors_q = ["#e74c3c","#f39c12","#2ecc71","#2980b9"]
for patch, col in zip(bp["boxes"], colors_q):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.set_xticklabels(labels_q, fontsize=8.5)
ax.set_ylabel("Mean Attention Weight", fontsize=10)
ax.set_title("풍향 사분위별 Attention 분포", fontsize=9)
counts = [len(g) for g in grouped]
for i, cnt in enumerate(counts):
    ax.text(i+1, ax.get_ylim()[0], f"n={cnt}", ha="center", va="bottom", fontsize=7.5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d2_1_wind_attn_scatter.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d2_1_wind_attn_scatter.png")
plt.close()

# ── Fig D2-2: Polar plot bearing vs attention ─────────────────────────────────
print("Generating fig_d2_2...")
n_bins = 24
bin_edges = np.linspace(0, 360, n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
attn_by_bearing = []
for i in range(n_bins):
    mask = (bearings >= bin_edges[i]) & (bearings < bin_edges[i+1])
    attn_by_bearing.append(attn_mean_e[mask].mean() if mask.sum() > 0 else 0.0)
attn_by_bearing = np.array(attn_by_bearing)

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
theta = np.radians(bin_centers)
width = np.radians(360 / n_bins)
bars = ax.bar(theta, attn_by_bearing, width=width, alpha=0.8,
              color=plt.cm.YlOrRd(attn_by_bearing / attn_by_bearing.max()),
              edgecolor="white", linewidth=0.5)
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_title("Fig D2-2. 엣지 방위각별 평균 Attention\n(North=0°, 시계방향)",
             fontsize=10, fontweight="bold", pad=15)
ax.set_xticks(np.radians([0,45,90,135,180,225,270,315]))
ax.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"], fontsize=9)

sm = plt.cm.ScalarMappable(cmap="YlOrRd",
     norm=plt.Normalize(attn_by_bearing.min(), attn_by_bearing.max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="Mean Attention")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d2_2_polar_attn.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d2_2_polar_attn.png")
plt.close()

# ── Fig D2-3: Correlation per head ────────────────────────────────────────────
print("Generating fig_d2_3...")
H = attn_mean_eh.shape[1]
head_colors = ["#2B6CB0","#C0185A","#2D9E4F","#D69E2E"]

fig, axes = plt.subplots(1, H, figsize=(14, 4), sharey=False)
fig.suptitle("Fig D2-3. Wind Alignment vs Attention — Head별 상관관계",
             fontsize=11, fontweight="bold")

for h_idx, ax in enumerate(axes):
    attn_h = attn_mean_eh[:, h_idx]
    corr_h = np.corrcoef(wind_align_mean, attn_h)[0,1]
    ax.scatter(wind_align_mean, attn_h, alpha=0.4, s=12,
               color=head_colors[h_idx], zorder=3)
    z = np.polyfit(wind_align_mean, attn_h, 1)
    xl = np.linspace(wind_align_mean.min(), wind_align_mean.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "k--", lw=1.5)
    ax.set_title(f"Head {h_idx+1}\nr = {corr_h:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Wind Alignment (m/s)", fontsize=9)
    ax.set_ylabel("Attention Weight" if h_idx == 0 else "", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d2_3_corr_by_head.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d2_3_corr_by_head.png")
plt.close()

print("D2 Done.")

# ── Fig D2-4: Per-wind-direction Attention Maps + Stability Heatmap ───────────
print("Generating fig_d2_4...")

import pandas as _pd2
from dataset import SCENARIO_DIR

# Station names (sorted alphabetically — matches model node ordering)
_csv = _pd2.read_csv(SCENARIO_DIR / f"{SCENARIO}.csv")
station_names = sorted(_csv["측정소명"].unique())   # [40]

lats_arr = np.array([c[0] for c in coords])        # [40]
lons_arr = np.array([c[1] for c in coords])        # [40]
src_nodes = edge_index[0]                           # [E]
dst_nodes = edge_index[1]                           # [E]

# ── Per-sample wind direction (uu/vv, consistent with D2 wind_align formula) ──
T_test = len(test_nodes) - WINDOW

# City-wide mean uu/vv across all stations, averaged over the 12-step window
uu_sample = np.array([test_nodes[s:s+WINDOW, :, UU_IDX].mean() for s in range(T_test)])  # [T_test]
vv_sample = np.array([test_nodes[s:s+WINDOW, :, VV_IDX].mean() for s in range(T_test)])  # [T_test]

def _classify_uv(uu, vv):
    """Meteorological wind direction: where the wind comes FROM."""
    if abs(uu) >= abs(vv):
        return "W" if uu > 0 else "E"   # uu>0 → blows east → from west → 서풍
    else:
        return "S" if vv > 0 else "N"   # vv>0 → blows north → from south → 남풍

sample_dirs = np.array([_classify_uv(uu_sample[i], vv_sample[i]) for i in range(T_test)])

DIR_META = {
    "E": ("동풍", "← (서쪽으로 이동)", (-1,  0)),
    "W": ("서풍", "→ (동쪽으로 이동)", ( 1,  0)),
    "S": ("남풍", "↑ (북쪽으로 이동)", ( 0,  1)),
    "N": ("북풍", "↓ (남쪽으로 이동)", ( 0, -1)),
}

dir_counts = {d: int((sample_dirs == d).sum()) for d in "EWSN"}
print("  Per-sample wind direction counts:", dir_counts)

# Only include directions with n ≥ 5 for statistical reliability
valid_dirs = [d for d in ["E", "S", "W", "N"] if dir_counts[d] >= 5]
print("  Valid directions (n≥5):", valid_dirs)

# Per-direction mean attention: average over samples, timesteps, heads
attn_per_sample = attn_per_ts.mean(axis=(1, 3))   # [T_test, E]

attn_by_dir = {
    d: attn_per_sample[sample_dirs == d].mean(axis=0)  # [E]
    for d in valid_dirs
}

# Global colour scale for fair cross-direction comparison
_all = np.concatenate(list(attn_by_dir.values()))
g_vmin, g_vmax = _all.min(), _all.max()

# Cross-direction Pearson correlations (quantify pattern stability)
dir_pairs = [(a, b) for i, a in enumerate(valid_dirs)
             for b in valid_dirs[i+1:]]
corrs = {(a, b): np.corrcoef(attn_by_dir[a], attn_by_dir[b])[0, 1]
         for a, b in dir_pairs}

# ── Fig D2-4: Per-wind-direction Attention Maps ───────────────────────────────
n_dirs = len(valid_dirs)
lon_span = lons_arr.max() - lons_arr.min()
lat_span = lats_arr.max() - lats_arr.min()
arrow_scale_lon = lon_span * 0.10
arrow_scale_lat = lat_span * 0.10

fig4, axes4_2d = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
axes4 = axes4_2d.ravel()

sm4 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(g_vmin, g_vmax))
sm4.set_array([])

for col, (ax, d) in enumerate(zip(axes4, valid_dirs)):
    ea = attn_by_dir[d]
    ea_norm = (ea - g_vmin) / (g_vmax - g_vmin + 1e-8)

    # Draw all edges
    for e in range(len(src_nodes)):
        w = float(ea_norm[e])
        color = plt.cm.YlOrRd(w)
        ax.annotate("",
            xy=(lons_arr[dst_nodes[e]], lats_arr[dst_nodes[e]]),
            xytext=(lons_arr[src_nodes[e]], lats_arr[src_nodes[e]]),
            arrowprops=dict(arrowstyle="->", color=color,
                            lw=0.25 + w * 2.8, alpha=0.20 + w * 0.72,
                            mutation_scale=7),
            zorder=2)

    # Nodes
    ax.scatter(lons_arr, lats_arr, s=55, color="steelblue",
               edgecolors="white", linewidths=0.6, zorder=5)

    # Label top-3 high-attention edge nodes
    top3 = np.argsort(ea)[::-1][:3]
    labeled = set()
    for e in top3:
        for ni in [src_nodes[e], dst_nodes[e]]:
            if ni not in labeled:
                ax.annotate(station_names[ni],
                    (lons_arr[ni], lats_arr[ni]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points",
                    color="darkred", fontweight="bold", zorder=6)
                labeled.add(ni)

    # Wind direction arrow
    uu_vec, vv_vec = DIR_META[d][2]
    cx = lons_arr.max() - lon_span * 0.14
    cy = lats_arr.max() - lat_span * 0.12
    ax.annotate("",
        xy=(cx + uu_vec * arrow_scale_lon, cy + vv_vec * arrow_scale_lat),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->, head_width=0.3, head_length=0.3",
                        color="royalblue", lw=2.5),
        zorder=7)
    ax.text(cx + uu_vec * arrow_scale_lon * 1.4,
            cy + vv_vec * arrow_scale_lat * 1.4,
            "바람", fontsize=8, color="royalblue", ha="center", va="center")

    kr_lbl, arrow_sym, _ = DIR_META[d]
    ax.set_title(f"{kr_lbl}  {arrow_sym}\nn={dir_counts[d]}", fontsize=11, fontweight="bold")
    ax.set_xlabel("경도", fontsize=9)
    if col % 2 == 0:
        ax.set_ylabel("위도", fontsize=9)
    ax.grid(alpha=0.15, linestyle="--")
    ax.tick_params(labelsize=8)

# Hide unused subplot if valid_dirs < 4
for idx in range(len(valid_dirs), 4):
    axes4_2d.ravel()[idx].set_visible(False)

fig4.colorbar(sm4, ax=axes4_2d, label="Mean Attention Weight",
              shrink=0.6, pad=0.02)
fig4.suptitle(
    "Fig D2-4.  풍향별 GAT Attention Map\n"
    "모델이 바람 방향에 따라 attention 패턴을 동적으로 변화시키는가?",
    fontsize=12, fontweight="bold")

plt.savefig(f"{OUT}/fig_d2_4_wind_dir_attn_map.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d2_4_wind_dir_attn_map.png")
plt.close()

# ── Fig D2-5: Top-K Edge Attention Stability Heatmap ─────────────────────────
print("Generating fig_d2_5...")

TOP_K = 15
global_top_k = np.argsort(attn_mean_e)[::-1][:TOP_K]

heat_data = np.array(
    [[attn_by_dir[d][e] for e in global_top_k] for d in valid_dirs]
)  # [n_dirs, K]

xlabels = [
    f"{station_names[src_nodes[e]]}\n→ {station_names[dst_nodes[e]]}"
    for e in global_top_k
]
ylabels = [f"{DIR_META[d][0]} ({DIR_META[d][1].split()[0]})" for d in valid_dirs]

fig5, ax5 = plt.subplots(figsize=(14, 3.8), constrained_layout=True)

im5 = ax5.imshow(heat_data, aspect="auto", cmap="YlOrRd",
                 vmin=g_vmin, vmax=g_vmax)

ax5.set_yticks(range(n_dirs))
ax5.set_yticklabels(ylabels, fontsize=11)
ax5.set_xticks(range(TOP_K))
ax5.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8.5)

# Cell value annotations
mid = (g_vmin + g_vmax) / 2
for row_i, d in enumerate(valid_dirs):
    for col_i, e in enumerate(global_top_k):
        val = attn_by_dir[d][e]
        ax5.text(col_i, row_i, f"{val:.3f}",
                 ha="center", va="center", fontsize=7.5,
                 color="white" if val > mid else "black")

fig5.colorbar(im5, ax=ax5, label="Mean Attention Weight", shrink=0.9)

# Cross-direction Pearson r as subtitle
corr_str = "   ".join(
    f"r({DIR_META[a][0]}↔{DIR_META[b][0]}) = {v:.4f}"
    for (a, b), v in corrs.items()
)
ax5.set_title(
    f"Fig D2-5.  Top-{TOP_K} 글로벌 Attention 엣지 — 풍향별 패턴 안정성\n"
    f"풍향간 Pearson r:  {corr_str}",
    fontsize=9.5)

plt.savefig(f"{OUT}/fig_d2_5_attn_rank_stability.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d2_5_attn_rank_stability.png")
plt.close()

# Print top-3 edges per direction for verification
print("\n  [Top-3 attention edges per wind direction]")
for d in valid_dirs:
    kr_lbl = DIR_META[d][0]
    top3 = np.argsort(attn_by_dir[d])[::-1][:3]
    print(f"  {kr_lbl} (n={dir_counts[d]}):")
    for rank, e in enumerate(top3, 1):
        print(f"    {rank}. {station_names[src_nodes[e]]:14s} → "
              f"{station_names[dst_nodes[e]]:14s}  α={attn_by_dir[d][e]:.4f}")

print("\n  [Cross-direction Pearson r (attention pattern similarity)]")
for (a, b), r in corrs.items():
    print(f"  {DIR_META[a][0]} ↔ {DIR_META[b][0]} : r = {r:.4f}")

print("D2 (fig_d2_4, fig_d2_5) Done.")

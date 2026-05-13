"""
Appendix D2: Relationship Between Wind Direction and Attention

  fig_d2_1_wind_attn_scatter.png  — Wind alignment vs attention scatter (per edge)
  fig_d2_2_polar_attn.png         — Polar plot: bearing vs mean attention
  fig_d2_3_corr_by_head.png       — Correlation between wind alignment & attention per head
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

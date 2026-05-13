"""
Appendix D1: Spatial Attention Visualization

Loads pre-cached attention weights and generates:
  fig_d1_1_attn_map_osm.png      — Edge attention on Seoul OSM map
  fig_d1_2_node_attention.png    — Per-node mean received attention (bar + map)
  fig_d1_3_head_comparison.png   — Attention pattern per GAT head
"""

import sys, io, time, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import requests
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = "/workspace/ST-GNN Modeling"
sys.path.insert(0, ROOT)

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})

OUT  = "Appendix/Appendix_D1"
ZOOM = 12

# ── Load cache ────────────────────────────────────────────────────────────────
cache = np.load(f"{ROOT}/Appendix/attn_cache.npz")
attn_mean    = cache["attn_mean"]       # [E]
attn_per_ts  = cache["attn_per_ts"]    # [N_test, T, E, H]
edge_index   = cache["edge_index"]     # [2, E]
static_attr  = cache["static_attr"]   # [E, 3]
edge_bearings = cache["edge_bearings"] # [E]

# Station coords
import pandas as _pd
_csv = _pd.read_csv("/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios/S3_transport_pm10_pollutants.csv")
_stations = sorted(_csv["측정소명"].unique())
_cdf = _csv.drop_duplicates("측정소명").set_index("측정소명")[["위도","경도"]]
coords   = [(float(_cdf.loc[s,"위도"]), float(_cdf.loc[s,"경도"])) for s in _stations]
stations = _stations
N = len(stations)

# ── OSM tile helpers (same as Appendix_C2) ───────────────────────────────────

def deg_to_tile(lat, lon, z):
    n = 2**z
    x = int((lon+180)/360*n)
    y = int((1 - math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
    return x, y

def tile_to_deg(x, y, z):
    n = 2**z
    lon = x/n*360-180
    lat = math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n))))
    return lat, lon

def fetch_map(lat_min, lat_max, lon_min, lon_max, zoom=12, pad=0.06):
    dlat=(lat_max-lat_min)*pad; dlon=(lon_max-lon_min)*pad
    lat_min-=dlat; lat_max+=dlat; lon_min-=dlon; lon_max+=dlon
    x0,y0=deg_to_tile(lat_max,lon_min,zoom); x1,y1=deg_to_tile(lat_min,lon_max,zoom)
    x0,x1=min(x0,x1),max(x0,x1); y0,y1=min(y0,y1),max(y0,y1)
    tw,th=x1-x0+1, y1-y0+1
    canvas=Image.new("RGB",(tw*256,th*256))
    sess=requests.Session()
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try:
                r=sess.get(f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png",
                           headers={"User-Agent":"ST-GNN-Research/1.0"},timeout=10)
                img=Image.open(io.BytesIO(r.content)).convert("RGB")
            except:
                img=Image.new("RGB",(256,256),(220,220,220))
            canvas.paste(img,((tx-x0)*256,(ty-y0)*256))
            time.sleep(0.04)
    lat_top,lon_left=tile_to_deg(x0,y0,zoom)
    lat_bot,lon_right=tile_to_deg(x1+1,y1+1,zoom)
    return np.array(canvas),(lat_top,lat_bot,lon_left,lon_right)

def ll2px(lat, lon, bounds, img_shape):
    lat_top,lat_bot,lon_left,lon_right = bounds
    h,w = img_shape[:2]
    def my(ld): return math.log(math.tan(math.pi/4+math.radians(ld)/2))
    fx=(lon-lon_left)/(lon_right-lon_left)
    fy=(my(lat_top)-my(lat)  )/(my(lat_top)-my(lat_bot))
    return fx*w, fy*h

# ── Pre-compute node-level attention (received) ───────────────────────────────
# For each node i: mean attention from all edges j→i
node_attn = np.zeros(N)
count = np.zeros(N)
for e in range(edge_index.shape[1]):
    dst = edge_index[1, e]
    node_attn[dst] += attn_mean[e]
    count[dst] += 1
node_attn_mean = node_attn / (count + 1e-8)   # mean received attention per node

# ── Fetch map ─────────────────────────────────────────────────────────────────
lats = [c[0] for c in coords]; lons = [c[1] for c in coords]
print("Fetching OSM tiles...")
img, bounds = fetch_map(min(lats), max(lats), min(lons), max(lons), zoom=ZOOM)
px_list = [ll2px(lat, lon, bounds, img.shape) for lat, lon in coords]
h_img, w_img = img.shape[:2]

# ── Fig D1-1: Attention map on OSM ───────────────────────────────────────────
print("Generating fig_d1_1...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=[0, w_img, h_img, 0], zorder=0)
ax.set_xlim(0, w_img); ax.set_ylim(h_img, 0); ax.axis("off")

cmap_e = plt.cm.YlOrRd
attn_norm = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

E = edge_index.shape[1]
for e in range(E):
    si, di = edge_index[0, e], edge_index[1, e]
    ps, pd_ = px_list[si], px_list[di]
    c = cmap_e(attn_norm[e])
    lw = 0.5 + attn_norm[e] * 2.5
    ax.plot([ps[0], pd_[0]], [ps[1], pd_[1]], color=c, lw=lw, alpha=0.6, zorder=2)

# Nodes colored by received attention
na_norm = (node_attn_mean - node_attn_mean.min()) / (node_attn_mean.max() - node_attn_mean.min() + 1e-8)
cmap_n = plt.cm.Blues
for i, (px, py) in enumerate(px_list):
    c = cmap_n(0.3 + na_norm[i]*0.7)
    circle = plt.Circle((px, py), 6, color=c, ec="#1a1a2e", lw=1.2, zorder=5)
    ax.add_patch(circle)
    ax.text(px, py-9, stations[i], ha="center", va="bottom", fontsize=5.5,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3), zorder=6)

sm = ScalarMappable(cmap=cmap_e, norm=Normalize(attn_mean.min(), attn_mean.max()))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
cb.set_label("Mean GAT Attention Weight", fontsize=8)

ax.set_title("Fig D1-1. GAT Spatial Attention Map (평균)\n"
             "엣지 색상·굵기 = mean attention, 노드 색 = received attention",
             fontsize=10, fontweight="bold", pad=8)
ax.text(0.99, 0.01, "© OpenStreetMap contributors",
        transform=ax.transAxes, fontsize=5, ha="right", va="bottom",
        color="#555", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d1_1_attn_map_osm.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d1_1_attn_map_osm.png")
plt.close()

# ── Fig D1-2: Per-node received attention bar chart ───────────────────────────
print("Generating fig_d1_2...")
order = np.argsort(node_attn_mean)[::-1]
sorted_names = [stations[i] for i in order]
sorted_vals  = node_attn_mean[order]

fig, ax = plt.subplots(figsize=(12, 4.5))
colors = plt.cm.YlOrRd(0.3 + (sorted_vals - sorted_vals.min()) /
                        (sorted_vals.max() - sorted_vals.min() + 1e-8) * 0.7)
bars = ax.bar(range(N), sorted_vals, color=colors, width=0.7, alpha=0.9)
ax.set_xticks(range(N))
ax.set_xticklabels(sorted_names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Mean Received Attention", fontsize=10)
ax.set_title("Fig D1-2. 관측소별 평균 수신 Attention (높은 순 정렬)\n"
             "해당 관측소로 들어오는 엣지의 attention weight 평균", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d1_2_node_attention.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d1_2_node_attention.png")
plt.close()

# ── Fig D1-3: Per-head attention comparison ───────────────────────────────────
print("Generating fig_d1_3...")
attn_by_head = attn_per_ts.mean(axis=(0, 1))   # [E, H]
H = attn_by_head.shape[1]

fig, axes = plt.subplots(1, H, figsize=(14, 4), sharey=False)
fig.suptitle("Fig D1-3. GAT Attention Head별 엣지 분포\n"
             "(각 head의 attention weight histogram)", fontsize=11, fontweight="bold")

head_colors = ["#2B6CB0", "#C0185A", "#2D9E4F", "#D69E2E"]
for h_idx, ax in enumerate(axes):
    vals = attn_by_head[:, h_idx]
    ax.hist(vals, bins=40, color=head_colors[h_idx], alpha=0.8, edgecolor="white")
    ax.axvline(vals.mean(), color="black", lw=1.5, linestyle="--",
               label=f"mean={vals.mean():.4f}")
    ax.set_title(f"Head {h_idx+1}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Attention Weight", fontsize=9)
    ax.set_ylabel("Edge Count" if h_idx == 0 else "", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d1_3_head_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d1_3_head_comparison.png")
plt.close()

print("D1 Done.")

"""
Appendix D3: Case Studies on High-PM Events

High-PM 기준: 테스트셋 평균 PM10 > 80 µg/m³ (국내 '나쁨' 기준)

  fig_d3_1_pm_threshold.png       — PM10 분포 + High-PM 기준선
  fig_d3_2_attn_highvsnormal.png  — High-PM vs Normal 기간 attention 비교
  fig_d3_3_case_event.png         — 가장 심각한 고농도 이벤트 attention 지도
"""

import sys, io, time, math, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import requests
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = "/workspace/ST-GNN Modeling"
sys.path.insert(0, ROOT)

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})

OUT = "Appendix/Appendix_D3"
HIGH_PM_THRESH = 33.0   # µg/m³, 테스트셋 도시평균 PM10 상위 10% (p90)

# ── Load cache ────────────────────────────────────────────────────────────────
cache = np.load(f"{ROOT}/Appendix/attn_cache.npz")
attn_per_ts  = cache["attn_per_ts"]   # [N_test, T, E, H]
edge_index   = cache["edge_index"]
static_attr  = cache["static_attr"]
edge_bearings = cache["edge_bearings"]
test_pm10    = cache["test_pm10"]     # [N_test, N] true PM10 µg/m³

from dataset import load_scenario_split
(_, _, test_nodes, *_rest, coords, feat_cols) = load_scenario_split("S3_transport_pm10_pollutants")

_csv = pd.read_csv("/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios/S3_transport_pm10_pollutants.csv")
stations = sorted(_csv["측정소명"].unique())
_cdf = _csv.drop_duplicates("측정소명").set_index("측정소명")[["위도","경도"]]
coords   = [(float(_cdf.loc[s,"위도"]), float(_cdf.loc[s,"경도"])) for s in stations]
N = len(stations)
E = edge_index.shape[1]

# City-wide mean PM10 per test sample
pm10_city_mean = test_pm10.mean(axis=1)   # [N_test]
high_mask   = pm10_city_mean >= HIGH_PM_THRESH
normal_mask = pm10_city_mean <  HIGH_PM_THRESH

print(f"Total test samples : {len(pm10_city_mean)}")
print(f"High-PM (≥{HIGH_PM_THRESH}) : {high_mask.sum()} ({high_mask.mean()*100:.1f}%)")
print(f"Normal (<{HIGH_PM_THRESH})  : {normal_mask.sum()} ({normal_mask.mean()*100:.1f}%)")

# ── Fig D3-1: PM10 distribution + threshold ───────────────────────────────────
print("Generating fig_d3_1...")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(pm10_city_mean, bins=60, color="#4A90D9", alpha=0.8, edgecolor="white", label="Normal")
ax.hist(pm10_city_mean[high_mask], bins=30, color="#E74C3C", alpha=0.85,
        edgecolor="white", label=f"High-PM (≥{HIGH_PM_THRESH} µg/m³)")
ax.axvline(HIGH_PM_THRESH, color="#C0392B", lw=2, linestyle="--",
           label=f"기준선 {HIGH_PM_THRESH} µg/m³ (상위 10%, p90)")
ax.axvline(pm10_city_mean.mean(), color="#2D3748", lw=1.5, linestyle=":",
           label=f"평균 {pm10_city_mean.mean():.1f} µg/m³")

ax.set_xlabel("City-wide Mean PM10 (µg/m³)", fontsize=10)
ax.set_ylabel("Sample Count", fontsize=10)
ax.set_title("Fig D3-1. 테스트셋 PM10 농도 분포 및 High-PM 이벤트 기준\n"
             f"(총 {len(pm10_city_mean)}개 샘플, High-PM {high_mask.sum()}개)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d3_1_pm_threshold.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d3_1_pm_threshold.png")
plt.close()

# ── Fig D3-2: Attention High-PM vs Normal ────────────────────────────────────
print("Generating fig_d3_2...")
attn_high   = attn_per_ts[high_mask].mean(axis=(0,1,3))    # [E]
attn_normal = attn_per_ts[normal_mask].mean(axis=(0,1,3))  # [E]
attn_diff   = attn_high - attn_normal                      # [E]

# Node-level received attention
def node_received(attn_e, N, ei):
    na = np.zeros(N); cnt = np.zeros(N)
    for e in range(ei.shape[1]):
        na[ei[1,e]] += attn_e[e]; cnt[ei[1,e]] += 1
    return na / (cnt + 1e-8)

na_high   = node_received(attn_high,   N, edge_index)
na_normal = node_received(attn_normal, N, edge_index)
na_diff   = na_high - na_normal

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Fig D3-2. High-PM vs Normal 기간 — 관측소별 수신 Attention 비교",
             fontsize=11, fontweight="bold")

def bar_plot(ax, vals, title, cmap, label):
    order = np.argsort(vals)[::-1]
    colors = plt.get_cmap(cmap)(0.3 + (vals[order] - vals[order].min()) /
                                (vals[order].max() - vals[order].min() + 1e-8) * 0.7)
    ax.bar(range(N), vals[order], color=colors, width=0.7, alpha=0.9)
    ax.set_xticks(range(N))
    ax.set_xticklabels([stations[i] for i in order], rotation=65, ha="right", fontsize=6.5)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

bar_plot(axes[0], na_high,   "High-PM 기간", "Reds",   "Mean Received Attention")
bar_plot(axes[1], na_normal, "Normal 기간",  "Blues",  "Mean Received Attention")

# Diff plot
order_d = np.argsort(na_diff)[::-1]
colors_d = ["#e74c3c" if v>0 else "#2980b9" for v in na_diff[order_d]]
axes[2].bar(range(N), na_diff[order_d], color=colors_d, width=0.7, alpha=0.9)
axes[2].axhline(0, color="black", lw=1)
axes[2].set_xticks(range(N))
axes[2].set_xticklabels([stations[i] for i in order_d], rotation=65, ha="right", fontsize=6.5)
axes[2].set_ylabel("Δ Attention (High − Normal)", fontsize=9)
axes[2].set_title("차이 (빨강=High-PM에서 ↑)", fontsize=9, fontweight="bold")
axes[2].grid(axis="y", alpha=0.3, linestyle="--")
axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d3_2_attn_highvsnormal.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d3_2_attn_highvsnormal.png")
plt.close()

# ── Fig D3-3: Worst event case study on OSM map ───────────────────────────────
print("Generating fig_d3_3...")

worst_idx = int(np.argmax(pm10_city_mean))
worst_pm10_val = pm10_city_mean[worst_idx]
worst_attn_e   = attn_per_ts[worst_idx].mean(axis=(0,2))   # [E] mean over T, H
worst_pm10_n   = test_pm10[worst_idx]                      # [N]
print(f"  Worst event: sample {worst_idx}, mean PM10={worst_pm10_val:.1f} µg/m³")

# OSM map helpers (same as D1)
def deg_to_tile(lat, lon, z):
    n=2**z; x=int((lon+180)/360*n)
    y=int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
    return x,y
def tile_to_deg(x,y,z):
    n=2**z; lon=x/n*360-180
    lat=math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n))))
    return lat,lon
def fetch_map(lat_min,lat_max,lon_min,lon_max,zoom=12,pad=0.06):
    dlat=(lat_max-lat_min)*pad; dlon=(lon_max-lon_min)*pad
    lat_min-=dlat; lat_max+=dlat; lon_min-=dlon; lon_max+=dlon
    x0,y0=deg_to_tile(lat_max,lon_min,zoom); x1,y1=deg_to_tile(lat_min,lon_max,zoom)
    x0,x1=min(x0,x1),max(x0,x1); y0,y1=min(y0,y1),max(y0,y1)
    tw,th=x1-x0+1,y1-y0+1
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
    lt,ll=tile_to_deg(x0,y0,zoom); lb,lr=tile_to_deg(x1+1,y1+1,zoom)
    return np.array(canvas),(lt,lb,ll,lr)
def ll2px(lat,lon,bounds,img_shape):
    lt,lb,ll,lr=bounds; h,w=img_shape[:2]
    def my(ld): return math.log(math.tan(math.pi/4+math.radians(ld)/2))
    return (lon-ll)/(lr-ll)*w, (my(lt)-my(lat))/(my(lt)-my(lb))*h

lats=[c[0] for c in coords]; lons=[c[1] for c in coords]
print("  Fetching OSM tiles...")
img, bounds = fetch_map(min(lats),max(lats),min(lons),max(lons),zoom=12)
px_list = [ll2px(lat,lon,bounds,img.shape) for lat,lon in coords]
h_img, w_img = img.shape[:2]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"Fig D3-3. High-PM 이벤트 Case Study\n"
             f"(테스트 최고 농도 시점, 도시평균 PM10 = {worst_pm10_val:.1f} µg/m³)",
             fontsize=11, fontweight="bold")

# Left: Attention map for this event
ax = axes[0]
ax.imshow(img, extent=[0,w_img,h_img,0], zorder=0)
ax.set_xlim(0,w_img); ax.set_ylim(h_img,0); ax.axis("off")
ax.set_title("GAT Attention (이벤트 시점)", fontsize=10, fontweight="bold")

attn_n = (worst_attn_e - worst_attn_e.min()) / (worst_attn_e.max() - worst_attn_e.min() + 1e-8)
cmap_e = plt.cm.YlOrRd
for e in range(E):
    si, di = edge_index[0,e], edge_index[1,e]
    ps, pd_ = px_list[si], px_list[di]
    ax.plot([ps[0],pd_[0]],[ps[1],pd_[1]], color=cmap_e(attn_n[e]),
            lw=0.5+attn_n[e]*2.5, alpha=0.65, zorder=2)

pm10_norm = (worst_pm10_n - worst_pm10_n.min()) / (worst_pm10_n.max() - worst_pm10_n.min() + 1e-8)
cmap_n = plt.cm.Reds
for i,(px,py) in enumerate(px_list):
    c = cmap_n(0.25 + pm10_norm[i]*0.75)
    circle = plt.Circle((px,py), 7, color=c, ec="#1a1a2e", lw=1.2, zorder=5)
    ax.add_patch(circle)
    ax.text(px, py-10, stations[i], ha="center", va="bottom", fontsize=5,
            bbox=dict(fc="white",ec="none",alpha=0.7,pad=0.3), zorder=6)

sm1 = ScalarMappable(cmap=cmap_e, norm=Normalize(worst_attn_e.min(), worst_attn_e.max()))
sm1.set_array([]); plt.colorbar(sm1, ax=ax, shrink=0.5, pad=0.01, label="Attention Weight")

# Right: PM10 distribution bar
ax2 = axes[1]
order_n = np.argsort(worst_pm10_n)[::-1]
colors_n = plt.cm.Reds(0.3 + (worst_pm10_n[order_n] - worst_pm10_n.min()) /
                        (worst_pm10_n.max() - worst_pm10_n.min() + 1e-8) * 0.7)
bars = ax2.bar(range(N), worst_pm10_n[order_n], color=colors_n, width=0.7, alpha=0.9)
ax2.axhline(HIGH_PM_THRESH, color="#C0392B", lw=2, linestyle="--", label=f"High-PM 기준 ({HIGH_PM_THRESH}, p90)")
ax2.axhline(worst_pm10_n.mean(), color="#2D3748", lw=1.5, linestyle=":",
            label=f"도시 평균 ({worst_pm10_n.mean():.1f})")
ax2.set_xticks(range(N))
ax2.set_xticklabels([stations[i] for i in order_n], rotation=65, ha="right", fontsize=7)
ax2.set_ylabel("PM10 (µg/m³)", fontsize=10)
ax2.set_title("관측소별 PM10 농도 (이벤트 시점)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

ax.text(0.99,0.01,"© OpenStreetMap contributors",transform=ax.transAxes,
        fontsize=5,ha="right",va="bottom",color="#555",style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_d3_3_case_event.png", dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUT}/fig_d3_3_case_event.png")
plt.close()

print("D3 Done.")

"""
Appendix: ST-GNN Graph Mode visualizations on real Seoul OpenStreetMap.

Fetches OSM tiles via requests + PIL, then overlays graph edges/nodes
using matplotlib with proper lat/lon → pixel coordinate transforms.

Output:
    appendix/osm_graph_modes_overview.png   — 3-panel overview on Seoul map
    appendix/osm_static_graph.png
    appendix/osm_climatological_graph.png
    appendix/osm_soft_dynamic_graph.png
"""

import math
import io
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import requests
from PIL import Image
import pickle
from pathlib import Path

# ── Korean font ───────────────────────────────────────────────────────────────
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV   = Path("/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios/S1_transport_pm10.csv")
SPLIT_INFO = Path("/workspace/ST-GNN Modeling/split_info.pkl")
GRAPH_DIR  = Path("/workspace/ST-GNN Modeling/graphs")
OUT_DIR    = Path("/workspace/ST-GNN Modeling/appendix")

# ── OSM tile helpers ──────────────────────────────────────────────────────────

def deg_to_tile(lat_deg: float, lon_deg: float, zoom: int):
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_deg(x: int, y: int, zoom: int):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def fetch_tile(x: int, y: int, zoom: int, session, retries=3) -> Image.Image:
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {"User-Agent": "ST-GNN-Research/1.0 (academic use)"}
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
        time.sleep(1)
    # Return a blank gray tile on failure
    return Image.new("RGB", (256, 256), (220, 220, 220))

def fetch_map_image(lat_min, lat_max, lon_min, lon_max, zoom):
    """Fetch and stitch OSM tiles covering the bounding box."""
    # Add padding
    pad_frac = 0.08
    dlat = (lat_max - lat_min) * pad_frac
    dlon = (lon_max - lon_min) * pad_frac
    lat_min -= dlat;  lat_max += dlat
    lon_min -= dlon;  lon_max += dlon

    x0, y0 = deg_to_tile(lat_max, lon_min, zoom)
    x1, y1 = deg_to_tile(lat_min, lon_max, zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    tile_w = x1 - x0 + 1
    tile_h = y1 - y0 + 1
    canvas = Image.new("RGB", (tile_w * 256, tile_h * 256))

    session = requests.Session()
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            tile = fetch_tile(tx, ty, zoom, session)
            px = (tx - x0) * 256
            py = (ty - y0) * 256
            canvas.paste(tile, (px, py))
            time.sleep(0.05)   # polite rate limit

    # Bounds in degrees of the stitched image
    lat_top, lon_left  = tile_to_deg(x0,     y0,     zoom)
    lat_bot, lon_right = tile_to_deg(x1 + 1, y1 + 1, zoom)
    return np.array(canvas), (lat_top, lat_bot, lon_left, lon_right), (tile_w, tile_h)

def latlon_to_px(lat, lon, lat_top, lat_bot, lon_left, lon_right, img_w, img_h, zoom):
    """Project lat/lon to pixel coordinates in the stitched tile image."""
    n = 2 ** zoom

    def merc_y(lat_d):
        lat_r = math.radians(lat_d)
        return math.log(math.tan(math.pi / 4 + lat_r / 2))

    my_top = merc_y(lat_top)
    my_bot = merc_y(lat_bot)
    my_pt  = merc_y(lat)

    frac_x = (lon - lon_left)  / (lon_right - lon_left)
    frac_y = (my_top - my_pt)  / (my_top - my_bot)
    return frac_x * img_w, frac_y * img_h


# ── Graph + station data loading ──────────────────────────────────────────────

def load_stations():
    df = pd.read_csv(DATA_CSV)
    stations = sorted(df["측정소명"].unique())
    coord_df = df.drop_duplicates("측정소명").set_index("측정소명")[["위도", "경도"]]
    coords = [(float(coord_df.loc[s, "위도"]), float(coord_df.loc[s, "경도"])) for s in stations]
    return stations, coords

def load_graph(mode: str):
    key = "static" if mode == "soft_dynamic" else mode
    ei  = np.load(GRAPH_DIR / key / "edge_index.npy")
    sa  = np.load(GRAPH_DIR / key / "static_attr.npy")
    eb  = np.load(GRAPH_DIR / key / "edge_bearings.npy")
    return ei, sa, eb

def get_soft_dynamic_active(edge_index, edge_bearings, coords, wind_dir_deg=290.0):
    """Simulate soft-dynamic active/inactive edges using a representative wind."""
    uw = math.sin(math.radians(wind_dir_deg))
    vw = math.cos(math.radians(wind_dir_deg))
    active, inactive = [], []
    for e in range(edge_index.shape[1]):
        b = math.radians(edge_bearings[e])
        alignment = uw * math.sin(b) + vw * math.cos(b)
        (active if alignment > 0 else inactive).append(e)
    return active, inactive


# ── Drawing helpers ───────────────────────────────────────────────────────────

PALETTE = {
    "node_face":   "#1E6FBA",
    "node_edge":   "#0D3B6E",
    "edge_close":  "#2D9E4F",
    "edge_mid":    "#D4891A",
    "edge_far":    "#C0392B",
    "clim_edge":   "#1B6B3A",
    "active_edge": "#C0185A",
    "inactive_edge": "#B0BEC5",
    "wind_arrow":  "#37474F",
    "label_bg":    "white",
}

def edge_dist_color(norm_dist):
    if norm_dist < 0.4:   return PALETTE["edge_close"]
    elif norm_dist < 0.7: return PALETTE["edge_mid"]
    return PALETTE["edge_far"]

def px_coords(lat, lon, bounds, img_shape):
    lat_top, lat_bot, lon_left, lon_right = bounds
    h, w = img_shape[:2]
    return latlon_to_px(lat, lon, lat_top, lat_bot, lon_left, lon_right, w, h, ZOOM)

def draw_edge_line(ax, p_src, p_dst, color, lw=1.2, alpha=0.7, zorder=2):
    ax.plot([p_src[0], p_dst[0]], [p_src[1], p_dst[1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round")

def draw_edge_arrow(ax, p_src, p_dst, color, lw=1.5, alpha=0.85, zorder=3):
    dx = p_dst[0] - p_src[0]
    dy = p_dst[1] - p_src[1]
    length = math.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        return
    sh = 0.15
    sx = p_src[0] + dx * sh
    sy = p_src[1] + dy * sh
    ex = p_dst[0] - dx * sh
    ey = p_dst[1] - dy * sh
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=9),
                zorder=zorder, alpha=alpha)

def draw_nodes(ax, stations, px_list):
    for name, (px, py) in zip(stations, px_list):
        circle = plt.Circle((px, py), NODE_R, color=PALETTE["node_face"],
                             ec=PALETTE["node_edge"], lw=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(px, py + NODE_R + 2, name, ha="center", va="bottom",
                fontsize=5.5, color="#1A1A2E", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.3))

def add_wind_indicator(ax, wind_deg, img_w, img_h, scale=40):
    """Small wind arrow in bottom-right corner."""
    cx, cy = img_w * 0.90, img_h * 0.88
    uw = math.sin(math.radians(wind_deg)) * scale
    vw = math.cos(math.radians(wind_deg)) * scale
    ax.annotate("", xy=(cx + uw, cy - vw), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["wind_arrow"],
                                lw=2.0, mutation_scale=12), zorder=10)
    ax.text(cx, cy + scale * 0.6 + 4, f"지배 풍향\n{int(wind_deg)}°",
            ha="center", va="bottom", fontsize=6, color=PALETTE["wind_arrow"],
            bbox=dict(fc="white", ec=PALETTE["wind_arrow"], alpha=0.85,
                      pad=2, boxstyle="round,pad=0.3"), zorder=11)

def style_ax(ax, img, title, subtitle=""):
    h, w = img.shape[:2]
    ax.imshow(img, extent=[0, w, h, 0], zorder=0, aspect="equal")
    ax.set_xlim(0, w);  ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold",
                 color="#1A202C", pad=6, loc="center")


# ── Main generation ───────────────────────────────────────────────────────────

ZOOM   = 12
NODE_R = 5.5       # pixel radius for nodes


def generate():
    print("Loading station data...")
    stations, coords = load_stations()
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    print(f"Fetching OSM tiles (zoom={ZOOM}) for Seoul bounding box...")
    img, bounds, _ = fetch_map_image(
        min(lats), max(lats), min(lons), max(lons), ZOOM
    )
    h, w = img.shape[:2]
    print(f"  Map image: {w} x {h} px")

    # Pre-compute pixel positions for all stations
    px_list = [px_coords(lat, lon, bounds, img.shape) for lat, lon in coords]

    # Load graphs
    ei_s, sa_s, eb_s = load_graph("static")
    ei_c, sa_c, eb_c = load_graph("climatological")
    active_idx, inactive_idx = get_soft_dynamic_active(ei_s, eb_s, coords, wind_dir_deg=290.0)

    # ── 3-panel overview ──────────────────────────────────────────────────────
    print("Generating 3-panel overview...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.92, bottom=0.02)

    # Panel A — Static
    ax = axes[0]
    style_ax(ax, img, "Static Graph",
             f"양방향 · {ei_s.shape[1]} edges · 거리 기반 임계값 (10 km)")
    grp = {}
    for e in range(ei_s.shape[1]):
        si, di = ei_s[0, e], ei_s[1, e]
        c = edge_dist_color(float(sa_s[e, 0]))
        draw_edge_line(ax, px_list[si], px_list[di], c, lw=1.1, alpha=0.60)
    draw_nodes(ax, stations, px_list)
    ax.text(0.02, 0.98, "(A)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", color="#2D3748")
    ax.legend(handles=[
        Line2D([0],[0], color=PALETTE["edge_close"], lw=2, label="< 4 km"),
        Line2D([0],[0], color=PALETTE["edge_mid"],   lw=2, label="4–7 km"),
        Line2D([0],[0], color=PALETTE["edge_far"],   lw=2, label="7–10 km"),
    ], title="엣지 거리", title_fontsize=7, fontsize=7,
       loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

    # Panel B — Climatological
    ax = axes[1]
    style_ax(ax, img, "Climatological Graph",
             f"단방향 · {ei_c.shape[1]} edges · 훈련 기간 평균 풍향 필터링")
    for e in range(ei_c.shape[1]):
        si, di = ei_c[0, e], ei_c[1, e]
        draw_edge_arrow(ax, px_list[si], px_list[di],
                        PALETTE["clim_edge"], lw=1.3, alpha=0.75)
    draw_nodes(ax, stations, px_list)
    add_wind_indicator(ax, 315, w, h, scale=35)
    ax.text(0.02, 0.98, "(B)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", color="#2D3748")
    ax.legend(handles=[
        Line2D([0],[0], color=PALETTE["clim_edge"], lw=2,
               marker=">", markersize=6, label="평균 풍향 방향 엣지"),
    ], fontsize=7, loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

    # Panel C — Soft-Dynamic
    ax = axes[2]
    style_ax(ax, img, "Soft-Dynamic Graph",
             f"활성 {len(active_idx)} / {ei_s.shape[1]} edges · 실시간 풍향 masking")
    for e in inactive_idx:
        si, di = ei_s[0, e], ei_s[1, e]
        draw_edge_line(ax, px_list[si], px_list[di],
                       PALETTE["inactive_edge"], lw=0.8, alpha=0.35)
    for e in active_idx:
        si, di = ei_s[0, e], ei_s[1, e]
        draw_edge_arrow(ax, px_list[si], px_list[di],
                        PALETTE["active_edge"], lw=1.5, alpha=0.85)
    draw_nodes(ax, stations, px_list)
    add_wind_indicator(ax, 290, w, h, scale=35)
    ax.text(0.02, 0.98, "(C)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", color="#2D3748")
    ax.legend(handles=[
        Line2D([0],[0], color=PALETTE["active_edge"], lw=2,
               marker=">", markersize=6, label=f"활성 엣지 ({len(active_idx)})"),
        Line2D([0],[0], color=PALETTE["inactive_edge"], lw=2,
               label=f"비활성 엣지 ({len(inactive_idx)})"),
    ], fontsize=7, loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

    fig.suptitle("ST-GNN Graph Mode 비교 (서울 관측소)", fontsize=13,
                 fontweight="bold", y=0.99, color="#1A202C")
    out = OUT_DIR / "osm_graph_modes_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out}")
    plt.close(fig)

    # ── Individual figures ────────────────────────────────────────────────────
    configs = [
        ("static", "Static Graph",
         f"양방향 · {ei_s.shape[1]} edges · 거리 기반 임계값 (10 km)",
         None),
        ("climatological", "Climatological Graph",
         f"단방향 · {ei_c.shape[1]} edges · 훈련 기간 평균 풍향 필터링",
         315),
        ("soft_dynamic", "Soft-Dynamic Graph",
         f"활성 {len(active_idx)} / {ei_s.shape[1]} edges · 실시간 풍향 masking (290°)",
         290),
    ]

    for mode, title, subtitle, wind_deg in configs:
        print(f"Generating individual: {mode}...")
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("white")
        style_ax(ax, img, title, subtitle)

        if mode == "static":
            for e in range(ei_s.shape[1]):
                si, di = ei_s[0, e], ei_s[1, e]
                c = edge_dist_color(float(sa_s[e, 0]))
                draw_edge_line(ax, px_list[si], px_list[di], c, lw=1.2, alpha=0.65)
            ax.legend(handles=[
                Line2D([0],[0], color=PALETTE["edge_close"], lw=2, label="< 4 km"),
                Line2D([0],[0], color=PALETTE["edge_mid"],   lw=2, label="4–7 km"),
                Line2D([0],[0], color=PALETTE["edge_far"],   lw=2, label="7–10 km"),
            ], title="엣지 거리", title_fontsize=8, fontsize=8,
               loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

        elif mode == "climatological":
            for e in range(ei_c.shape[1]):
                si, di = ei_c[0, e], ei_c[1, e]
                draw_edge_arrow(ax, px_list[si], px_list[di],
                                PALETTE["clim_edge"], lw=1.4, alpha=0.80)
            add_wind_indicator(ax, wind_deg, w, h, scale=40)
            ax.legend(handles=[
                Line2D([0],[0], color=PALETTE["clim_edge"], lw=2,
                       marker=">", markersize=7, label="평균 풍향 방향 엣지"),
            ], fontsize=8, loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

        else:  # soft_dynamic
            for e in inactive_idx:
                si, di = ei_s[0, e], ei_s[1, e]
                draw_edge_line(ax, px_list[si], px_list[di],
                               PALETTE["inactive_edge"], lw=0.9, alpha=0.38)
            for e in active_idx:
                si, di = ei_s[0, e], ei_s[1, e]
                draw_edge_arrow(ax, px_list[si], px_list[di],
                                PALETTE["active_edge"], lw=1.6, alpha=0.88)
            add_wind_indicator(ax, wind_deg, w, h, scale=40)
            ax.legend(handles=[
                Line2D([0],[0], color=PALETTE["active_edge"], lw=2,
                       marker=">", markersize=7, label=f"활성 엣지 ({len(active_idx)})"),
                Line2D([0],[0], color=PALETTE["inactive_edge"], lw=2,
                       label=f"비활성 엣지 ({len(inactive_idx)})"),
            ], fontsize=8, loc="lower left", framealpha=0.9, edgecolor="#CBD5E0")

        draw_nodes(ax, stations, px_list)

        # OSM attribution (required)
        ax.text(0.99, 0.01, "© OpenStreetMap contributors",
                transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom",
                color="#555555", style="italic")

        plt.tight_layout(pad=0.5)
        out = OUT_DIR / f"osm_{mode}_graph.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    generate()

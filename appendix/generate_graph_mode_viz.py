"""
Appendix: Graph Mode Schematic Visualizations for ST-GNN Paper.

Generates three panel figures illustrating Static, Climatological, and Soft-Dynamic
graph modes using a synthetic but representative Seoul station layout.

Output files:
    appendix/fig_graph_modes_overview.png   — 3-panel overview figure
    appendix/fig_static_graph.png           — Static mode only
    appendix/fig_climatological_graph.png   — Climatological mode only
    appendix/fig_soft_dynamic_graph.png     — Soft-Dynamic mode only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# Register NanumGothic for Korean text rendering
_NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fm.fontManager.addfont(_NANUM_PATH)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ─── Synthetic station layout (mimics Seoul station spread) ───────────────────
# 14 representative nodes placed in a 2D canvas (lon-like x, lat-like y)
NODES = np.array([
    [0.15, 0.85],  # 0 은평
    [0.32, 0.90],  # 1 도봉
    [0.55, 0.88],  # 2 노원
    [0.75, 0.82],  # 3 중랑
    [0.10, 0.65],  # 4 서대문
    [0.28, 0.70],  # 5 종로
    [0.50, 0.72],  # 6 성북
    [0.70, 0.68],  # 7 광진
    [0.18, 0.48],  # 8 마포
    [0.35, 0.52],  # 9 중구
    [0.55, 0.50],  # 10 성동
    [0.72, 0.52],  # 11 강동
    [0.30, 0.30],  # 12 영등포
    [0.52, 0.28],  # 13 서초
    [0.72, 0.30],  # 14 강남
])
N = len(NODES)
NAMES = ["은평", "도봉", "노원", "중랑", "서대문", "종로", "성북",
         "광진", "마포", "중구", "성동", "강동", "영등포", "서초", "강남"]
THRESHOLD = 0.32   # distance threshold in canvas units

# ─── Edge construction ────────────────────────────────────────────────────────

def get_dist(i, j):
    return np.linalg.norm(NODES[i] - NODES[j])

def get_bearing_deg(i, j):
    dx = NODES[j, 0] - NODES[i, 0]
    dy = NODES[j, 1] - NODES[i, 1]
    return np.degrees(np.arctan2(dx, dy)) % 360

# Static graph: all pairs within threshold
static_edges = []
for i in range(N):
    for j in range(N):
        if i != j and get_dist(i, j) < THRESHOLD:
            static_edges.append((i, j))

# Climatological: dominant wind from NW (330°) → keep edge if wind component > 0
# Seoul's mean winter-spring wind is ~NW-W (dominant pollution transport direction)
DOMINANT_WIND_DIR = 315.0   # degrees (NW)
uw = np.sin(np.radians(DOMINANT_WIND_DIR))   # east component
vw = np.cos(np.radians(DOMINANT_WIND_DIR))   # north component

def wind_alignment(i, j, uw=uw, vw=vw):
    b = np.radians(get_bearing_deg(i, j))
    return uw * np.sin(b) + vw * np.cos(b)

clim_edges = [(i, j) for (i, j) in static_edges if wind_alignment(i, j) > 0]

# Soft-dynamic: active edges at a specific timestep (wind rotated slightly)
TIMESTEP_WIND_DIR = 290.0   # slightly W of NW — more westerly at this snapshot
uw_t = np.sin(np.radians(TIMESTEP_WIND_DIR))
vw_t = np.cos(np.radians(TIMESTEP_WIND_DIR))
active_edges = [(i, j) for (i, j) in static_edges
                if wind_alignment(i, j, uw_t, vw_t) > 0]
inactive_edges = [(i, j) for (i, j) in static_edges
                  if (i, j) not in active_edges]


# ─── Style helpers ────────────────────────────────────────────────────────────

PALETTE = {
    "bg":          "#F8F9FA",
    "node_face":   "#2B6CB0",
    "node_edge":   "#1A365D",
    "edge_close":  "#38A169",
    "edge_mid":    "#D69E2E",
    "edge_far":    "#C53030",
    "clim_edge":   "#2D6A4F",
    "active_edge": "#D53F8C",
    "inactive_edge": "#CBD5E0",
    "arrow_color": "#744210",
    "wind_arrow":  "#4A5568",
    "label_bg":    "white",
}

def edge_dist_color(i, j):
    d = get_dist(i, j) / THRESHOLD
    if d < 0.4:
        return PALETTE["edge_close"]
    elif d < 0.7:
        return PALETTE["edge_mid"]
    return PALETTE["edge_far"]

def draw_nodes(ax, highlight=None):
    for i, (x, y) in enumerate(NODES):
        fc = "#4299E1" if highlight and i in highlight else PALETTE["node_face"]
        circle = plt.Circle((x, y), 0.028, color=fc, ec=PALETTE["node_edge"],
                             linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y - 0.062, NAMES[i], ha="center", va="top",
                fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5), zorder=6)

def draw_line(ax, i, j, color, lw=1.2, alpha=0.7, zorder=2):
    x0, y0 = NODES[i]
    x1, y1 = NODES[j]
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=zorder)

def draw_arrow(ax, i, j, color, lw=1.5, alpha=0.85, zorder=3,
               head_width=0.018, shrink_frac=0.12):
    x0, y0 = NODES[i]
    x1, y1 = NODES[j]
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    shrink = shrink_frac * length
    sx, sy = x0 + dx * shrink / length, y0 + dy * shrink / length
    ex, ey = x1 - dx * shrink / length, y1 - dy * shrink / length
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=zorder, alpha=alpha)

def draw_wind_compass(ax, deg, x=0.88, y=0.88, scale=0.09, label=True):
    uw = np.sin(np.radians(deg)) * scale
    vw = np.cos(np.radians(deg)) * scale
    ax.annotate("", xy=(x + uw, y + vw), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["wind_arrow"],
                                lw=2.0, mutation_scale=12), zorder=10)
    if label:
        ax.text(x, y - 0.12, f"지배 풍향\n{int(deg)}°", ha="center", va="top",
                fontsize=6.5, color=PALETTE["wind_arrow"],
                bbox=dict(fc="white", ec=PALETTE["wind_arrow"], alpha=0.85,
                          pad=2, boxstyle="round,pad=0.3"))

def style_ax(ax, title, subtitle=""):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])
    ax.text(0.5, 1.05, title, ha="center", va="bottom", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#1A202C")
    if subtitle:
        ax.text(0.5, 1.01, subtitle, ha="center", va="bottom", transform=ax.transAxes,
                fontsize=8, color="#4A5568")


# ─── Panel 1: Static Graph ────────────────────────────────────────────────────

def draw_static(ax):
    for (i, j) in static_edges:
        draw_line(ax, i, j, edge_dist_color(i, j), lw=1.3, alpha=0.55)
    draw_nodes(ax)
    style_ax(ax, "Static Graph",
             f"양방향 · {len(static_edges)} edges · 거리 기반 임계값 (10 km)")

    legend_els = [
        Line2D([0], [0], color=PALETTE["edge_close"], lw=2, label="< 4 km"),
        Line2D([0], [0], color=PALETTE["edge_mid"],   lw=2, label="4–7 km"),
        Line2D([0], [0], color=PALETTE["edge_far"],   lw=2, label="7–10 km"),
    ]
    ax.legend(handles=legend_els, title="엣지 거리", title_fontsize=7,
              fontsize=6.5, loc="lower left", framealpha=0.9,
              edgecolor="#CBD5E0", fancybox=True)


# ─── Panel 2: Climatological Graph ───────────────────────────────────────────

def draw_climatological(ax):
    for (i, j) in clim_edges:
        draw_arrow(ax, i, j, PALETTE["clim_edge"], lw=1.4, alpha=0.75)
    draw_nodes(ax)
    draw_wind_compass(ax, DOMINANT_WIND_DIR, x=0.87, y=0.17, scale=0.09)
    style_ax(ax, "Climatological Graph",
             f"단방향 · {len(clim_edges)} edges · 훈련 기간 평균 풍향 필터링")

    legend_els = [
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.005,
                            color=PALETTE["clim_edge"]),
    ]
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["clim_edge"], lw=2,
                   marker=">", markersize=6, label="평균 풍향 방향 엣지"),
        ],
        fontsize=6.5, loc="lower left", framealpha=0.9,
        edgecolor="#CBD5E0", fancybox=True)


# ─── Panel 3: Soft-Dynamic Graph ─────────────────────────────────────────────

def draw_soft_dynamic(ax):
    for (i, j) in inactive_edges:
        draw_line(ax, i, j, PALETTE["inactive_edge"], lw=0.9, alpha=0.4)
    for (i, j) in active_edges:
        wa = wind_alignment(i, j, uw_t, vw_t)
        lw = 1.0 + min(wa * 0.8, 2.5)
        draw_arrow(ax, i, j, PALETTE["active_edge"], lw=lw, alpha=0.85)
    draw_nodes(ax)
    draw_wind_compass(ax, TIMESTEP_WIND_DIR, x=0.87, y=0.17, scale=0.09, label=True)
    style_ax(ax, "Soft-Dynamic Graph",
             f"t={0} · 활성 {len(active_edges)} / {len(static_edges)} edges · 실시간 풍향 masking")

    legend_els = [
        Line2D([0], [0], color=PALETTE["active_edge"], lw=2,
               marker=">", markersize=6, label=f"활성 엣지 ({len(active_edges)})"),
        Line2D([0], [0], color=PALETTE["inactive_edge"], lw=2,
               label=f"비활성 엣지 ({len(inactive_edges)})"),
    ]
    ax.legend(handles=legend_els, fontsize=6.5, loc="lower left",
              framealpha=0.9, edgecolor="#CBD5E0", fancybox=True)


# ─── Combined overview figure ─────────────────────────────────────────────────

def generate_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor(PALETTE["bg"])

    draw_static(axes[0])
    draw_climatological(axes[1])
    draw_soft_dynamic(axes[2])

    fig.suptitle("ST-GNN Graph Mode 비교", fontsize=14, fontweight="bold",
                 y=1.01, color="#1A202C")

    # Mode labels (A, B, C)
    for ax, label in zip(axes, ["(A)", "(B)", "(C)"]):
        ax.text(-0.04, 1.07, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", color="#2D3748")

    plt.tight_layout(pad=1.2)
    out = "appendix/fig_graph_modes_overview.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


# ─── Individual figures ───────────────────────────────────────────────────────

def generate_individual():
    configs = [
        ("static",         draw_static,         "Static Graph",         "fig_static_graph.png"),
        ("climatological", draw_climatological,  "Climatological Graph", "fig_climatological_graph.png"),
        ("soft_dynamic",   draw_soft_dynamic,    "Soft-Dynamic Graph",   "fig_soft_dynamic_graph.png"),
    ]
    for mode, draw_fn, title, fname in configs:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor(PALETTE["bg"])
        draw_fn(ax)
        plt.tight_layout(pad=1.0)
        out = f"appendix/{fname}"
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
        plt.close()


if __name__ == "__main__":
    generate_overview()
    generate_individual()
    print("Done.")

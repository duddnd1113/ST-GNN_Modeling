"""
Interactive HTML visualization of the ST-GNN graph on a Seoul map.

Graph structure is scenario-independent (same stations, same wind data),
so no scenario argument is needed.

Supports all three graph modes:
  static         : all edges within threshold, colored by distance (undirected look)
  climatological : pre-filtered directed edges by mean wind, colored by distance + arrows
  soft_dynamic   : only edges where wind blows src→dst at chosen timestep + arrows

Generates:
  visualizations/{graph_mode}.html

Usage:
    python3 visualize_graph.py --graph_mode static
    python3 visualize_graph.py --graph_mode climatological
    python3 visualize_graph.py --graph_mode soft_dynamic --timestep 100
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath

from dataset import SCENARIO_DIR, SPLIT_INFO_PATH, _ID_COLS
from graph_builder import (
    build_static_graph,
    build_climatological_graph,
    get_active_edges,
)


THRESHOLD_KM  = 10.0
GRAPH_DIR     = Path("graphs")
OUTPUT_BASE   = Path("visualizations")
BASE_SCENARIO = "S1_transport_pm10"   # 좌표·바람 추출용 (그래프는 시나리오 무관)


# ── Colour helpers ────────────────────────────────────────────────────────────

def lerp_color(t: float, c0: tuple, c1: tuple) -> str:
    r = int(c0[0] + t * (c1[0] - c0[0]))
    g = int(c0[1] + t * (c1[1] - c0[1]))
    b = int(c0[2] + t * (c1[2] - c0[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def distance_color(norm_dist: float) -> str:
    """green → yellow → red as distance increases."""
    t = max(0.0, min(1.0, norm_dist))
    if t < 0.5:
        return lerp_color(t * 2, (50, 180, 80), (230, 200, 0))
    return lerp_color((t - 0.5) * 2, (230, 200, 0), (200, 60, 60))


def alignment_color(alignment: float) -> str:
    """blue (weak) → orange → red (strong) for positive alignment."""
    t = max(0.0, min(1.0, alignment / 5.0))   # normalise to ~[0,1]
    return lerp_color(t, (100, 160, 220), (210, 60, 40))


# ── Map helpers ───────────────────────────────────────────────────────────────

def make_base_map():
    return folium.Map(
        location=[37.545, 127.0],
        zoom_start=11,
        tiles="CartoDB positron",
    )


def add_nodes(m, stations, coords):
    group = folium.FeatureGroup(name="Stations")
    for i, (name, (lat, lon)) in enumerate(zip(stations, coords)):
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#1a1a2e",
            fill=True,
            fill_color="#4a90d9",
            fill_opacity=0.9,
            weight=1.5,
            tooltip=f"<b>{name}</b><br>({lat:.4f}, {lon:.4f})",
            popup=folium.Popup(
                f"<b>{name}</b><br>위도: {lat:.4f}<br>경도: {lon:.4f}<br>ID: {i}",
                max_width=200,
            ),
        ).add_to(group)
    group.add_to(m)


def add_directed_edge(group, lat_s, lon_s, lat_d, lon_d, color, width, tooltip_text):
    """Draw a directed edge with an arrow along the line."""
    line = folium.PolyLine(
        locations=[[lat_s, lon_s], [lat_d, lon_d]],
        color=color,
        weight=width,
        opacity=0.75,
        tooltip=tooltip_text,
    )
    line.add_to(group)
    PolyLineTextPath(
        line,
        "        ▶",
        repeat=False,
        offset=60,
        attributes={"fill": color, "font-size": "14px", "font-weight": "bold"},
    ).add_to(group)


def add_title(m, text):
    html = f"""
    <div style="position:fixed; top:15px; left:50%; transform:translateX(-50%);
                z-index:1000; background:white; padding:8px 18px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2); font-family:sans-serif; font-size:14px;">
        {text}
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


def add_legend(m, html_body):
    html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2); font-family:sans-serif; font-size:13px;">
        {html_body}
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_graph(graph_mode: str, coords, train_nodes=None):
    """Load pre-built graph from graphs/ or build on the fly."""
    graph_key = "static" if graph_mode == "soft_dynamic" else graph_mode
    graph_path = GRAPH_DIR / graph_key
    files = {k: graph_path / f"{k}.npy" for k in ("edge_index", "static_attr", "edge_bearings")}

    if all(p.exists() for p in files.values()):
        return (
            np.load(files["edge_index"]),
            np.load(files["static_attr"]),
            np.load(files["edge_bearings"]),
        )

    print(f"  Pre-built graph not found → building on the fly (run prepare_graphs.py to cache)")
    if graph_mode == "climatological":
        return build_climatological_graph(coords, train_nodes, THRESHOLD_KM)
    return build_static_graph(coords, THRESHOLD_KM)


def load_graph_data(timestep: int):
    """Load station metadata and one timestep of raw node features from BASE_SCENARIO.

    Graph structure is scenario-independent, so we always read from BASE_SCENARIO.
    """
    import pickle
    csv_path = SCENARIO_DIR / f"{BASE_SCENARIO}.csv"
    df = pd.read_csv(csv_path)

    stations = sorted(df["측정소명"].unique())
    N = len(stations)
    coord_df = df.drop_duplicates("측정소명").set_index("측정소명")[["위도", "경도"]]
    coords = [
        (float(coord_df.loc[s, "위도"]), float(coord_df.loc[s, "경도"]))
        for s in stations
    ]

    feat_cols = [c for c in df.columns if c not in _ID_COLS and not c.endswith("_mask")]
    times = sorted(df["time"].unique())
    t_str = times[timestep % len(times)]

    df_t = df[df["time"] == t_str].sort_values("측정소명")
    feat_t = df_t[feat_cols].values.astype(np.float32)   # [N, F]

    # Train nodes for climatological graph fallback
    with open(SPLIT_INFO_PATH, "rb") as f:
        split_info = pickle.load(f)
    train_set = set(split_info["train_times"])
    df_train = df[df["time"].isin(train_set)].sort_values(["time", "측정소명"])
    T_train = df_train["time"].nunique()
    train_nodes = df_train[feat_cols].values.reshape(T_train, N, len(feat_cols)).astype(np.float32)

    return stations, coords, feat_t, t_str, train_nodes


# ── Mode-specific visualizations ─────────────────────────────────────────────

def build_static_map(stations, coords, edge_index, static_attr):
    m = make_base_map()
    E = edge_index.shape[1]

    group = folium.FeatureGroup(name="Edges (distance)")
    for e in range(E):
        si, di = edge_index[0, e], edge_index[1, e]
        norm_dist = float(static_attr[e, 0])
        color = distance_color(norm_dist)
        width = max(1.0, 4.0 * (1.0 - norm_dist))
        dist_km = norm_dist * THRESHOLD_KM

        folium.PolyLine(
            locations=[list(coords[si]), list(coords[di])],
            color=color,
            weight=width,
            opacity=0.6,
            tooltip=f"{stations[si]} → {stations[di]}<br>거리: {dist_km:.1f} km",
        ).add_to(group)
    group.add_to(m)

    add_nodes(m, stations, coords)
    add_title(m, f"<b>Static Graph</b> &nbsp;|&nbsp; E={E} &nbsp;|&nbsp; threshold={THRESHOLD_KM} km")
    add_legend(m, """
        <b>엣지 색상 — 거리</b><br>
        <span style="color:#32b450">●</span> 가까움 (0 km)<br>
        <span style="color:#e6c800">●</span> 중간<br>
        <span style="color:#c83c3c">●</span> 멀음""")
    folium.LayerControl().add_to(m)
    return m


def build_climatological_map(stations, coords, edge_index, static_attr):
    m = make_base_map()
    E = edge_index.shape[1]

    group = folium.FeatureGroup(name="Edges (climatological, directed)")
    for e in range(E):
        si, di = edge_index[0, e], edge_index[1, e]
        norm_dist = float(static_attr[e, 0])
        color = distance_color(norm_dist)
        width = max(1.5, 3.5 * (1.0 - norm_dist))
        dist_km = norm_dist * THRESHOLD_KM

        add_directed_edge(
            group,
            *coords[si], *coords[di],
            color=color,
            width=width,
            tooltip_text=f"{stations[si]} → {stations[di]}<br>거리: {dist_km:.1f} km<br>↑ 지배 풍향 방향",
        )
    group.add_to(m)

    add_nodes(m, stations, coords)
    add_title(m, f"<b>Climatological Graph</b> &nbsp;|&nbsp; E={E} &nbsp;|&nbsp; 평균 풍향 기준 단방향")
    add_legend(m, """
        <b>엣지 색상 — 거리</b><br>
        <span style="color:#32b450">●</span> 가까움<br>
        <span style="color:#c83c3c">●</span> 멀음<br>
        <br><b>▶</b> 지배 풍향 방향""")
    folium.LayerControl().add_to(m)
    return m


def build_soft_dynamic_map(stations, coords, edge_index, static_attr, edge_bearings, feat_t, t_str):
    m = make_base_map()

    ei_a, _, _, dyn_a = get_active_edges(
        edge_index, static_attr, edge_bearings, feat_t
    )
    E_active = ei_a.shape[1] if ei_a.ndim == 2 and ei_a.shape[1] > 0 else 0

    group = folium.FeatureGroup(name="Active edges (wind → dst)")
    for e in range(E_active):
        si, di = ei_a[0, e], ei_a[1, e]
        alignment   = float(dyn_a[e, 0])
        eff_wind    = float(dyn_a[e, 1])
        color = alignment_color(alignment)
        width = max(1.5, 1.5 + eff_wind * 6.0)

        uu = float(feat_t[si, 2])
        vv = float(feat_t[si, 3])
        ws = math.sqrt(uu ** 2 + vv ** 2)

        add_directed_edge(
            group,
            *coords[si], *coords[di],
            color=color,
            width=width,
            tooltip_text=(
                f"{stations[si]} → {stations[di]}<br>"
                f"wind alignment: {alignment:+.2f}<br>"
                f"effective wind: {eff_wind:.2f}<br>"
                f"풍속: {ws:.1f} m/s"
            ),
        )
    group.add_to(m)

    add_nodes(m, stations, coords)
    add_title(m, f"<b>Soft-Dynamic Graph</b> &nbsp;|&nbsp; {t_str} &nbsp;|&nbsp; 활성 엣지: {E_active}")
    add_legend(m, """
        <b>엣지 색상 — Wind Alignment</b><br>
        <span style="color:#64a0dc">●</span> 약한 순풍<br>
        <span style="color:#d23c28">●</span> 강한 순풍<br>
        <br><b>엣지 굵기</b> — effective wind 크기<br>
        <br><b>▶</b> 바람 전달 방향<br>
        역풍 엣지는 표시 안 함""")
    folium.LayerControl().add_to(m)
    return m


# ── Entry point ───────────────────────────────────────────────────────────────

def main(graph_mode: str, timestep: int):
    print(f"Graph mode: {graph_mode}  |  Timestep: {timestep}")

    stations, coords, feat_t, t_str, train_nodes = load_graph_data(timestep)
    print(f"  Stations: {len(stations)}  |  Snapshot: {t_str}")

    edge_index, static_attr, edge_bearings = load_graph(graph_mode, coords, train_nodes)
    print(f"  Graph edges: {edge_index.shape[1]}")

    if graph_mode == "static":
        m = build_static_map(stations, coords, edge_index, static_attr)
    elif graph_mode == "climatological":
        m = build_climatological_map(stations, coords, edge_index, static_attr)
    else:  # soft_dynamic
        m = build_soft_dynamic_map(stations, coords, edge_index, static_attr, edge_bearings, feat_t, t_str)

    out_path = OUTPUT_BASE / f"{graph_mode}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"  → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_mode", type=str, default="static",
                        choices=("static", "climatological", "soft_dynamic"))
    parser.add_argument("--timestep",   type=int, default=0,
                        help="Index into full time series (for soft_dynamic snapshot)")
    args = parser.parse_args()
    main(args.graph_mode, args.timestep)

"""
Static graph construction and edge feature computation for Seoul PM2.5 ST-GNN.

Provides:
  - Distance-threshold based sparse directed graph from station coordinates.
  - Static edge features: normalized distance + directional (sin/cos bearing).
  - Dynamic edge features per time step: wind alignment and effective wind.
  - Helper to combine static + dynamic into full 5-D edge feature tensor.
"""

from typing import Tuple, List

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Compute Haversine great-circle distance between two lat/lon points.

    Args:
        lat1, lon1: Coordinates of point 1 in decimal degrees.
        lat2, lon2: Coordinates of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Compute initial compass bearing from point 1 → point 2.

    Args:
        lat1, lon1: Origin coordinates in decimal degrees.
        lat2, lon2: Destination coordinates in decimal degrees.

    Returns:
        Bearing in degrees in [0, 360), measured clockwise from North.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2)
         - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_static_graph(
    coords: List[Tuple[float, float]],
    threshold_km: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a directed sparse graph from station coordinates.

    Two stations are connected if their Haversine distance is below
    *threshold_km*. Both i→j and j→i edges are created (undirected
    stored as bidirectional directed edges).

    Static edge features for a directed edge src → dst (3-D):
        [distance_norm, sin(bearing_src→dst), cos(bearing_src→dst)]

    where distance_norm = distance / threshold_km ∈ [0, 1].

    Args:
        coords: List of N (lat, lon) tuples.
        threshold_km: Distance cutoff in km (default 20 km).

    Returns:
        edge_index: np.ndarray int64 [2, E] — (src, dst) rows.
        static_edge_attr: np.ndarray float32 [E, 3].
        edge_bearings: np.ndarray float32 [E] — bearing (degrees) src→dst.
    """
    N = len(coords)
    edges: List[List[int]] = []
    attrs: List[List[float]] = []
    bearings: List[float] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            if dist >= threshold_km:
                continue

            b_deg = compute_bearing(lat1, lon1, lat2, lon2)
            b_rad = np.radians(b_deg)
            norm_dist = dist / threshold_km

            edges.append([i, j])
            attrs.append([norm_dist, float(np.sin(b_rad)), float(np.cos(b_rad))])
            bearings.append(b_deg)

    edge_index = np.array(edges, dtype=np.int64).T          # [2, E]
    static_edge_attr = np.array(attrs, dtype=np.float32)    # [E, 3]
    edge_bearings = np.array(bearings, dtype=np.float32)    # [E]
    return edge_index, static_edge_attr, edge_bearings


# ──────────────────────────────────────────────────────────────────────────────
# Climatological directed graph
# ──────────────────────────────────────────────────────────────────────────────

def build_climatological_graph(
    coords: List[Tuple[float, float]],
    train_nodes: np.ndarray,
    threshold_km: float = 10.0,
    uu_idx: int = 2,
    vv_idx: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a directed graph using distance threshold + dominant wind direction.

    For each candidate edge i→j (within threshold_km), compute the mean wind
    alignment over the training period at station i. Only keep the edge if the
    dominant (time-averaged) wind blows from i toward j.

    Args:
        coords: List of N (lat, lon) tuples.
        train_nodes: np.ndarray [T_train, N, F] — raw (unnormalised) node features.
        threshold_km: Distance cutoff in km.
        uu_idx: Column index of east-west wind component.
        vv_idx: Column index of north-south wind component.

    Returns:
        edge_index, static_edge_attr, edge_bearings — same format as build_static_graph.
    """
    mean_uu = train_nodes[:, :, uu_idx].mean(axis=0)  # [N]
    mean_vv = train_nodes[:, :, vv_idx].mean(axis=0)  # [N]

    N = len(coords)
    edges: List[List[int]] = []
    attrs: List[List[float]] = []
    bearings: List[float] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            if dist >= threshold_km:
                continue

            b_deg = compute_bearing(lat1, lon1, lat2, lon2)
            b_rad = np.radians(b_deg)

            # Keep i→j only if mean wind at i blows toward j
            alignment = float(mean_uu[i] * np.sin(b_rad) + mean_vv[i] * np.cos(b_rad))
            if alignment <= 0:
                continue

            norm_dist = dist / threshold_km
            edges.append([i, j])
            attrs.append([norm_dist, float(np.sin(b_rad)), float(np.cos(b_rad))])
            bearings.append(b_deg)

    edge_index = np.array(edges, dtype=np.int64).T
    static_edge_attr = np.array(attrs, dtype=np.float32)
    edge_bearings = np.array(bearings, dtype=np.float32)
    return edge_index, static_edge_attr, edge_bearings


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic edge features
# ──────────────────────────────────────────────────────────────────────────────

def compute_dynamic_edge_features(
    edge_index: np.ndarray,
    node_features_t: np.ndarray,
    bearings: np.ndarray,
    uu_idx: int = 2,
    vv_idx: int = 3,
) -> np.ndarray:
    """Compute wind-based dynamic edge features for a single time step.

    For directed edge src → dst, the two dynamic features capture how
    effectively the wind at *src* carries pollution toward *dst*:

        wind_alignment = uu_src * sin(bearing) + vv_src * cos(bearing)
                         ∈ [−1, 1] × wind_speed  (dot product of wind vector
                         with unit bearing vector)

        effective_wind = max(0, wind_alignment) / 10.0
                         ∈ [0, 1]  (normalised by max wind speed 10 m/s)

    Node feature channel layout assumed by default (index):
        0: 풍향_10m, 1: 풍속_10m, 2: uu (동서), 3: vv (남북), 4: PM10, ...

    Args:
        edge_index: np.ndarray int64 [2, E].
        node_features_t: np.ndarray float32 [N, F] at time t.
        bearings: np.ndarray float32 [E] — bearing in degrees, src→dst.
        uu_idx: column index of east-west wind component (default 2).
        vv_idx: column index of north-south wind component (default 3).

    Returns:
        dynamic_edge_attr: np.ndarray float32 [E, 2].
    """
    src = edge_index[0]                              # [E]

    uu_src = node_features_t[src, uu_idx]            # [E] east-west wind (m/s)
    vv_src = node_features_t[src, vv_idx]            # [E] north-south wind (m/s)

    sin_b = np.sin(np.radians(bearings))             # [E]
    cos_b = np.cos(np.radians(bearings))             # [E]

    wind_alignment = uu_src * sin_b + vv_src * cos_b             # [E] ∈ [-ws, ws]
    effective_wind = np.maximum(0.0, wind_alignment) / 10.0      # [E] ∈ [0, 1]

    return np.stack([wind_alignment, effective_wind], axis=-1).astype(np.float32)


def compute_all_dynamic_edge_features(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    edge_bearings: np.ndarray,
    uu_idx: int = 2,
    vv_idx: int = 3,
) -> np.ndarray:
    """Vectorised version of compute_dynamic_edge_features over all T steps.

    Args:
        edge_index: np.ndarray int64 [2, E].
        node_features: np.ndarray float32 [T, N, F].
        edge_bearings: np.ndarray float32 [E] — bearing src→dst, degrees.
        uu_idx: column index of east-west wind component (default 2).
        vv_idx: column index of north-south wind component (default 3).

    Returns:
        dynamic_all: np.ndarray float32 [T, E, 2].
    """
    src = edge_index[0]                                      # [E]

    uu_src = node_features[:, src, uu_idx]                   # [T, E]
    vv_src = node_features[:, src, vv_idx]                   # [T, E]

    sin_b = np.sin(np.radians(edge_bearings))                # [E]
    cos_b = np.cos(np.radians(edge_bearings))                # [E]

    wind_alignment = uu_src * sin_b[None, :] + vv_src * cos_b[None, :]   # [T, E]
    effective_wind = np.maximum(0.0, wind_alignment) / 10.0               # [T, E]

    dynamic_all = np.stack([wind_alignment, effective_wind], axis=-1)     # [T, E, 2]
    return dynamic_all.astype(np.float32)


def get_full_edge_features(
    static_attr: np.ndarray,
    dynamic_attr: np.ndarray,
) -> np.ndarray:
    """Concatenate static and dynamic edge features into a 5-D vector.

    Supports both single-step ([E, 3] + [E, 2]) and multi-step
    ([T, E, 3] + [T, E, 2]) inputs.

    Args:
        static_attr: [E, 3] or [T, E, 3].
        dynamic_attr: [E, 2] or [T, E, 2].

    Returns:
        full_attr: [E, 5] or [T, E, 5].
    """
    return np.concatenate([static_attr, dynamic_attr], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Active edge filter (for soft_dynamic visualization)
# ──────────────────────────────────────────────────────────────────────────────

def get_active_edges(
    edge_index: np.ndarray,
    static_attr: np.ndarray,
    edge_bearings: np.ndarray,
    node_features_t: np.ndarray,
    uu_idx: int = 2,
    vv_idx: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return only edges where wind blows from src toward dst at time t.

    Computes wind alignment for each edge and keeps those with alignment > 0,
    matching the soft_dynamic masking logic used during training.

    Args:
        edge_index:      [2, E] int64
        static_attr:     [E, 3] float32
        edge_bearings:   [E]    float32 — degrees, src→dst
        node_features_t: [N, F] float32 — raw node features at one timestep
        uu_idx:          column index of east-west wind component
        vv_idx:          column index of north-south wind component

    Returns:
        edge_index_active:    [2, E'] — filtered edges
        static_attr_active:   [E', 3]
        edge_bearings_active: [E']
        dyn_attr_active:      [E', 2] — [wind_alignment, effective_wind]
    """
    dyn_attr = compute_dynamic_edge_features(
        edge_index, node_features_t, edge_bearings, uu_idx, vv_idx
    )                                                  # [E, 2]
    active = dyn_attr[:, 0] > 0                        # wind_alignment > 0
    return (
        edge_index[:, active],
        static_attr[active],
        edge_bearings[active],
        dyn_attr[active],
    )

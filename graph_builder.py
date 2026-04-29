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
# Dynamic edge features
# ──────────────────────────────────────────────────────────────────────────────

def compute_dynamic_edge_features(
    edge_index: np.ndarray,
    node_features_t: np.ndarray,
    bearings: np.ndarray,
) -> np.ndarray:
    """Compute wind-based dynamic edge features for a single time step.

    For directed edge src → dst, the two dynamic features capture how
    effectively the wind at *src* carries pollution toward *dst*:

        wind_alignment  = cos(wind_direction_src − bearing_src→dst)
                          ∈ [−1, 1]  (1 = wind blowing directly toward dst)

        effective_wind  = wind_speed_src × max(0, wind_alignment) / 10.0
                          ∈ [0, 1]   (normalised by max wind speed 10 m/s)

    Node feature channel layout (index):
        0: pm25, 1: pm10, 2: wind_speed, 3: wind_direction,
        4: temperature, 5: humidity

    Args:
        edge_index: np.ndarray int64 [2, E].
        node_features_t: np.ndarray float32 [N, 6] at time t.
        bearings: np.ndarray float32 [E] — bearing in degrees, src→dst.

    Returns:
        dynamic_edge_attr: np.ndarray float32 [E, 2].
    """
    src = edge_index[0]                          # [E]

    wind_speed_src = node_features_t[src, 2]    # [E]
    wind_dir_src = node_features_t[src, 3]      # [E], degrees

    bearings_rad = np.radians(bearings)          # [E]
    wind_dir_rad = np.radians(wind_dir_src)      # [E]

    wind_alignment = np.cos(wind_dir_rad - bearings_rad)              # [E] ∈ [−1,1]
    effective_wind = wind_speed_src * np.maximum(0.0, wind_alignment) / 10.0  # [E] ∈ [0,1]

    return np.stack([wind_alignment, effective_wind], axis=-1).astype(np.float32)


def compute_all_dynamic_edge_features(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    edge_bearings: np.ndarray,
) -> np.ndarray:
    """Vectorised version of compute_dynamic_edge_features over all T steps.

    Args:
        edge_index: np.ndarray int64 [2, E].
        node_features: np.ndarray float32 [T, N, 6].
        edge_bearings: np.ndarray float32 [E] — bearing src→dst, degrees.

    Returns:
        dynamic_all: np.ndarray float32 [T, E, 2].
    """
    src = edge_index[0]                                     # [E]

    wind_speed_src = node_features[:, src, 2]               # [T, E]
    wind_dir_src = node_features[:, src, 3]                 # [T, E], degrees

    bearings_rad = np.radians(edge_bearings)                # [E]
    wind_dir_rad = np.radians(wind_dir_src)                 # [T, E]

    wind_alignment = np.cos(wind_dir_rad - bearings_rad[None, :])              # [T, E]
    effective_wind = (wind_speed_src
                      * np.maximum(0.0, wind_alignment) / 10.0)               # [T, E]

    dynamic_all = np.stack([wind_alignment, effective_wind], axis=-1)          # [T, E, 2]
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

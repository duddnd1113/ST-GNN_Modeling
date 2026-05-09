"""
Orienteering Problem (OP) with spacing constraint — GRASP metaheuristic.

Pipeline:
    full scored grid
    -> top-M 후보 풀 필터링 (score 기준)
    -> 거리 행렬 사전 계산 (O(1) lookup)
    -> GRASP: score / marginal_dist 기반 확률적 노드 선택
    -> node-swap + node-insert local search  (스크리닝은 NN, 수락 시만 2-opt)
    -> best route 출력
"""

import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    import contextily as ctx
    from pyproj import Transformer
    HAS_MAP = True
except ImportError:
    HAS_MAP = False


# ──────────────────────────────────────────────────────────
# 1. Synthetic snapshot
# ──────────────────────────────────────────────────────────

def generate_synthetic_snapshot(args, device):
    """원본 코드와 동일한 인터페이스: (df, coords, scores) 반환.
    coords: (1, N, 2) torch tensor
    scores: (1, N)   torch tensor
    """
    torch.manual_seed(args.seed)
    grid_size = int(math.sqrt(args.total_grid_size))
    if grid_size * grid_size != args.total_grid_size:
        raise ValueError("total_grid_size must be a perfect square, e.g. 10000, 22500")

    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).unsqueeze(0)  # (1, N, 2)

    if args.score_mode == "random":
        scores = torch.rand(1, args.total_grid_size, device=device)             # (1, N)
    else:
        center = torch.tensor([args.hotspot_x, args.hotspot_y], device=device)
        dist = ((coords - center) ** 2).sum(dim=2).sqrt()
        scores = (
            torch.exp(-args.hotspot_strength * dist)
            + args.noise * torch.rand(1, args.total_grid_size, device=device)
        )

    coords_np = coords[0].cpu().numpy()   # (N, 2)
    scores_np = scores[0].cpu().numpy()   # (N,)
    df = pd.DataFrame({
        "grid_id": np.arange(args.total_grid_size),
        "x": coords_np[:, 0],
        "y": coords_np[:, 1],
        "score": scores_np,
    })
    return df, coords, scores


# ──────────────────────────────────────────────────────────
# 2. 거리 행렬 사전 계산
# ──────────────────────────────────────────────────────────

def build_dist_matrix(candidates, depot):
    """
    (M+1) x (M+1) 거리 행렬.  마지막 인덱스 M = depot.
    이후 모든 거리 계산은 D[i, j] 로 O(1).
    """
    pts = list(candidates) + [depot]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    return np.sqrt(dx * dx + dy * dy)


# ──────────────────────────────────────────────────────────
# 3. 라우팅 유틸  (거리 행렬 기반)
# ──────────────────────────────────────────────────────────

def route_dist_dm(order, D, depot_idx):
    if not order:
        return 0.0
    d = D[depot_idx, order[0]]
    for i in range(len(order) - 1):
        d += D[order[i], order[i + 1]]
    d += D[order[-1], depot_idx]
    return d


def nn_order_dm(sel_indices, D, depot_idx):
    unvisited = list(sel_indices)
    order, cur = [], depot_idx
    while unvisited:
        nxt = min(unvisited, key=lambda i: D[cur, i])
        order.append(nxt)
        cur = nxt
        unvisited.remove(nxt)
    return order


def two_opt_dm(order, D, depot_idx):
    best = list(order)
    best_d = route_dist_dm(best, D, depot_idx)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                new = best[:i] + list(reversed(best[i:j])) + best[j:]
                d = route_dist_dm(new, D, depot_idx)
                if d < best_d - 1e-10:
                    best, best_d, improved = new, d, True
    return best, best_d


def reorder_dm(sel_indices, D, depot_idx, do_two_opt=True):
    """NN (+선택적 2-opt) 순서 + 거리."""
    if not sel_indices:
        return [], 0.0
    order = nn_order_dm(sel_indices, D, depot_idx)
    if do_two_opt:
        order, d = two_opt_dm(order, D, depot_idx)
    else:
        d = route_dist_dm(order, D, depot_idx)
    return order, d


# ──────────────────────────────────────────────────────────
# 4. GRASP Construction
# ──────────────────────────────────────────────────────────

def grasp_construct(scores, D, depot_idx, distance_budget, min_spacing, alpha, rng):
    """거리 행렬 D 기반 GRASP construction."""
    M = len(scores)
    selected = []
    current_dist = 0.0
    available = list(range(M))

    while available:
        last = selected[-1] if selected else depot_idx

        feasible = []
        for i in available:
            # spacing: 기존 선택 노드와 너무 가까우면 skip
            if selected and D[selected, i].min() < min_spacing:
                continue

            marginal = D[last, i] + D[i, depot_idx] - D[last, depot_idx]
            if current_dist + marginal > distance_budget + 1e-9:
                continue

            ratio = scores[i] / (marginal + 1e-8)
            feasible.append((i, ratio, marginal))

        if not feasible:
            break

        feasible.sort(key=lambda x: -x[1])
        rcl_size = max(1, math.ceil(alpha * len(feasible)))
        chosen_idx, _, chosen_marginal = feasible[rng.integers(rcl_size)]

        current_dist += chosen_marginal
        selected.append(chosen_idx)
        available.remove(chosen_idx)

    return selected


# ──────────────────────────────────────────────────────────
# 5. Local Search  (속도 최적화 버전)
# ──────────────────────────────────────────────────────────

def local_search(selected, scores, D, depot_idx, distance_budget, min_spacing, swap_pool):
    """
    속도 최적화 포인트:
      - 스크리닝: NN 거리만  (2-opt 없음)
      - 수락 시만: 2-opt 한 번
      - swap 후보 풀: score 상위 swap_pool개로 제한
    """
    M = len(scores)
    sel = list(selected)
    sel, cur_dist = reorder_dm(sel, D, depot_idx, do_two_opt=True)
    cur_score = float(scores[sel].sum()) if sel else 0.0
    all_idx = set(range(M))

    improved = True
    while improved:
        improved = False
        sel_set = set(sel)

        # node-swap: 미방문 중 score 상위 swap_pool개만
        unvisited_ranked = sorted(all_idx - sel_set, key=lambda x: -scores[x])[:swap_pool]

        for pos in range(len(sel)):
            for add_idx in unvisited_ranked:
                others = [sel[k] for k in range(len(sel)) if k != pos]

                # spacing 체크
                if others and D[others, add_idx].min() < min_spacing:
                    continue

                new_sel = others + [add_idx]
                # 스크리닝: NN만 (빠름)
                _, new_dist = reorder_dm(new_sel, D, depot_idx, do_two_opt=False)
                if new_dist > distance_budget + 1e-9:
                    continue

                new_score = float(scores[new_sel].sum())
                if new_score > cur_score + 1e-10:
                    # 수락: 2-opt 한 번만
                    new_sel, new_dist = reorder_dm(new_sel, D, depot_idx, do_two_opt=True)
                    if new_dist <= distance_budget + 1e-9:
                        sel = new_sel
                        cur_dist = new_dist
                        cur_score = new_score
                        improved = True
                        break

            if improved:
                break

        if improved:
            continue

        # node-insert
        sel_set = set(sel)
        unvisited_ranked = sorted(all_idx - sel_set, key=lambda x: -scores[x])[:swap_pool]

        for add_idx in unvisited_ranked:
            if sel and D[sel, add_idx].min() < min_spacing:
                continue

            new_sel = sel + [add_idx]
            _, new_dist = reorder_dm(new_sel, D, depot_idx, do_two_opt=False)
            if new_dist > distance_budget + 1e-9:
                continue

            new_sel, new_dist = reorder_dm(new_sel, D, depot_idx, do_two_opt=True)
            if new_dist <= distance_budget + 1e-9:
                sel = new_sel
                cur_dist = new_dist
                cur_score = float(scores[sel].sum())
                improved = True
                break

    return sel, cur_dist, cur_score


# ──────────────────────────────────────────────────────────
# 6. Multi-start GRASP
# ──────────────────────────────────────────────────────────

def run_grasp(candidates, scores, depot, distance_budget, min_spacing,
              alpha, n_restarts, swap_pool, seed):

    D = build_dist_matrix(candidates, depot)
    depot_idx = len(candidates)
    rng = np.random.default_rng(seed)

    best_sel, best_score, best_dist = [], -1.0, 0.0

    for restart in range(n_restarts):
        sel = grasp_construct(scores, D, depot_idx, distance_budget, min_spacing, alpha, rng)
        sel, dist, score = local_search(sel, scores, D, depot_idx, distance_budget, min_spacing, swap_pool)

        if score > best_score:
            best_sel, best_score, best_dist = sel, score, dist
            logging.info(
                "restart %d/%d | nodes=%d | score=%.4f | dist=%.4f",
                restart + 1, n_restarts, len(sel), score, dist,
            )

    return best_sel, best_dist, best_score


# ──────────────────────────────────────────────────────────
# 7. 출력
# ──────────────────────────────────────────────────────────

def save_route_csv(df, route_global_ids, best_dist, best_score, args):
    rows = []
    for step, gid in enumerate(route_global_ids):
        row = df.iloc[gid].to_dict()
        row["route_order"] = step
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_route, index=False)
    logging.info(
        "Saved route CSV: %s  (nodes=%d, dist=%.4f, score=%.4f)",
        args.output_route, len(route_global_ids), best_dist, best_score,
    )


def _norm_to_geo(xn, yn, args):
    """0-1 정규화 좌표 → (lon, lat) 변환."""
    lon = args.bbox_lon_min + xn * (args.bbox_lon_max - args.bbox_lon_min)
    lat = args.bbox_lat_min + yn * (args.bbox_lat_max - args.bbox_lat_min)
    return lon, lat


def _to_merc(lons, lats):
    """lon/lat (EPSG:4326) → Web Mercator (EPSG:3857)."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return tr.transform(lons, lats)


def visualize(df, candidate_global_ids, route_global_ids, depot, best_dist, best_score, args):
    use_map = args.use_map and HAS_MAP

    if args.use_map and not HAS_MAP:
        logging.warning("contextily / pyproj 미설치 — 지도 없이 시각화합니다.")
        logging.warning("pip install contextily pyproj")

    xn, yn, s = df["x"].values, df["y"].values, df["score"].values

    # ── 좌표 준비 ──────────────────────────────────────────
    if use_map:
        lons, lats = _norm_to_geo(xn, yn, args)
        px_all, py_all = _to_merc(lons, lats)
    else:
        px_all, py_all = xn, yn

    depot_lon, depot_lat = _norm_to_geo(depot[0], depot[1], args)
    depot_mx, depot_my   = (_to_merc([depot_lon], [depot_lat]) if use_map
                            else ([depot[0]], [depot[1]]))

    fig, ax = plt.subplots(figsize=(11, 9))

    # ── 그리드 히트맵 ──────────────────────────────────────
    sc = ax.scatter(px_all, py_all, c=s, cmap="Reds",
                    s=args.grid_marker_size, alpha=0.35, zorder=2)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("PM10 score")

    if route_global_ids:
        r_xn = xn[route_global_ids]
        r_yn = yn[route_global_ids]

        if use_map:
            r_lons, r_lats = _norm_to_geo(r_xn, r_yn, args)
            rx, ry = _to_merc(r_lons, r_lats)
            rx, ry = np.array(rx), np.array(ry)
        else:
            rx, ry = r_xn, r_yn

        # 자동 줌인 (Mercator or normalized 공통)
        margin_ratio = 0.08
        dx = max(rx.max() - rx.min(), 1e-6)
        dy = max(ry.max() - ry.min(), 1e-6)
        margin_x = max(dx * margin_ratio, dy * margin_ratio)
        margin_y = margin_x

        dxp, dyp = depot_mx[0], depot_my[0]
        x_min = min(rx.min(), dxp) - margin_x
        x_max = max(rx.max(), dxp) + margin_x
        y_min = min(ry.min(), dyp) - margin_y
        y_max = max(ry.max(), dyp) + margin_y

        # 줌인 영역 후보 풀
        if use_map:
            c_lons, c_lats = _norm_to_geo(xn[candidate_global_ids], yn[candidate_global_ids], args)
            cx_all, cy_all = _to_merc(c_lons, c_lats)
            cx_all, cy_all = np.array(cx_all), np.array(cy_all)
        else:
            cx_all = xn[candidate_global_ids]
            cy_all = yn[candidate_global_ids]

        in_view_mask = (
            (cx_all >= x_min) & (cx_all <= x_max) &
            (cy_all >= y_min) & (cy_all <= y_max)
        )
        if in_view_mask.any():
            ax.scatter(
                cx_all[in_view_mask], cy_all[in_view_mask],
                s=args.topk_marker_size, marker="o",
                facecolors="none", edgecolors="steelblue",
                linewidths=1.0, alpha=0.7, zorder=3,
                label=f"Candidate pool (top-{args.candidate_pool}, in view)",
            )

        ax.scatter(
            rx, ry,
            s=args.topk_marker_size * 1.6, marker="o",
            c=list(range(len(route_global_ids))), cmap="RdYlGn",
            edgecolors="black", linewidths=1.4, zorder=5,
            label=f"Selected nodes ({len(route_global_ids)})",
        )

        rpx = [dxp] + list(rx) + [dxp]
        rpy = [dyp] + list(ry) + [dyp]
        ax.plot(rpx, rpy, color="royalblue", linewidth=2.5, zorder=4, label="Route")

        ax.annotate(
            "", xy=(rx[0], ry[0]), xytext=(dxp, dyp),
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.8),
            zorder=7,
        )

        for step, (xi, yi) in enumerate(zip(rx, ry)):
            ax.annotate(
                str(step + 1), (xi, yi),
                fontsize=8, ha="center", va="center",
                color="black", fontweight="bold", zorder=8,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.scatter(
        depot_mx[0], depot_my[0],
        s=args.start_marker_size, marker="*",
        color="gold", edgecolors="black", zorder=6, label="Depot",
    )

    # ── 서울 지도 배경 ────────────────────────────────────
    if use_map:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom="auto", attribution_size=7)
        ax.set_xlabel("Longitude (Mercator)")
        ax.set_ylabel("Latitude (Mercator)")
    else:
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        ax.grid(True, alpha=0.25)

    ax.set_title(
        f"OP GRASP  |  nodes={len(route_global_ids)}  |  score={best_score:.4f}  |  dist={best_dist:.4f}\n"
        f"(budget={args.distance_budget:.2f},  min_spacing={args.min_spacing:.3f},  "
        f"alpha={args.grasp_alpha},  restarts={args.n_restarts},  swap_pool={args.swap_pool})"
    )
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=200)
    plt.close(fig)
    logging.info("Saved visualization: %s", args.output_fig)


# ──────────────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OP GRASP metaheuristic — joint node selection + routing."
    )

    # 그리드 / 스냅샷
    parser.add_argument("--total-grid-size",  type=int,   default=22500)
    parser.add_argument("--score-mode",       choices=["random", "hotspot"], default="random")
    parser.add_argument("--hotspot-x",        type=float, default=0.55)
    parser.add_argument("--hotspot-y",        type=float, default=0.45)
    parser.add_argument("--hotspot-strength", type=float, default=8.0)
    parser.add_argument("--noise",            type=float, default=0.25)
    parser.add_argument("--seed",             type=int,   default=42)

    # OP 문제 파라미터
    parser.add_argument("--candidate-pool",  type=int,   default=300,
                        help="score 상위 M개를 후보 풀로 사전 필터링")
    parser.add_argument("--distance-budget", type=float, default=2.0,
                        help="최대 총 이동 거리 (0-1 정규화 그리드 기준)")
    parser.add_argument("--min-spacing",     type=float, default=0.05,
                        help="방문 노드 간 최소 거리")
    parser.add_argument("--depot-x",         type=float, default=0.5)
    parser.add_argument("--depot-y",         type=float, default=0.5)

    # GRASP 파라미터
    parser.add_argument("--grasp-alpha",  type=float, default=0.3,
                        help="RCL 크기 비율: 0=pure greedy, 1=pure random")
    parser.add_argument("--n-restarts",   type=int,   default=50)
    parser.add_argument("--swap-pool",    type=int,   default=50,
                        help="local search swap 후보 상위 K개 제한 (속도 조절)")

    # 지도 배경 (contextily)
    parser.add_argument("--use-map",       action="store_true",
                        help="서울 지도 배경 표시 (pip install contextily pyproj 필요)")
    parser.add_argument("--bbox-lon-min",  type=float, default=126.734,
                        help="서울 bbox 서쪽 경도")
    parser.add_argument("--bbox-lon-max",  type=float, default=127.269,
                        help="서울 bbox 동쪽 경도")
    parser.add_argument("--bbox-lat-min",  type=float, default=37.413,
                        help="서울 bbox 남쪽 위도")
    parser.add_argument("--bbox-lat-max",  type=float, default=37.715,
                        help="서울 bbox 북쪽 위도")

    # 출력
    parser.add_argument("--output-route",      default="op_route_output.csv")
    parser.add_argument("--output-fig",        default="op_route_visualization.png")
    parser.add_argument("--grid-marker-size",  type=float, default=8)
    parser.add_argument("--topk-marker-size",  type=float, default=90)
    parser.add_argument("--start-marker-size", type=float, default=220)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = torch.device("cpu")

    # Step 1: 스냅샷 생성  (원본과 동일 인터페이스)
    df, coords, scores = generate_synthetic_snapshot(args, device)
    # coords: (1, N, 2) | scores: (1, N)  — 실데이터 교체 시 이 두 줄만 수정

    # Step 2: 후보 풀 필터링
    top_m_idx = df["score"].nlargest(args.candidate_pool).index.tolist()
    candidates = [tuple(df.loc[i, ["x", "y"]]) for i in top_m_idx]
    scores_arr = df.loc[top_m_idx, "score"].values
    depot = (args.depot_x, args.depot_y)

    logging.info(
        "Grid=%d | candidates=%d | budget=%.3f | min_spacing=%.3f | restarts=%d | swap_pool=%d",
        args.total_grid_size, len(candidates),
        args.distance_budget, args.min_spacing, args.n_restarts, args.swap_pool,
    )

    # Step 3: GRASP
    selected_local, best_dist, best_score = run_grasp(
        candidates, scores_arr, depot,
        args.distance_budget, args.min_spacing,
        args.grasp_alpha, args.n_restarts, args.swap_pool, args.seed,
    )

    route_global_ids = [top_m_idx[i] for i in selected_local]

    # Step 4: 저장 & 시각화
    save_route_csv(df, route_global_ids, best_dist, best_score, args)
    visualize(df, top_m_idx, route_global_ids, depot, best_dist, best_score, args)

    print("best_score      :", best_score)
    print("best_dist       :", best_dist)
    print("nodes_selected  :", len(route_global_ids))
    print("saved_route_csv :", args.output_route)
    print("saved_visualization:", args.output_fig)


if __name__ == "__main__":
    main()
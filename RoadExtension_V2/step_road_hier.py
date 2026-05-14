"""
서울 도로 위계 데이터 전처리 → 격자 단위 집계

입력: 서울 도로 위계 및 GVI.gpkg (OSM 기반)
  - highway_type: trunk / primary / secondary / secondary_link / trunk_link / 기타
  - lanes_num: 차선 수
  - GVI: Green View Index (거리 녹지율)
  - length: 도로 구간 길이 (m)

처리:
  - WKB 라인스트링 파싱 → 도로 중점 좌표 추출
  - 250m 격자와 최근접 매칭 (cKDTree)
  - 격자별 집계: 최고 위계 도로 / 최대 차선 / 평균 GVI / 총 길이

출력: checkpoints/road_hier_grid.csv
  - CELL_ID, highway_rank (0-4), max_lanes, mean_gvi, total_road_length_m

실행:
    python3 step_road_hier.py
"""
import os, sys, struct, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import CKPT_DIR, GRID_CSV

GPKG_PATH       = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/서울 도로 위계 및 GVI.gpkg"
ROAD_HIER_CACHE = os.path.join(CKPT_DIR, "road_hier_grid.csv")

# highway_type → 도로 위계 점수 (높을수록 주요 도로)
HIGHWAY_RANK = {
    "trunk":          4,
    "primary":        3,
    "secondary":      2,
    "trunk_link":     1,
    "secondary_link": 1,
    "other":          0,
}


def parse_gpkg_geom(data):
    """
    GeoPackage WKB 파싱 → 도로 중점 (lon, lat) 반환.
    GPKG 포맷: 'GP' magic + version + flags + SRID(4) + envelope + WKB
    """
    if data is None or len(data) < 8 or data[0:2] != b"GP":
        return None

    flags = data[3]
    envelope_flag = (flags >> 1) & 0x07
    env_size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_flag, 0)
    wkb_offset = 8 + env_size  # 2(magic)+1(ver)+1(flags)+4(srid) + envelope
    wkb = data[wkb_offset:]

    if len(wkb) < 9:
        return None

    byte_order = struct.unpack_from("B", wkb, 0)[0]
    endian = "<" if byte_order == 1 else ">"

    geom_type = struct.unpack_from(endian + "I", wkb, 1)[0]
    geom_clean = geom_type & 0x1FFFFFFF

    offset = 5
    if geom_clean == 5:  # MultiLineString → 첫 서브지오메트리
        n_geoms = struct.unpack_from(endian + "I", wkb, offset)[0]
        if n_geoms == 0:
            return None
        offset += 4
        bo2 = struct.unpack_from("B", wkb, offset)[0]
        endian = "<" if bo2 == 1 else ">"
        offset += 5  # sub-geom header
    elif geom_clean != 2:
        return None

    n_pts = struct.unpack_from(endian + "I", wkb, offset)[0]
    if n_pts == 0:
        return None
    offset += 4

    xs, ys = [], []
    for _ in range(n_pts):
        x, y = struct.unpack_from(endian + "dd", wkb, offset)
        xs.append(x)
        ys.append(y)
        offset += 16

    return (sum(xs) / len(xs), sum(ys) / len(ys))  # (lon, lat) 중점


def build():
    if os.path.exists(ROAD_HIER_CACHE):
        df = pd.read_csv(ROAD_HIER_CACHE)
        print("캐시 로드: {} ({:,}건)".format(ROAD_HIER_CACHE, len(df)))
        return df

    print("도로 위계 데이터 처리 시작...")

    # 격자 데이터 로드
    grid_df = pd.read_csv(GRID_CSV)
    coords_g = grid_df[["lat", "lon"]].values.astype(float)
    tree = cKDTree(coords_g)
    cell_ids = grid_df["CELL_ID"].tolist()
    print("  격자 수: {:,}".format(len(grid_df)))

    # 도로 데이터 읽기
    conn = sqlite3.connect(GPKG_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT "geom", "highway_type", "lanes_num", "GVI", "length"
                      FROM "서울 도로 위계 및 GVI"''')
    rows = cursor.fetchall()
    conn.close()
    print("  도로 구간 수: {:,}".format(len(rows)))

    # 도로별 중점 추출 + 격자 매핑
    cell_data = {}  # CELL_ID → list of (highway_rank, lanes, gvi, length)

    n_failed = 0
    for geom_bytes, hw_type, lanes, gvi, length in rows:
        midpoint = parse_gpkg_geom(geom_bytes)
        if midpoint is None:
            n_failed += 1
            continue

        lon, lat = midpoint
        rank = HIGHWAY_RANK.get(hw_type, 0)

        # 가장 가까운 격자 탐색 (위도도 단위 근사)
        dist, idx = tree.query([lat, lon])
        cell_id = cell_ids[idx]

        if cell_id not in cell_data:
            cell_data[cell_id] = []
        cell_data[cell_id].append((rank, lanes or 2.0, gvi or 0.0, length or 0.0))

    print("  파싱 성공: {:,}건  실패: {:,}건".format(len(rows)-n_failed, n_failed))

    # 격자별 집계
    records = []
    for cell_id, segs in cell_data.items():
        ranks   = [s[0] for s in segs]
        lanes   = [s[1] for s in segs]
        gvis    = [s[2] for s in segs]
        lengths = [s[3] for s in segs]
        records.append({
            "CELL_ID":              cell_id,
            "highway_rank":         max(ranks),           # 격자 내 최고 위계 도로
            "max_lanes":            max(lanes),
            "mean_gvi":             float(np.mean(gvis)),
            "total_road_length_m":  sum(lengths),
            "n_road_segments":      len(segs),
        })

    df_out = pd.DataFrame(records)

    # 도로 없는 격자 → 0으로 채움
    all_cells = pd.DataFrame({"CELL_ID": cell_ids})
    df_out = all_cells.merge(df_out, on="CELL_ID", how="left")
    df_out["highway_rank"]       = df_out["highway_rank"].fillna(0).astype(int)
    df_out["max_lanes"]          = df_out["max_lanes"].fillna(0)
    df_out["mean_gvi"]           = df_out["mean_gvi"].fillna(0.0)
    df_out["total_road_length_m"]= df_out["total_road_length_m"].fillna(0.0)
    df_out["n_road_segments"]    = df_out["n_road_segments"].fillna(0).astype(int)

    # 통계
    covered = (df_out["highway_rank"] > 0).sum()
    print("\n도로 위계 피처 통계:")
    print("  도로 있는 격자: {:,} / {:,} ({:.1f}%)".format(
        covered, len(df_out), 100*covered/len(df_out)))
    print("  highway_rank 분포:")
    print(df_out["highway_rank"].value_counts().sort_index().to_string())
    print("  max_lanes: mean={:.1f}  max={:.0f}".format(
        df_out["max_lanes"].mean(), df_out["max_lanes"].max()))
    print("  mean_gvi: mean={:.3f}  std={:.3f}".format(
        df_out["mean_gvi"].mean(), df_out["mean_gvi"].std()))

    df_out.to_csv(ROAD_HIER_CACHE, index=False, encoding="utf-8-sig")
    print("\n저장 완료: {}".format(ROAD_HIER_CACHE))
    return df_out


if __name__ == "__main__":
    build()

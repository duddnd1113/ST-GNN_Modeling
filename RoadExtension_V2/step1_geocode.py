"""
[Step 1] 도로명 Geocoding → 격자 매핑

실행:
    python3 step1_geocode.py

출력:
    checkpoints/road_geocoded.csv
    - 컬럼: 지역명, 도로명, lat, lon, CELL_ID, geocode_status
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import openpyxl
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial import cKDTree

from config import ROAD_PM_FILES, GRID_CSV, GEOCODE_CACHE


def load_unique_roads():
    """도로 측정 데이터에서 고유 (지역명, 도로명) 추출."""
    dfs = []
    for path in ROAD_PM_FILES:
        wb = openpyxl.load_workbook(path, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()
        dfs.append(pd.DataFrame(rows[1:], columns=rows[0]))

    df = pd.concat(dfs, ignore_index=True)
    df["지역명"] = df["지역명"].astype(str).str.strip()
    df["도로명"] = df["도로명"].astype(str).str.strip()

    roads = df[["지역명", "도로명"]].drop_duplicates().reset_index(drop=True)
    print("고유 도로 수: {}개".format(len(roads)))
    return roads


def geocode_roads(roads: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    """각 (지역명, 도로명)을 Nominatim으로 geocoding → 가장 가까운 CELL_ID 연결."""
    geolocator = Nominatim(user_agent="stgnn_road_v2", timeout=15)
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, error_wait_seconds=5)

    # grid KDTree (lat/lon)
    coords_grid = grid_df[["lat", "lon"]].values  # (G, 2)
    tree = cKDTree(coords_grid)

    results = []
    n = len(roads)
    for i, row in roads.iterrows():
        query = "서울특별시 {} {}".format(row["지역명"], row["도로명"])
        try:
            loc = geocode(query)
            if loc is None:
                # 구 이름 없이 재시도
                loc = geocode("서울 {}".format(row["도로명"]))
        except Exception:
            loc = None

        if loc is not None:
            lat, lon = loc.latitude, loc.longitude
            dist_deg, idx = tree.query([lat, lon])
            # 위도 1° ≈ 111 km → 도 단위 거리를 km로 근사
            dist_m = dist_deg * 111_000
            cell_id = grid_df.iloc[idx]["CELL_ID"]
            status = "ok"
        else:
            lat, lon, cell_id, dist_m = None, None, None, None
            status = "failed"

        results.append({
            "지역명":    row["지역명"],
            "도로명":    row["도로명"],
            "lat":       lat,
            "lon":       lon,
            "CELL_ID":   cell_id,
            "dist_m":    dist_m,
            "status":    status,
        })

        if (i + 1) % 20 == 0:
            ok_cnt = sum(1 for r in results if r["status"] == "ok")
            print("  [{}/{}] 성공: {}건".format(i + 1, n, ok_cnt))

    return pd.DataFrame(results)


def main():
    if os.path.exists(GEOCODE_CACHE):
        df = pd.read_csv(GEOCODE_CACHE)
        ok  = (df["status"] == "ok").sum()
        tot = len(df)
        print("캐시 로드: {} → 성공 {}/{}, 실패 {}".format(
            GEOCODE_CACHE, ok, tot, tot - ok))
        return df

    print("=== [Step 1] 도로명 Geocoding ===\n")
    grid_df = pd.read_csv(GRID_CSV)
    roads   = load_unique_roads()

    print("\nNominatim geocoding 시작 (약 5~6분 소요)...")
    result_df = geocode_roads(roads, grid_df)

    ok  = (result_df["status"] == "ok").sum()
    tot = len(result_df)
    print("\nGeocoding 완료:")
    print("  성공: {}/{} ({:.1f}%)".format(ok, tot, 100 * ok / tot))
    print("  실패: {}개".format(tot - ok))
    if ok > 0:
        ok_df = result_df[result_df["status"] == "ok"]
        print("  격자 거리 mean: {:.0f}m  max: {:.0f}m".format(
            ok_df["dist_m"].mean(), ok_df["dist_m"].max()))

    result_df.to_csv(GEOCODE_CACHE, index=False, encoding="utf-8-sig")
    print("\n저장: {}".format(GEOCODE_CACHE))
    return result_df


if __name__ == "__main__":
    main()

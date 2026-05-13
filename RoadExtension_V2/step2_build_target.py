"""
[Step 2] 도로 재비산먼지 → 격자 단위 정답 레이블 생성

이상치 처리 (2단계):
  1단계 - 절대 극단치 제거: "매우나쁨" 상태 (49건, >200 μg/m³)
  2단계 - 맥락적 IQR 탐지: (지역명+계절) 그룹별 Q3 + k*IQR 초과값 → 그룹 중앙값으로 대체
    - 단순 percentile clip과 달리 같은 구라도 계절별 기준이 다름
    - cap이 아닌 대체(replacement)이므로 분포가 더 자연스러움

격자 할당:
  - geocoded midpoint 기준 반경 GEOCODE_RADIUS_M 이내 격자에 할당
  - road_struc% > MIN_ROAD_STRUC 조건 필터
  - 같은 (date, hour, CELL_ID)에 여러 측정값 → 평균

출력:
    checkpoints/road_pm_grid.csv

실행:
    python3 step2_build_target.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import (
    ROAD_PM_FILES, GRID_LUR_CSV, GEOCODE_CACHE, ROAD_TARGET_CSV,
    GEOCODE_RADIUS_M, MIN_ROAD_STRUC, EXCLUDE_STATUS, CKPT_DIR,
    OUTLIER_GROUP, OUTLIER_IQR_K,
)


def load_road_pm() -> pd.DataFrame:
    dfs = []
    for path in ROAD_PM_FILES:
        wb = openpyxl.load_workbook(path, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()
        dfs.append(pd.DataFrame(rows[1:], columns=rows[0]))

    df = pd.concat(dfs, ignore_index=True)
    df["측정일자"]  = pd.to_datetime(df["측정일자"])
    df["재비산먼지"] = pd.to_numeric(df["재비산먼지 평균농도(㎍/㎥)"], errors="coerce")
    df["기온"]      = pd.to_numeric(df["기온(℃)"], errors="coerce")
    df["습도"]      = pd.to_numeric(df["습도(%)"], errors="coerce")
    df["지역명"]    = df["지역명"].astype(str).str.strip()
    df["도로명"]    = df["도로명"].astype(str).str.strip()
    df["상태"]      = df["상태"].astype(str).str.strip()

    df = df.dropna(subset=["재비산먼지", "측정일자"])
    df = df[df["재비산먼지"] >= 0]

    # 1단계: 극단 상태 제외
    n_before = len(df)
    df = df[~df["상태"].isin(EXCLUDE_STATUS)]
    print("  1단계 극단 상태 제외: {} → {}건 ({} 건 제거)".format(
        n_before, len(df), n_before - len(df)))

    # 시간/날짜 정보
    df["hour"] = df["측정시간"].astype(str).str.split(":").str[0]
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(10).astype(int)
    df["date"] = df["측정일자"].dt.normalize()
    df["year"] = df["측정일자"].dt.year
    df["month"] = df["측정일자"].dt.month
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    return df


def contextual_outlier_replace(df: pd.DataFrame) -> pd.DataFrame:
    """
    2단계: 그룹별 IQR 기반 이상치를 그룹 중앙값으로 대체.

    그룹: OUTLIER_GROUP = ["지역명", "season"]
    기준: Q3 + OUTLIER_IQR_K * IQR 초과 → 해당 그룹의 중앙값으로 대체
    (단순 cap이 아닌 replacement → 분포가 더 자연스러움)
    """
    df = df.copy()
    col = "재비산먼지"
    total_replaced = 0
    group_stats = []

    for key, grp in df.groupby(OUTLIER_GROUP):
        vals = grp[col]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        fence = q3 + OUTLIER_IQR_K * iqr
        group_median = vals.median()
        outlier_mask = grp.index[vals > fence]

        if len(outlier_mask) == 0:
            continue

        df.loc[outlier_mask, col] = group_median
        total_replaced += len(outlier_mask)
        group_stats.append({
            "group": key,
            "fence": fence,
            "median": group_median,
            "n_replaced": len(outlier_mask),
        })

    print("  2단계 맥락적 이상치 대체 (그룹: {}, k={}):".format(
        "+".join(OUTLIER_GROUP), OUTLIER_IQR_K))
    print("    {}건 → 해당 그룹 중앙값으로 교체".format(total_replaced))

    # 상위 5개 그룹 출력
    group_stats.sort(key=lambda x: -x["n_replaced"])
    for s in group_stats[:5]:
        print("    그룹 {}: fence={:.1f}  median={:.1f}  대체={}건".format(
            s["group"], s["fence"], s["median"], s["n_replaced"]))

    return df, total_replaced


def assign_to_grid(
    road_df: pd.DataFrame,
    geocode_df: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """각 측정값을 반경 내 격자에 할당."""
    geo_ok = geocode_df[geocode_df["status"] == "ok"].copy()
    geo_ok["lat"] = pd.to_numeric(geo_ok["lat"])
    geo_ok["lon"] = pd.to_numeric(geo_ok["lon"])
    print("  geocode 성공 도로: {}개".format(len(geo_ok)))

    coords_g = grid_df[["lat", "lon"]].values.astype(float)
    tree = cKDTree(coords_g)
    radius_deg = GEOCODE_RADIUS_M / 111_000.0

    road_to_cells = {}
    for _, row in geo_ok.iterrows():
        idxs = tree.query_ball_point([row["lat"], row["lon"]], r=radius_deg)
        key = (row["지역명"], row["도로명"])
        road_to_cells[key] = idxs

    n_avg = np.mean([len(v) for v in road_to_cells.values()]) if road_to_cells else 0
    print("  도로당 평균 격자 수: {:.1f}".format(n_avg))

    rows_out = []
    for _, rec in road_df.iterrows():
        key = (rec["지역명"], rec["도로명"])
        cell_idxs = road_to_cells.get(key, [])
        for ci in cell_idxs:
            g_row = grid_df.iloc[ci]
            if g_row["road_struc"] < MIN_ROAD_STRUC:
                continue
            rows_out.append({
                "date":    rec["date"],
                "hour":    rec["hour"],
                "CELL_ID": g_row["CELL_ID"],
                "road_pm": rec["재비산먼지"],
                "year":    rec["year"],
                "기온":     rec["기온"],
                "습도":     rec["습도"],
                "지역명":   rec["지역명"],
                "도로명":   rec["도로명"],
            })

    if not rows_out:
        raise RuntimeError("격자 할당 결과 없음. geocoding 결과를 확인하세요.")

    df_out = pd.DataFrame(rows_out)
    print("  할당 전: {}건 → 격자 확장 후: {}건  (커버 격자: {}개)".format(
        len(road_df), len(df_out), df_out["CELL_ID"].nunique()))

    # 같은 (date, hour, CELL_ID) 중복 → 평균
    df_agg = df_out.groupby(["date", "hour", "CELL_ID", "year"]).agg(
        road_pm=("road_pm", "mean"),
        기온=("기온", "mean"),
        습도=("습도", "mean"),
    ).reset_index()
    print("  중복 평균 후: {}건".format(len(df_agg)))
    return df_agg


def validate_grid_assignment(df: pd.DataFrame, grid_df: pd.DataFrame):
    """격자 할당 검증 통계 및 분포 시각화."""
    print("\n=== 격자 할당 검증 ===")
    print("커버 격자: {} / 전체 {}".format(df["CELL_ID"].nunique(), len(grid_df)))

    daily = df.groupby("date")["CELL_ID"].nunique()
    print("날짜별 커버 격자 수: mean={:.1f}  min={}  max={}".format(
        daily.mean(), daily.min(), daily.max()))

    print("\nroad_pm 분포 (이상치 처리 후):")
    pm = df["road_pm"].dropna()
    print("  mean={:.2f}  std={:.2f}  max={:.2f}".format(pm.mean(), pm.std(), pm.max()))
    print("  25th={:.1f}  50th={:.1f}  75th={:.1f}  99th={:.1f}".format(
        *np.percentile(pm, [25, 50, 75, 99])))

    print("\n연도별 건수:")
    print(df.groupby("year").size().to_string())

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(pm, bins=50, edgecolor="none", color="#2196F3", alpha=0.8)
    ax.set_title("Road PM10 Distribution")
    ax.set_xlabel("road_pm (μg/m³)")
    ax.set_ylabel("count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    cell_counts = df.groupby("CELL_ID").size()
    ax.hist(cell_counts, bins=30, edgecolor="none", color="#4CAF50", alpha=0.8)
    ax.set_title("Samples per Grid Cell")
    ax.set_xlabel("sample count")
    ax.set_ylabel("grid count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    hour_dist = df.groupby("hour").size()
    ax.bar(hour_dist.index, hour_dist.values, color="#FF9800", alpha=0.8)
    ax.set_title("Measurement Hour Distribution")
    ax.set_xlabel("hour")
    ax.set_ylabel("count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(CKPT_DIR, "target_validation.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("\n검증 그래프 저장: {}".format(out_path))


def main():
    print("=== [Step 2] 도로 PM → 격자 정답 레이블 생성 ===\n")

    if os.path.exists(ROAD_TARGET_CSV):
        df = pd.read_csv(ROAD_TARGET_CSV, parse_dates=["date"])
        print("캐시 로드: {} ({} 건)".format(ROAD_TARGET_CSV, len(df)))
        return df

    # 1. 원시 데이터 로드
    print("1. 도로 PM 로드...")
    road_df = load_road_pm()

    # 2. 맥락적 이상치 대체 (격자 할당 전에 적용)
    print("\n2. 이상치 처리...")
    road_df, n_replaced = contextual_outlier_replace(road_df)
    print("  처리 후 분포: mean={:.1f}  std={:.1f}  max={:.1f}".format(
        road_df["재비산먼지"].mean(),
        road_df["재비산먼지"].std(),
        road_df["재비산먼지"].max()))

    # 3. geocoding 결과 로드
    print("\n3. Geocoding 결과 로드...")
    if not os.path.exists(GEOCODE_CACHE):
        raise FileNotFoundError(
            "geocoding 캐시 없음. step1_geocode.py를 먼저 실행하세요: {}".format(GEOCODE_CACHE))
    geo_df = pd.read_csv(GEOCODE_CACHE)

    # 4. 격자 LUR 데이터 로드
    print("\n4. 격자 LUR 데이터 로드...")
    grid_df = pd.read_csv(GRID_LUR_CSV)
    print("  격자 수: {}".format(len(grid_df)))

    # 5. 격자 할당
    print("\n5. 격자 할당 (반경 {}m, road_struc >= {}%)...".format(
        GEOCODE_RADIUS_M, MIN_ROAD_STRUC))
    df_grid = assign_to_grid(road_df, geo_df, grid_df)

    # 6. 저장
    df_grid.to_csv(ROAD_TARGET_CSV, index=False, encoding="utf-8-sig")
    print("\n저장 완료: {}".format(ROAD_TARGET_CSV))

    # 7. 검증
    validate_grid_assignment(df_grid, grid_df)

    return df_grid


if __name__ == "__main__":
    main()

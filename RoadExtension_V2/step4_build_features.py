"""
[Step 4] 피처 행렬 생성

road_pm_grid.csv (정답 레이블) + 모든 피처를 합쳐
features_train.csv / features_test.csv 생성.

피처:
  temporal   : month_sin/cos, hour_sin/cos, weekday, is_weekend, season
  weather    : 기온, 습도, is_dry
  lur        : buildings, greenspace, road_struc, river_zone, ndvi, ibi, 건물통계
  traffic    : traffic (격자별 교통량)
  ambient_pm : ambient_pm10 (V5 grid PM10)
  interaction: hum_x_pm10, temp_x_pm10, traffic_x_road, is_dry_winter

출력:
    checkpoints/features_train.csv
    checkpoints/features_test.csv

실행:
    python3 step4_build_features.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    ROAD_TARGET_CSV, GRID_LUR_CSV, TRAFFIC_PARQUET,
    TIMESTAMPS_PATH, NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH, TIME_IDX, STGNN_WINDOW,
    V5_GRID_PM, FEATURES_TRAIN_CSV, FEATURES_TEST_CSV, CKPT_DIR,
)


# ── Temporal 피처 ─────────────────────────────────────────────────────────────
def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]      = df["date"].dt.month
    df["weekday"]    = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["season"]     = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ── Weather 피처 ──────────────────────────────────────────────────────────────
def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["기온"] = pd.to_numeric(df["기온"], errors="coerce")
    df["습도"] = pd.to_numeric(df["습도"], errors="coerce")
    # month-median 보간
    df["기온"] = df["기온"].fillna(df.groupby("month")["기온"].transform("median"))
    df["습도"] = df["습도"].fillna(df.groupby("month")["습도"].transform("median"))
    df["is_dry"] = (df["습도"] < 40).astype(int)
    return df


# ── LUR 피처 (정적 + 시간 평균) ──────────────────────────────────────────────
def add_lur(df: pd.DataFrame) -> pd.DataFrame:
    """grid LUR CSV + building/landcover 배열에서 정적 피처 추가."""
    grid_df = pd.read_csv(GRID_LUR_CSV)

    # ndvi_hourly, ibi_hourly → 기간 평균 (전체 기간 mean)
    ndvi_arr = np.load(NDVI_PATH)   # (T_all, G)
    ibi_arr  = np.load(IBI_PATH)    # (T_all, G)
    lc_arr   = np.load(LC_PATH)     # (G, 4)  [buildings%, green%, road%, river%]
    bldg_arr = np.load(BLDG_PATH)   # (G, 3)  [elev_mean, sum_area, sum_height]

    # grid 인덱스: CELL_ID → row idx
    cell2idx = {cid: i for i, cid in enumerate(grid_df["CELL_ID"])}

    # 시간 평균 NDVI, IBI per grid
    ndvi_mean = ndvi_arr.mean(axis=0)  # (G,)
    ibi_mean  = ibi_arr.mean(axis=0)

    lur_records = {}
    for cid, idx in cell2idx.items():
        lur_records[cid] = {
            "buildings":   float(grid_df.at[idx, "buildings"]),
            "greenspace":  float(grid_df.at[idx, "greenspace"]),
            "road_struc":  float(grid_df.at[idx, "road_struc"]),
            "river_zone":  float(grid_df.at[idx, "river_zone"]),
            "ndvi":        float(ndvi_mean[idx]),
            "ibi":         float(ibi_mean[idx]),
            "elev_mean":   float(bldg_arr[idx, 0]),
            "sum_area":    float(bldg_arr[idx, 1]),
            "sum_height":  float(bldg_arr[idx, 2]),
        }
    lur_df = pd.DataFrame.from_dict(lur_records, orient="index")
    lur_df.index.name = "CELL_ID"
    lur_df = lur_df.reset_index()

    df = df.merge(lur_df, on="CELL_ID", how="left")
    print("  LUR 병합 완료. NaN: {}".format(df[list(lur_df.columns[1:])].isna().sum().sum()))
    return df


# ── 교통량 피처 ───────────────────────────────────────────────────────────────
def add_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """격자별 교통량 parquet에서 (CELL_ID, date, hour) 조회."""
    print("  교통량 데이터 로드 (필요한 날짜만)...")

    needed_dates = df["date"].dt.normalize().unique()
    print("  필요 날짜: {}일".format(len(needed_dates)))

    # parquet에서 필요한 날짜 행만 읽기
    needed_dates_pd = pd.DatetimeIndex(needed_dates)

    tc_rows = []
    pf = pq.ParquetFile(TRAFFIC_PARQUET)
    for batch in pf.iter_batches(
            batch_size=2_000_000,
            columns=["CELL_ID", "일자", "시간", "교통량_합계"]):
        b = batch.to_pandas()
        b["date"] = b["일자"].dt.normalize()
        b = b[b["date"].isin(needed_dates_pd)]
        if len(b) > 0:
            tc_rows.append(b)

    if not tc_rows:
        print("  경고: 교통량 매칭 없음. traffic=NaN으로 처리.")
        df["traffic"] = np.nan
        return df

    tc = pd.concat(tc_rows, ignore_index=True)
    tc = tc.rename(columns={"시간": "hour", "교통량_합계": "traffic"})
    tc = tc.groupby(["CELL_ID", "date", "hour"])["traffic"].mean().reset_index()

    before = len(df)
    df = df.merge(tc, on=["CELL_ID", "date", "hour"], how="left")
    matched = df["traffic"].notna().sum()
    print("  교통량 매칭: {}/{} ({:.1f}%)".format(matched, before, 100 * matched / before))
    return df


# ── V5 Ambient PM10 피처 ─────────────────────────────────────────────────────
def add_ambient_pm(df: pd.DataFrame) -> pd.DataFrame:
    """V5 grid PM10 lookup → (CELL_ID, timestamp) 기반 매칭."""
    lookup_path = os.path.join(CKPT_DIR, "v5_ts_lookup.csv")
    if not os.path.exists(lookup_path):
        print("  V5 TS lookup 없음. step3를 먼저 실행하세요. ambient_pm10=NaN")
        df["ambient_pm10"] = np.nan
        return df

    ts_lookup = pd.read_csv(lookup_path, parse_dates=["timestamp"])
    ts_lookup["date"] = ts_lookup["timestamp"].dt.normalize()
    ts_lookup["hour"] = ts_lookup["timestamp"].dt.hour

    # 필요한 (date, hour) 조합만 처리
    df_keys = df[["date", "hour"]].drop_duplicates()
    needed = ts_lookup.merge(df_keys, on=["date", "hour"], how="inner")

    if len(needed) == 0:
        print("  V5 timestamp 매칭 없음. ambient_pm10=NaN")
        df["ambient_pm10"] = np.nan
        return df

    # Grid CSV: CELL_ID → 배열 인덱스
    from config import GRID_CSV
    grid_df = pd.read_csv(GRID_CSV)
    cell2arr_idx = {cid: i for i, cid in enumerate(grid_df["CELL_ID"])}

    # V5 grid PM10 로드 (split별)
    pm_by_split = {}
    for split in ["train", "val", "test"]:
        path = V5_GRID_PM[split]
        if os.path.exists(path):
            pm_by_split[split] = np.load(path)

    if not pm_by_split:
        print("  V5 grid PM 파일 없음. step3 실행 필요")
        df["ambient_pm10"] = np.nan
        return df

    # (date, hour, CELL_ID) → ambient_pm10
    pm_lookup = {}
    for _, row in needed.iterrows():
        split = row["split"]
        if split not in pm_by_split:
            continue
        local_i = int(row["local_idx"])
        pm_arr = pm_by_split[split][local_i]  # (G,)
        key = (row["date"], int(row["hour"]))
        pm_lookup[key] = pm_arr

    def get_ambient(row):
        key = (pd.Timestamp(row["date"]).normalize(), int(row["hour"]))
        pm_arr = pm_lookup.get(key)
        if pm_arr is None:
            return np.nan
        arr_idx = cell2arr_idx.get(row["CELL_ID"])
        if arr_idx is None:
            return np.nan
        return float(pm_arr[arr_idx])

    df["ambient_pm10"] = df.apply(get_ambient, axis=1)
    matched = df["ambient_pm10"].notna().sum()
    print("  V5 ambient PM 매칭: {}/{} ({:.1f}%)".format(
        matched, len(df), 100 * matched / len(df)))
    return df


# ── 교호작용 피처 ─────────────────────────────────────────────────────────────
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    단독 변수로 포착하지 못하는 결합 효과를 교호작용 피처로 명시적 생성.

    hum_x_pm10    : 건조(낮은 습도) × 배경 PM 높음 → 극단 도로 먼지 시너지
    temp_x_pm10   : 기온 × ambient PM → 고온 도시 PM 환경
    traffic_x_road: 실제 교통 노출 = 교통량 × 도로 면적 비율
    is_dry_winter : 겨울(12~2월) × 건조 조건 → 1,2월 극단 패턴 포착
    """
    df = df.copy()

    # cold_and_dry: 기온<5°C AND 습도<50% → 겨울 극단 도로 먼지 조건
    # (상관계수 r=0.214: cold_and_dry=1 평균 43μg/m³ vs =0 평균 18μg/m³)
    df["cold_and_dry"] = (
        (df["기온"].fillna(999) < 5.0) & (df["습도"].fillna(999) < 50.0)
    ).astype(int)

    # traffic_x_road: 실질적 교통 노출 (traffic NaN → NaN 유지)
    df["traffic_x_road"] = df["traffic"] * df["road_struc"].fillna(0.0) / 100.0

    n_cad = df["cold_and_dry"].sum()
    print("  교호작용 피처 생성 완료 (cold_and_dry 샘플: {}건)".format(n_cad))
    return df


def main():
    print("=== [Step 4] 피처 행렬 생성 ===\n")

    if os.path.exists(FEATURES_TRAIN_CSV) and os.path.exists(FEATURES_TEST_CSV):
        df_tr = pd.read_csv(FEATURES_TRAIN_CSV, parse_dates=["date"])
        df_te = pd.read_csv(FEATURES_TEST_CSV,  parse_dates=["date"])
        print("캐시 로드: train={}, test={}".format(len(df_tr), len(df_te)))
        return df_tr, df_te

    # 정답 레이블
    if not os.path.exists(ROAD_TARGET_CSV):
        raise FileNotFoundError("step2_build_target.py를 먼저 실행하세요")
    print("1. 정답 레이블 로드...")
    df = pd.read_csv(ROAD_TARGET_CSV, parse_dates=["date"])
    print("  {} 건".format(len(df)))

    # 피처 추가
    print("\n2. Temporal 피처...")
    df = add_temporal(df)

    print("\n3. Weather 피처...")
    df = add_weather(df)

    print("\n4. LUR 피처...")
    df = add_lur(df)

    print("\n5. 교통량 피처...")
    df = add_traffic(df)

    print("\n6. V5 Ambient PM10 피처...")
    df = add_ambient_pm(df)

    print("\n7. 교호작용 피처...")
    df = add_interactions(df)

    # 결측치 요약
    print("\n피처 결측치:")
    for col in ["traffic", "ambient_pm10"]:
        nan_pct = 100 * df[col].isna().mean()
        print("  {}: {:.1f}% NaN".format(col, nan_pct))

    # Train / Test 분할 (2025 = test)
    mask_te = df["year"] == 2025
    df_tr = df[~mask_te].reset_index(drop=True)
    df_te = df[mask_te].reset_index(drop=True)

    print("\n분할:")
    print("  Train: {} 건 (2023~2024)".format(len(df_tr)))
    print("  Test : {} 건 (2025)".format(len(df_te)))

    df_tr.to_csv(FEATURES_TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_te.to_csv(FEATURES_TEST_CSV,  index=False, encoding="utf-8-sig")
    print("\n저장 완료:")
    print("  {}".format(FEATURES_TRAIN_CSV))
    print("  {}".format(FEATURES_TEST_CSV))
    return df_tr, df_te


if __name__ == "__main__":
    main()

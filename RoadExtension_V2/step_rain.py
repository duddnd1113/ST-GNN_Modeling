"""
강수 데이터 전처리 (격자별 마지막 비 이후 일수)

입력: days from the last rain_by grid.parquet
  - CELL_ID, 조사년월일, 일일강수량, days from the last rain
  - 2023-10-01 ~ 2025-10-31 기간, 10125개 격자

출력: checkpoints/rain_cache.csv
  - CELL_ID, date, days_from_rain, daily_precip_mm

실행:
    python3 step_rain.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CKPT_DIR

RAIN_PARQUET = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/days from the last rain_by grid.parquet"
RAIN_CACHE   = os.path.join(CKPT_DIR, "rain_cache.csv")

BATCH_SIZE = 2_000_000


def build():
    if os.path.exists(RAIN_CACHE):
        df = pd.read_csv(RAIN_CACHE, parse_dates=["date"])
        print("캐시 로드: {} ({:,}건)".format(RAIN_CACHE, len(df)))
        return df

    print("강수 데이터 전처리 시작...")
    pf = pq.ParquetFile(RAIN_PARQUET)

    accum = {}  # (CELL_ID, date_int) → [days_from_rain, daily_precip]

    for i, batch in enumerate(pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["CELL_ID", "조사년월일", "일일강수량", "days from the last rain"])):
        df_b = batch.to_pandas()
        df_b = df_b.dropna(subset=["days from the last rain"])

        for _, row in df_b.iterrows():
            key = (row["CELL_ID"], int(row["조사년월일"]))
            if key not in accum:
                accum[key] = [row["days from the last rain"], row["일일강수량"]]

        if i % 5 == 0:
            print("  배치 {:3d}  누적 keys={:,}".format(i, len(accum)))

    records = []
    for (cell_id, date_int), (dlr, precip) in accum.items():
        records.append({
            "CELL_ID":        cell_id,
            "date":           pd.to_datetime(str(date_int)),
            "days_from_rain": dlr,
            "daily_precip_mm": precip,
        })

    df_out = pd.DataFrame(records).sort_values(["CELL_ID", "date"]).reset_index(drop=True)

    # 통계 출력
    dlr = df_out["days_from_rain"]
    print("\n강수 피처 통계:")
    print("  격자 수: {:,}".format(df_out["CELL_ID"].nunique()))
    print("  날짜 범위: {} ~ {}".format(df_out["date"].min().date(), df_out["date"].max().date()))
    print("  days_from_rain: mean={:.1f}  median={:.1f}  max={:.0f}".format(
        dlr.mean(), dlr.median(), dlr.max()))
    print("  0~3일 비율: {:.1f}%  3~7일: {:.1f}%  7~15일: {:.1f}%  >15일: {:.1f}%".format(
        100*(dlr<=3).mean(), 100*((dlr>3)&(dlr<=7)).mean(),
        100*((dlr>7)&(dlr<=15)).mean(), 100*(dlr>15).mean()))

    df_out.to_csv(RAIN_CACHE, index=False, encoding="utf-8-sig")
    print("\n저장 완료: {}".format(RAIN_CACHE))
    return df_out


if __name__ == "__main__":
    build()

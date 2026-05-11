"""
교통량 parquet → (날짜, 시간)별 서울 전체 평균 사전 집계

185M 행을 한 번만 처리해 CSV로 캐싱.
이후 preprocess.py에서 빠르게 로드.

실행 (처음 한 번만):
    python3 build_traffic_cache.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import TRAFFIC_PARQUET, TRAFFIC_CACHE

BATCH_SIZE = 2_000_000


def build():
    os.makedirs(os.path.dirname(TRAFFIC_CACHE), exist_ok=True)

    if os.path.exists(TRAFFIC_CACHE):
        print(f"캐시 이미 존재: {TRAFFIC_CACHE}")
        df = pd.read_csv(TRAFFIC_CACHE, parse_dates=['date'])
        print(f"  rows={len(df)}  columns={df.columns.tolist()}")
        return df

    print("교통량 집계 시작 (약 2~3분 소요)...")
    pf = pq.ParquetFile(TRAFFIC_PARQUET)

    # (date, hour) → [sum, count] 누적
    accum = {}

    for i, batch in enumerate(pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=['일자', '시간', '교통량_합계'])):
        df_b = batch.to_pandas()
        df_b['date'] = df_b['일자'].dt.normalize()
        grp = df_b.groupby(['date', '시간'])['교통량_합계'].agg(['sum', 'count'])

        for (date, hour), row in grp.iterrows():
            key = (date, int(hour))
            if key in accum:
                accum[key][0] += row['sum']
                accum[key][1] += row['count']
            else:
                accum[key] = [row['sum'], row['count']]

        if i % 20 == 0:
            print(f"  배치 {i:3d}  누적 keys={len(accum)}")

    # dict → DataFrame
    records = []
    for (date, hour), (s, c) in accum.items():
        records.append({'date': date, 'hour': hour, 'traffic_mean': s / c if c > 0 else 0})

    df_cache = pd.DataFrame(records).sort_values(['date', 'hour']).reset_index(drop=True)

    # 추가 집계 피처
    df_cache['traffic_log']  = np.log1p(df_cache['traffic_mean'])

    # 시간대 구분
    def hour_bin(h):
        if h <= 6:   return 0   # 야간
        if h <= 9:   return 1   # 아침 러시
        if h <= 12:  return 2   # 오전
        if h <= 15:  return 3   # 오후
        if h <= 19:  return 4   # 저녁 러시
        return 5                # 야간

    df_cache['hour_bin']    = df_cache['hour'].apply(hour_bin)
    df_cache['is_rush']     = df_cache['hour_bin'].isin([1, 4]).astype(int)

    df_cache.to_csv(TRAFFIC_CACHE, index=False)
    print(f"\n저장 완료: {TRAFFIC_CACHE}")
    print(f"  shape: {df_cache.shape}")
    print(f"  날짜 범위: {df_cache['date'].min()} ~ {df_cache['date'].max()}")
    print(f"  교통량 mean: {df_cache['traffic_mean'].mean():.1f}  "
          f"max: {df_cache['traffic_mean'].max():.1f}")
    return df_cache


if __name__ == "__main__":
    build()

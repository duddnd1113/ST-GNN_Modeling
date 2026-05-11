"""
도로 재비산먼지 데이터 전처리 + 피처 엔지니어링
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import openpyxl
from sklearn.preprocessing import LabelEncoder

from config import ROAD_PM_FILES, SCENARIO_DIR, SCENARIO_NAME, HIDDEN_DIR, CKPT_DIR, TRAFFIC_CACHE


def load_raw() -> pd.DataFrame:
    dfs = []
    for path in ROAD_PM_FILES:
        wb   = openpyxl.load_workbook(path, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()
        dfs.append(pd.DataFrame(rows[1:], columns=rows[0]))
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['측정일자']   = pd.to_datetime(df['측정일자'])
    df['재비산먼지']  = pd.to_numeric(df['재비산먼지 평균농도(㎍/㎥)'], errors='coerce')
    df['기온']       = pd.to_numeric(df['기온(℃)'], errors='coerce')
    df['습도']       = pd.to_numeric(df['습도(%)'], errors='coerce')
    df['측정거리']   = pd.to_numeric(df['측정거리(km)'], errors='coerce')
    df['지역명']     = df['지역명'].str.strip()
    df['도로명']     = df['도로명'].str.strip()
    df = df.dropna(subset=['재비산먼지', '측정일자', '지역명', '도로명'])
    df = df[df['재비산먼지'] >= 0].reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 시간 피처
    df['year']       = df['측정일자'].dt.year
    df['month']      = df['측정일자'].dt.month
    df['weekday']    = df['측정일자'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['season']     = df['month'].map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                         6:2,7:2,8:2, 9:3,10:3,11:3})
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
    # 기상 보완
    df['도로길이'] = df['측정거리'].fillna(df['측정거리'].median())
    df['기온']     = df['기온'].fillna(df.groupby('month')['기온'].transform('median'))
    df['습도']     = df['습도'].fillna(df.groupby('month')['습도'].transform('median'))
    df['is_dry']   = (df['습도'] < 40).astype(int)
    # 범주형 인코딩
    le_gu = LabelEncoder().fit(df['지역명'])
    df['gu_enc'] = le_gu.transform(df['지역명'])
    return df, le_gu


def merge_ambient_pm(df: pd.DataFrame) -> pd.DataFrame:
    """같은 날, 같은 구의 대기 PM10 일평균을 연결."""
    csv_path = os.path.join(SCENARIO_DIR, f"{SCENARIO_NAME}.csv")
    if not os.path.exists(csv_path):
        print(f"  대기 PM10 CSV 없음: {csv_path}")
        df['ambient_pm10_daily'] = np.nan
        return df

    pm = pd.read_csv(csv_path)
    pm['time'] = pd.to_datetime(pm['time'])
    pm['date'] = pm['time'].dt.normalize()
    pm['지역명'] = pm['측정소명'].apply(
        lambda x: x.split('구')[0]+'구' if '구' in str(x) else x)

    daily = pm.groupby(['date','지역명'])['PM10'].mean().reset_index()
    daily.columns = ['date','지역명','ambient_pm10_daily']

    df['date'] = df['측정일자'].dt.normalize()
    df = df.merge(daily, on=['date','지역명'], how='left')
    matched = df['ambient_pm10_daily'].notna().sum()
    print(f"  대기 PM10 연결: {matched}/{len(df)}건 매칭")
    return df


def merge_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """측정 날짜+시간에 해당하는 서울 평균 교통량 연결."""
    if not os.path.exists(TRAFFIC_CACHE):
        print("  교통량 캐시 없음 → build_traffic_cache.py를 먼저 실행하세요")
        df['traffic_mean'] = np.nan
        df['traffic_log']  = np.nan
        df['is_rush']      = np.nan
        return df

    tc = pd.read_csv(TRAFFIC_CACHE, parse_dates=['date'])
    tc['date'] = pd.to_datetime(tc['date']).dt.normalize()

    # 도로 측정 시간 → hour 추출 (예: '10:41' → 10)
    df['measure_hour'] = df['측정시간'].str.split(':').str[0].astype(int, errors='ignore')
    df['date']         = df['측정일자'].dt.normalize()

    before = len(df)
    df = df.merge(
        tc[['date', 'hour', 'traffic_mean', 'traffic_log', 'is_rush']],
        left_on=['date', 'measure_hour'],
        right_on=['date', 'hour'],
        how='left'
    ).drop(columns=['hour'])

    matched = df['traffic_mean'].notna().sum()
    print(f"  교통량 연결: {matched}/{before}건 매칭")
    print(f"  교통량 stats: mean={df['traffic_mean'].mean():.1f}  "
          f"min={df['traffic_mean'].min():.1f}  max={df['traffic_mean'].max():.1f}")
    return df


def build_dataset():
    print("1. 원시 데이터 로드...")
    df = load_raw()
    print(f"   {len(df)}건")

    print("2. 정제...")
    df = clean(df)
    print(f"   {len(df)}건 남음")

    print("3. 피처 엔지니어링...")
    df, le_gu = add_features(df)

    print("4. 대기 PM10 연결...")
    df = merge_ambient_pm(df)

    print("5. 교통량 연결...")
    df = merge_traffic(df)

    return df, le_gu


if __name__ == "__main__":
    df, le_gu = build_dataset()
    out = os.path.join(CKPT_DIR, "road_pm_features.csv")
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {out}  shape={df.shape}")

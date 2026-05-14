"""
공사 데이터 전처리 → 격자 단위 공사 피처 생성

입력: 서울시건설알림이정보.csv
  - 프로젝트 코드, 사업착수일, 사업기간, 위치좌표, 사업금액, 자치구

처리:
  1. 연구 기간(2023-01~2025-12) 관련 공사 필터
  2. 공사 위치 → 반경 500m 내 격자 매핑 (cKDTree)
  3. 각 (CELL_ID, date)에 대해 활성 공사 집계

출력: checkpoints/construction_cache.csv
  - CELL_ID, date, n_active_const, total_amount_억, max_amount_억, is_large_const

  n_active_const : 반경 500m 내 활성 공사 수
  total_amount_억: 활성 공사 사업금액 합 (억원)
  max_amount_억  : 활성 공사 중 최대 사업금액
  is_large_const : 10억 이상 공사 존재 여부 (binary)

실행:
    python3 step_construction.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CONST_CSV   = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/서울시건설알림이정보.csv"
GRID_CSV    = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/격자 기본/격자_250m_4326.csv"
CKPT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUTPUT_PATH = os.path.join(CKPT_DIR, "construction_cache.csv")

RADIUS_M      = 500    # 공사 영향 반경 (미터)
LARGE_AMOUNT  = 10.0   # 대형 공사 기준 (억원)
STUDY_START   = pd.Timestamp("2023-01-01")
STUDY_END     = pd.Timestamp("2025-12-31")


def load_construction() -> pd.DataFrame:
    df = pd.read_csv(CONST_CSV, encoding="cp949")

    df["start_date"] = pd.to_datetime(
        df["사업착수일(계약일)"].astype(str), format="%Y%m%d", errors="coerce")
    df["end_date"] = pd.to_datetime(
        df["사업기간"].astype(str).str.split("~").str[-1].str.strip(),
        format="%Y%m%d", errors="coerce")
    df["status"] = df["프로젝트 종료여부(진행:0 종료:1,예정:2, 중지:3)"]
    df["amount"] = pd.to_numeric(df["사업금액(억원)"], errors="coerce").fillna(0.0)
    df["lat"]    = pd.to_numeric(df["위치좌표(위도)"], errors="coerce")
    df["lon"]    = pd.to_numeric(df["위치좌표(경도)"], errors="coerce")

    # 필터: 연구 기간 겹침 + 서울 범위 + 중지 아닌 것
    mask = (
        df["start_date"].notna() &
        df["end_date"].notna() &
        (df["start_date"] <= STUDY_END) &
        (df["end_date"]   >= STUDY_START) &
        (df["status"] != 3) &
        df["lat"].between(37.4, 37.7) &
        df["lon"].between(126.7, 127.2)
    )
    result = df[mask].reset_index(drop=True)
    print("  유효 공사 프로젝트: {:,}건".format(len(result)))
    return result


def build_project_grid_map(const_df: pd.DataFrame, grid_df: pd.DataFrame) -> dict:
    """
    공사 위치 → 반경 RADIUS_M 내 CELL_ID 목록 매핑.
    반환: {project_idx: [CELL_ID, ...]}
    """
    coords_g = grid_df[["lat", "lon"]].values.astype(float)
    tree = cKDTree(coords_g)
    radius_deg = RADIUS_M / 111_000.0

    proj_to_cells = {}
    for idx, row in const_df.iterrows():
        cell_idxs = tree.query_ball_point([row["lat"], row["lon"]], r=radius_deg)
        if cell_idxs:
            proj_to_cells[idx] = [grid_df.iloc[i]["CELL_ID"] for i in cell_idxs]

    covered = len(proj_to_cells)
    print("  격자 매핑 성공: {:,} / {:,} 프로젝트".format(covered, len(const_df)))
    avg_cells = np.mean([len(v) for v in proj_to_cells.values()]) if proj_to_cells else 0
    print("  프로젝트당 평균 매핑 격자: {:.1f}개".format(avg_cells))
    return proj_to_cells


def compute_construction_features(
    const_df: pd.DataFrame,
    proj_to_cells: dict,
    target_dates: pd.DatetimeIndex,
    all_cell_ids: list,
) -> pd.DataFrame:
    """
    각 (CELL_ID, date)에 대해 활성 공사 피처 집계.

    처리 흐름:
      1. CELL_ID → 관련 프로젝트 역매핑 구성
      2. 날짜별로 활성 프로젝트 필터
      3. (CELL_ID, date) 피처 계산
    """
    # 역매핑: CELL_ID → 관련 프로젝트 인덱스 목록
    cell_to_projs = {}
    for proj_idx, cell_ids in proj_to_cells.items():
        for cid in cell_ids:
            if cid not in cell_to_projs:
                cell_to_projs[cid] = []
            cell_to_projs[cid].append(proj_idx)

    cells_with_proj = set(cell_to_projs.keys())
    print("  공사 인접 격자: {:,}개 / {:,}개 (전체)".format(
        len(cells_with_proj), len(all_cell_ids)))

    records = []
    for date in target_dates:
        # 해당 날짜에 활성인 프로젝트만 추출
        active_mask = (const_df["start_date"] <= date) & (const_df["end_date"] >= date)
        active_idxs = set(const_df[active_mask].index)

        for cid in cells_with_proj:
            proj_idxs = [p for p in cell_to_projs[cid] if p in active_idxs]
            if not proj_idxs:
                continue

            amounts = const_df.loc[proj_idxs, "amount"].values
            records.append({
                "CELL_ID":         cid,
                "date":            date,
                "n_active_const":  len(proj_idxs),
                "total_amount_억":  float(amounts.sum()),
                "max_amount_억":    float(amounts.max()),
                "is_large_const":  int((amounts >= LARGE_AMOUNT).any()),
            })

    df_out = pd.DataFrame(records)
    return df_out


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
        print("캐시 로드: {} ({:,}건)".format(OUTPUT_PATH, len(df)))
        return df

    print("=== 공사 데이터 전처리 ===\n")

    print("1. 공사 데이터 로드...")
    const_df = load_construction()

    print("\n2. 격자 데이터 로드...")
    grid_df = pd.read_csv(GRID_CSV)
    print("  격자 수: {:,}".format(len(grid_df)))

    print("\n3. 공사-격자 매핑 (반경 {}m)...".format(RADIUS_M))
    proj_to_cells = build_project_grid_map(const_df, grid_df)

    # 연구 기간 전체 날짜 생성
    all_dates = pd.date_range(STUDY_START, STUDY_END, freq="D")
    print("\n4. 날짜별 공사 피처 계산 ({:,}일)...".format(len(all_dates)))

    df_out = compute_construction_features(
        const_df, proj_to_cells, all_dates, grid_df["CELL_ID"].tolist())

    # 통계 출력
    n_total = len(all_dates) * len(grid_df)
    print("\n=== 공사 피처 통계 ===")
    print("  저장 건수: {:,} / {:,} 가능 (공사 없는 (격자,날짜) 제외)".format(
        len(df_out), n_total))
    print("  커버 격자: {:,}개".format(df_out["CELL_ID"].nunique()))
    print("  커버 날짜: {:,}일".format(df_out["date"].nunique()))
    print("\n  n_active_const: mean={:.2f}  max={:.0f}".format(
        df_out["n_active_const"].mean(), df_out["n_active_const"].max()))
    print("  total_amount_억: mean={:.2f}  max={:.1f}".format(
        df_out["total_amount_억"].mean(), df_out["total_amount_억"].max()))
    print("  is_large_const=1: {:.1f}%".format(
        100 * df_out["is_large_const"].mean()))

    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n저장 완료: {}".format(OUTPUT_PATH))
    return df_out


if __name__ == "__main__":
    main()

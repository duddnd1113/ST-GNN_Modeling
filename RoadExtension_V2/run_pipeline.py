"""
RoadExtension V2 전체 파이프라인 실행 스크립트

각 스텝이 이미 완료된 경우 스킵 (캐싱).

실행:
    python3 run_pipeline.py [--from_step N] [--skip_v5]

옵션:
    --from_step N  : N번 스텝부터 실행 (기본: 1)
    --skip_v5      : step3 (V5 전체 inference) 건너뜀 (ambient_pm 피처 없이 학습)
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GEOCODE_CACHE, ROAD_TARGET_CSV,
    FEATURES_TRAIN_CSV, FEATURES_TEST_CSV,
    ABLATION_RESULTS, CKPT_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_step", type=int, default=1)
    parser.add_argument("--skip_v5",  action="store_true",
                        help="V5 전체 inference 건너뜀 (ambient_pm 피처 없이 진행)")
    args = parser.parse_args()

    print("=" * 65)
    print("RoadExtension V2 Pipeline")
    print("=" * 65)

    # Step 1: Geocoding
    if args.from_step <= 1:
        print("\n[Step 1] 도로 Geocoding")
        if os.path.exists(GEOCODE_CACHE):
            print("  캐시 존재 → 스킵")
        else:
            import step1_geocode
            step1_geocode.main()
    else:
        print("[Step 1] 스킵")

    # Step 2: 격자 정답 생성
    if args.from_step <= 2:
        print("\n[Step 2] 격자 정답 레이블 생성")
        if os.path.exists(ROAD_TARGET_CSV):
            print("  캐시 존재 → 스킵")
        else:
            import step2_build_target
            step2_build_target.main()
    else:
        print("[Step 2] 스킵")

    # Step 3: V5 전체 inference
    if args.from_step <= 3:
        if args.skip_v5:
            print("\n[Step 3] V5 inference — --skip_v5 옵션으로 건너뜀")
        else:
            print("\n[Step 3] V5 Grid PM10 전체 기간 inference")
            import step3_v5_inference_all
            step3_v5_inference_all.main()
    else:
        print("[Step 3] 스킵")

    # Step 4: 피처 행렬 생성
    if args.from_step <= 4:
        print("\n[Step 4] 피처 행렬 생성")
        if os.path.exists(FEATURES_TRAIN_CSV) and os.path.exists(FEATURES_TEST_CSV):
            print("  캐시 존재 → 스킵")
        else:
            import step4_build_features
            step4_build_features.main()
    else:
        print("[Step 4] 스킵")

    # Step 5: 학습
    if args.from_step <= 5:
        print("\n[Step 5] Ablation Study 학습")
        import step5_train
        step5_train.main()
    else:
        print("[Step 5] 스킵")

    # Step 6: 평가
    if args.from_step <= 6:
        print("\n[Step 6] 결과 분석")
        import step6_evaluate
        step6_evaluate.main()
    else:
        print("[Step 6] 스킵")

    # Step 7: 시각화
    if args.from_step <= 7:
        print("\n[Step 7] 시각화")
        import step7_visualize
        step7_visualize.main()
    else:
        print("[Step 7] 스킵")

    print("\n" + "=" * 65)
    print("파이프라인 완료. 결과: {}".format(CKPT_DIR))
    print("=" * 65)


if __name__ == "__main__":
    main()

import json
import glob
import csv
import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_summary.csv")

FIELDS = [
    "experiment", "x_mode", "attn_mode", "lur_mode", "r_dim", "lam",
    "direct_mae", "direct_rmse", "direct_r2",
    "cross_mae", "cross_rmse", "cross_r2",
    "stgnn_mae", "best_val",
]


def load_results():
    results = []
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "**/metrics.json"), recursive=True)):
        with open(path) as f:
            d = json.load(f)
        d["experiment"] = path.split("checkpoints/")[1].replace("/metrics.json", "")
        results.append(d)
    return results


def print_table(results, sort_by="direct_mae"):
    sorted_results = sorted(results, key=lambda r: r.get(sort_by, 9999))
    baseline = results[0].get("stgnn_mae", None)

    col = {
        "rank":       5,
        "experiment": 55,
        "direct_mae": 12,
        "direct_r2":  10,
        "cross_mae":  12,
        "cross_r2":   10,
    }

    header = (
        f"{'Rank':<{col['rank']}} {'Experiment':<{col['experiment']}} "
        f"{'direct_MAE':<{col['direct_mae']}} {'direct_R2':<{col['direct_r2']}} "
        f"{'cross_MAE':<{col['cross_mae']}} {'cross_R2':<{col['cross_r2']}}"
    )
    print(f"\n=== 결과 정렬 기준: {sort_by} ===")
    if baseline is not None:
        print(f"[베이스라인 ST-GNN MAE: {baseline:.4f}]")
    print(header)
    print("-" * (sum(col.values()) + len(col)))

    for i, r in enumerate(sorted_results, 1):
        better = ""
        if baseline and sort_by == "direct_mae":
            diff = r.get("direct_mae", 0) - baseline
            better = f"  ({'↑' if diff < 0 else '↓'}{abs(diff):.4f})"
        print(
            f"{i:<{col['rank']}} {r['experiment']:<{col['experiment']}} "
            f"{r.get('direct_mae', 0):.4f}{'':>6} {r.get('direct_r2', 0):.4f}{'':>4} "
            f"{r.get('cross_mae', 0):.4f}{'':>6} {r.get('cross_r2', 0):.4f}"
            f"{better}"
        )

    best = sorted_results[0]
    print(f"\n[최고 성능] {best['experiment']}")
    print(f"  direct_MAE : {best.get('direct_mae', '-'):.4f}  direct_R2 : {best.get('direct_r2', '-'):.4f}")
    print(f"  cross_MAE  : {best.get('cross_mae', '-'):.4f}  cross_R2  : {best.get('cross_r2', '-'):.4f}")
    if baseline:
        diff = best.get("direct_mae", 0) - baseline
        sign = "개선" if diff < 0 else "악화"
        print(f"  vs baseline: {sign} {abs(diff):.4f} (ST-GNN MAE={baseline:.4f})")


def save_csv(results, sort_by="direct_mae"):
    sorted_results = sorted(results, key=lambda r: r.get(sort_by, 9999))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_results)
    print(f"\nCSV 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    results = load_results()
    print(f"총 {len(results)}개 실험 로드")

    print_table(results, sort_by="direct_mae")
    print_table(results, sort_by="cross_mae")

    save_csv(results, sort_by="direct_mae")

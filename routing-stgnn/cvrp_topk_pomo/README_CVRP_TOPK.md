# Top-k CVRP POMO-style prototype

This folder adapts the uploaded CVRP POMO code to your road-dust setting:

```text
22500 scored grid cells -> select top-k cells -> convert score to demand -> solve CVRP
```

- `CVRPTopKProblemDef.py`: top-k selection, score-to-demand conversion, synthetic 22500 grid generator, 8-fold augmentation.
- `CVRPTopKEnvironment.py`: depot + selected nodes + capacity/load mask.
- `CVRPTopKModel.py`: same CVRP attention model style; depot is embedded from `(x,y)`, customer nodes from `(x,y,demand)`.
- `CVRPTopKTrainer.py`: POMO shared-baseline RL training.
- `CVRPTopKTester.py`: testing and `solve_real_csv()` helper.
- `train_topk_cvrp_n50.py`: train on synthetic top-k CVRP instances.
- `test_topk_cvrp_n50.py`: test from checkpoint.
- `train_test_visualize_cvrp_snapshot.py`: one-snapshot demo that writes CSV and PNG.

Run demo:

```bash
python train_test_visualize_cvrp_snapshot.py --total-grid-size 22500 --top-k 50 --epochs 300
```

Real CSV expected format:

```csv
grid_id,x,y,score
0,126.97,37.56,42.1
1,126.98,37.57,35.2
```

Optional: include a `demand` column and pass `demand_col='demand'` in `solve_real_csv()`.

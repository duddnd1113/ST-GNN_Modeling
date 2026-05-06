# Top-k TSP POMO-style prototype

This folder modifies the POMO TSP setup for your Seoul road-dust use case:

```text
22,500 scored grid cells -> select top-k cells -> solve TSP over selected top-k cells
```

## Main change from vanilla POMO TSP

Vanilla TSP assumes all nodes are mandatory.
This version first filters candidate grid cells by score:

```python
selected_coords, selected_scores, selected_indices = select_topk_cells(coords, scores, top_k)
```

Then POMO runs exactly like TSP over only those selected cells.

## Files

- `TSProblemDef.py`: random scored grid generation, top-k selection, 8-fold augmentation
- `TSPTopKEnv.py`: environment; selects top-k before rollout
- `TSPTopKModel.py`: POMO-style TSP attention model
- `TSPTopKTrainer.py`: POMO training with shared baseline
- `TSPTopKTester.py`: testing and optional CSV solving helper
- `train_topk_n50.py`: training entrypoint
- `test_topk_n50.py`: testing entrypoint

## Real CSV format

For real Seoul LUR output, prepare a CSV like:

```csv
grid_id,x,y,score
0,126.97,37.56,42.1
1,126.98,37.57,35.2
...
```

Recommended score example:

```text
score = PM10_road_dust * road_length * population_exposure
```

The tester normalizes `x,y` internally, selects top-k by `score`, then returns route order.

# Feature selection

Shuffle features to see impact.

Based on ShuffleGate: https://arxiv.org/html/2503.09315v2

Used ChatGPT to generate the code.

## Runner

Use `shuffle_gate_importance.py` to evaluate the trained `tetris_dqn.pt` without
retraining. For each candidate action batch, the selected feature columns are
shuffled across candidate states. The report ranks feature sets by paired score
drop against the same episode seeds; larger positive drops mean the model relied
on those features more.

Run grouped feature families first:

```bash
cd /gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection
../.venv/bin/python shuffle_gate_importance.py --runs 100 --selection groups
```

Run every scalar feature separately:

```bash
../.venv/bin/python shuffle_gate_importance.py --runs 100 --selection individual --prefix individual
```

Run explicit combinations:

```bash
../.venv/bin/python shuffle_gate_importance.py \
  --runs 100 \
  --condition fill_heights,height_summary \
  --condition all_hole_features \
  --condition mean_height,height_deviation
```

Outputs go to `shuffle_reports/`:

- `*_report.md`: readable ranked report
- `*_summary.csv`: aggregate scores and deltas
- `*_raw.csv`: per-seed raw results
- `*_top.png`: bar chart of the biggest score drops

# ShuffleGate Feature Importance Report

Generated: 2026-05-23T23:19:52
Model: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/tetris_dqn.pt`
Runs per condition: `100`
Episode seeds: `0` to `99`
Max pieces per episode: `5000`
Shuffle mode: `independent`

This report uses ShuffleGate-style batch-wise feature shuffling during action evaluation. Larger positive deltas mean the agent performed worse when that feature set was shuffled, so the feature set appears more important to the current trained policy.

## Baseline

| Metric | Value |
| --- | ---: |
| Mean score | 692.40 |
| Score std | 693.45 |
| Median score | 480.00 |
| Mean lines | 13.36 |
| Mean pieces | 72.07 |
| Truncated runs | 0 |

## Top 20 Most Sensitive Conditions

| Rank | Condition | Kind | Features | Score drop | Lines drop | Shuffled mean score |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `height_deviation` | individual | 1 | 596.40 | 11.30 | 96.00 |
| 2 | `vertical_hole_depth_col_0` | individual | 1 | 362.60 | 6.29 | 329.80 |
| 3 | `horizontal_hole_edge_distance_row_7` | individual | 1 | 340.20 | 6.51 | 352.20 |
| 4 | `horizontal_hole_edge_distance_row_9` | individual | 1 | 310.60 | 5.51 | 381.80 |
| 5 | `horizontal_hole_edge_distance_row_6` | individual | 1 | 282.20 | 5.03 | 410.20 |
| 6 | `mean_hole_edge_distance` | individual | 1 | 246.60 | 4.17 | 445.80 |
| 7 | `vertical_hole_depth_col_5` | individual | 1 | 244.00 | 3.68 | 448.40 |
| 8 | `fill_height_col_1` | individual | 1 | 241.40 | 4.15 | 451.00 |
| 9 | `vertical_hole_clusteredness_col_4` | individual | 1 | 232.40 | 4.47 | 460.00 |
| 10 | `holeyness_col_5` | individual | 1 | 224.80 | 3.83 | 467.60 |
| 11 | `holeyness_col_4` | individual | 1 | 220.80 | 3.85 | 471.60 |
| 12 | `horizontal_hole_edge_distance_row_11` | individual | 1 | 219.20 | 4.63 | 473.20 |
| 13 | `highest_point` | individual | 1 | 211.00 | 4.16 | 481.40 |
| 14 | `fill_height_col_7` | individual | 1 | 209.40 | 3.61 | 483.00 |
| 15 | `mean_hole_vertical_clusteredness` | individual | 1 | 204.80 | 3.17 | 487.60 |
| 16 | `vertical_hole_clusteredness_col_3` | individual | 1 | 204.60 | 3.24 | 487.80 |
| 17 | `vertical_hole_depth_col_2` | individual | 1 | 186.40 | 3.15 | 506.00 |
| 18 | `fill_height_col_8` | individual | 1 | 185.80 | 3.67 | 506.60 |
| 19 | `fill_height_col_3` | individual | 1 | 185.00 | 3.19 | 507.40 |
| 20 | `horizontal_hole_edge_distance_row_14` | individual | 1 | 176.80 | 2.56 | 515.60 |

## Least Sensitive Or Helpful When Shuffled

| Rank | Condition | Kind | Features | Score drop | Lines drop | Shuffled mean score |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `horizontal_hole_edge_distance_row_19` | individual | 1 | -175.60 | -2.22 | 868.00 |
| 2 | `horizontal_hole_clusteredness_row_19` | individual | 1 | -89.00 | -0.88 | 781.40 |
| 3 | `vertical_hole_clusteredness_col_1` | individual | 1 | -31.80 | -0.13 | 724.20 |
| 4 | `general_hole_clusteredness` | individual | 1 | -31.20 | -0.94 | 723.60 |
| 5 | `holeyness_col_2` | individual | 1 | -17.60 | 0.11 | 710.00 |
| 6 | `hole_horizontal_instability` | individual | 1 | -14.20 | 0.25 | 706.60 |
| 7 | `fill_height_col_9` | individual | 1 | -9.40 | 0.42 | 701.80 |
| 8 | `hole_edge_distance_deviation` | individual | 1 | -5.80 | -0.01 | 698.20 |
| 9 | `horizontal_hole_clusteredness_row_0` | individual | 1 | 0.00 | 0.00 | 692.40 |
| 10 | `horizontal_hole_edge_distance_row_0` | individual | 1 | 0.00 | 0.00 | 692.40 |
| 11 | `horizontal_hole_clusteredness_row_2` | individual | 1 | 5.80 | 0.26 | 686.60 |
| 12 | `horizontal_hole_clusteredness_row_1` | individual | 1 | 6.60 | 0.24 | 685.80 |
| 13 | `horizontal_hole_edge_distance_row_13` | individual | 1 | 12.80 | 0.83 | 679.60 |
| 14 | `holeyness_col_6` | individual | 1 | 22.40 | 0.95 | 670.00 |
| 15 | `horizontal_hole_edge_distance_row_1` | individual | 1 | 24.80 | 0.62 | 667.60 |
| 16 | `horizontal_hole_clusteredness_row_14` | individual | 1 | 29.80 | 0.91 | 662.60 |
| 17 | `horizontal_hole_edge_distance_row_17` | individual | 1 | 31.40 | 0.17 | 661.00 |
| 18 | `holeyness_col_8` | individual | 1 | 35.00 | 1.54 | 657.40 |
| 19 | `holeyness_col_0` | individual | 1 | 35.00 | 0.28 | 657.40 |
| 20 | `mean_height` | individual | 1 | 37.20 | 0.88 | 655.20 |

## Files

- Summary CSV: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/individual_summary.csv`
- Raw per-run CSV: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/individual_raw.csv`
- Plot: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/individual_top.png`

## Feature Groups

| Group | Feature count |
| --- | ---: |
| `fill_heights` | 10 |
| `height_summary` | 4 |
| `total_holeyness` | 1 |
| `holeyness_by_column` | 10 |
| `vertical_hole_depths` | 10 |
| `vertical_hole_clusteredness` | 10 |
| `horizontal_hole_edge_distances` | 20 |
| `horizontal_hole_clusteredness` | 20 |
| `hole_summary` | 10 |
| `all_height_features` | 14 |
| `all_hole_features` | 81 |
| `all_features` | 95 |

## Feature Index Map

| Index | Feature |
| ---: | --- |
| 0 | `fill_height_col_0` |
| 1 | `fill_height_col_1` |
| 2 | `fill_height_col_2` |
| 3 | `fill_height_col_3` |
| 4 | `fill_height_col_4` |
| 5 | `fill_height_col_5` |
| 6 | `fill_height_col_6` |
| 7 | `fill_height_col_7` |
| 8 | `fill_height_col_8` |
| 9 | `fill_height_col_9` |
| 10 | `total_holeyness` |
| 11 | `mean_height` |
| 12 | `height_deviation` |
| 13 | `lowest_point` |
| 14 | `highest_point` |
| 15 | `holeyness_col_0` |
| 16 | `holeyness_col_1` |
| 17 | `holeyness_col_2` |
| 18 | `holeyness_col_3` |
| 19 | `holeyness_col_4` |
| 20 | `holeyness_col_5` |
| 21 | `holeyness_col_6` |
| 22 | `holeyness_col_7` |
| 23 | `holeyness_col_8` |
| 24 | `holeyness_col_9` |
| 25 | `vertical_hole_depth_col_0` |
| 26 | `vertical_hole_depth_col_1` |
| 27 | `vertical_hole_depth_col_2` |
| 28 | `vertical_hole_depth_col_3` |
| 29 | `vertical_hole_depth_col_4` |
| 30 | `vertical_hole_depth_col_5` |
| 31 | `vertical_hole_depth_col_6` |
| 32 | `vertical_hole_depth_col_7` |
| 33 | `vertical_hole_depth_col_8` |
| 34 | `vertical_hole_depth_col_9` |
| 35 | `vertical_hole_clusteredness_col_0` |
| 36 | `vertical_hole_clusteredness_col_1` |
| 37 | `vertical_hole_clusteredness_col_2` |
| 38 | `vertical_hole_clusteredness_col_3` |
| 39 | `vertical_hole_clusteredness_col_4` |
| 40 | `vertical_hole_clusteredness_col_5` |
| 41 | `vertical_hole_clusteredness_col_6` |
| 42 | `vertical_hole_clusteredness_col_7` |
| 43 | `vertical_hole_clusteredness_col_8` |
| 44 | `vertical_hole_clusteredness_col_9` |
| 45 | `horizontal_hole_edge_distance_row_0` |
| 46 | `horizontal_hole_edge_distance_row_1` |
| 47 | `horizontal_hole_edge_distance_row_2` |
| 48 | `horizontal_hole_edge_distance_row_3` |
| 49 | `horizontal_hole_edge_distance_row_4` |
| 50 | `horizontal_hole_edge_distance_row_5` |
| 51 | `horizontal_hole_edge_distance_row_6` |
| 52 | `horizontal_hole_edge_distance_row_7` |
| 53 | `horizontal_hole_edge_distance_row_8` |
| 54 | `horizontal_hole_edge_distance_row_9` |
| 55 | `horizontal_hole_edge_distance_row_10` |
| 56 | `horizontal_hole_edge_distance_row_11` |
| 57 | `horizontal_hole_edge_distance_row_12` |
| 58 | `horizontal_hole_edge_distance_row_13` |
| 59 | `horizontal_hole_edge_distance_row_14` |
| 60 | `horizontal_hole_edge_distance_row_15` |
| 61 | `horizontal_hole_edge_distance_row_16` |
| 62 | `horizontal_hole_edge_distance_row_17` |
| 63 | `horizontal_hole_edge_distance_row_18` |
| 64 | `horizontal_hole_edge_distance_row_19` |
| 65 | `horizontal_hole_clusteredness_row_0` |
| 66 | `horizontal_hole_clusteredness_row_1` |
| 67 | `horizontal_hole_clusteredness_row_2` |
| 68 | `horizontal_hole_clusteredness_row_3` |
| 69 | `horizontal_hole_clusteredness_row_4` |
| 70 | `horizontal_hole_clusteredness_row_5` |
| 71 | `horizontal_hole_clusteredness_row_6` |
| 72 | `horizontal_hole_clusteredness_row_7` |
| 73 | `horizontal_hole_clusteredness_row_8` |
| 74 | `horizontal_hole_clusteredness_row_9` |
| 75 | `horizontal_hole_clusteredness_row_10` |
| 76 | `horizontal_hole_clusteredness_row_11` |
| 77 | `horizontal_hole_clusteredness_row_12` |
| 78 | `horizontal_hole_clusteredness_row_13` |
| 79 | `horizontal_hole_clusteredness_row_14` |
| 80 | `horizontal_hole_clusteredness_row_15` |
| 81 | `horizontal_hole_clusteredness_row_16` |
| 82 | `horizontal_hole_clusteredness_row_17` |
| 83 | `horizontal_hole_clusteredness_row_18` |
| 84 | `horizontal_hole_clusteredness_row_19` |
| 85 | `mean_hole_depth` |
| 86 | `mean_hole_vertical_clusteredness` |
| 87 | `hole_depth_deviation` |
| 88 | `hole_vertical_instability` |
| 89 | `mean_hole_edge_distance` |
| 90 | `mean_hole_horizontal_clusteredness` |
| 91 | `hole_edge_distance_deviation` |
| 92 | `hole_horizontal_instability` |
| 93 | `mean_hole_distance` |
| 94 | `general_hole_clusteredness` |

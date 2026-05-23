# Shuffle Result Analysis

Summary CSV: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/individual_summary.csv`
Raw CSV: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/individual_raw.csv`
Combined analysis CSV: `/gpfs/helios/home/etais/hpc_maif/DRL_track/04_feature_selection/shuffle_reports/analysis/shuffle_analysis_combined.csv`
Feature/condition count: `95`

## Plots

- `features_execution_order.png` / `.svg`: all features in original execution order.
- `features_score_drop_order.png` / `.svg`: all features sorted by score drop.
- `execution_order_scatter.png` / `.svg`: quick view of where important features sit in the feature vector.
- `score_vs_lines_drop.png` / `.svg`: checks whether score sensitivity matches line-clear sensitivity.
- `family_score_drop_boxplot.png` / `.svg`: compares feature families.
- `per_seed_score_delta_heatmap.png` / `.svg`: shows whether each feature is consistently important or only matters for some seeds.

## Top Score Drops

| Rank | Execution | Feature | Family | Score drop | 95% CI | Lines drop |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 13 | `height_deviation` | height_summary | 596.40 | 138.67 | 11.30 |
| 2 | 26 | `vertical_hole_depth_col_0` | vertical_hole_depth | 362.60 | 142.50 | 6.29 |
| 3 | 53 | `horizontal_hole_edge_distance_row_7` | horizontal_hole_edge_distance | 340.20 | 127.11 | 6.51 |
| 4 | 55 | `horizontal_hole_edge_distance_row_9` | horizontal_hole_edge_distance | 310.60 | 127.86 | 5.51 |
| 5 | 52 | `horizontal_hole_edge_distance_row_6` | horizontal_hole_edge_distance | 282.20 | 111.62 | 5.03 |
| 6 | 90 | `mean_hole_edge_distance` | hole_summary | 246.60 | 161.12 | 4.17 |
| 7 | 31 | `vertical_hole_depth_col_5` | vertical_hole_depth | 244.00 | 143.12 | 3.68 |
| 8 | 2 | `fill_height_col_1` | fill_height | 241.40 | 159.33 | 4.15 |
| 9 | 40 | `vertical_hole_clusteredness_col_4` | vertical_hole_clusteredness | 232.40 | 189.05 | 4.47 |
| 10 | 21 | `holeyness_col_5` | holeyness_col | 224.80 | 163.71 | 3.83 |
| 11 | 20 | `holeyness_col_4` | holeyness_col | 220.80 | 172.22 | 3.85 |
| 12 | 57 | `horizontal_hole_edge_distance_row_11` | horizontal_hole_edge_distance | 219.20 | 175.31 | 4.63 |
| 13 | 15 | `highest_point` | height_summary | 211.00 | 167.34 | 4.16 |
| 14 | 8 | `fill_height_col_7` | fill_height | 209.40 | 160.45 | 3.61 |
| 15 | 87 | `mean_hole_vertical_clusteredness` | hole_summary | 204.80 | 155.12 | 3.17 |
| 16 | 39 | `vertical_hole_clusteredness_col_3` | vertical_hole_clusteredness | 204.60 | 172.87 | 3.24 |
| 17 | 28 | `vertical_hole_depth_col_2` | vertical_hole_depth | 186.40 | 158.24 | 3.15 |
| 18 | 9 | `fill_height_col_8` | fill_height | 185.80 | 182.06 | 3.67 |
| 19 | 4 | `fill_height_col_3` | fill_height | 185.00 | 169.32 | 3.19 |
| 20 | 60 | `horizontal_hole_edge_distance_row_14` | horizontal_hole_edge_distance | 176.80 | 168.55 | 2.56 |

## Lowest Or Negative Score Drops

| Rank | Execution | Feature | Family | Score drop | 95% CI | Lines drop |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 65 | `horizontal_hole_edge_distance_row_19` | horizontal_hole_edge_distance | -175.60 | 169.22 | -2.22 |
| 2 | 85 | `horizontal_hole_clusteredness_row_19` | horizontal_hole_clusteredness | -89.00 | 178.37 | -0.88 |
| 3 | 37 | `vertical_hole_clusteredness_col_1` | vertical_hole_clusteredness | -31.80 | 194.54 | -0.13 |
| 4 | 95 | `general_hole_clusteredness` | hole_summary | -31.20 | 194.26 | -0.94 |
| 5 | 18 | `holeyness_col_2` | holeyness_col | -17.60 | 189.29 | 0.11 |
| 6 | 93 | `hole_horizontal_instability` | hole_summary | -14.20 | 189.17 | 0.25 |
| 7 | 10 | `fill_height_col_9` | fill_height | -9.40 | 184.43 | 0.42 |
| 8 | 92 | `hole_edge_distance_deviation` | hole_summary | -5.80 | 166.83 | -0.01 |
| 9 | 46 | `horizontal_hole_edge_distance_row_0` | horizontal_hole_edge_distance | 0.00 | 0.00 | 0.00 |
| 10 | 66 | `horizontal_hole_clusteredness_row_0` | horizontal_hole_clusteredness | 0.00 | 0.00 | 0.00 |
| 11 | 68 | `horizontal_hole_clusteredness_row_2` | horizontal_hole_clusteredness | 5.80 | 43.74 | 0.26 |
| 12 | 67 | `horizontal_hole_clusteredness_row_1` | horizontal_hole_clusteredness | 6.60 | 15.32 | 0.24 |
| 13 | 59 | `horizontal_hole_edge_distance_row_13` | horizontal_hole_edge_distance | 12.80 | 159.44 | 0.83 |
| 14 | 22 | `holeyness_col_6` | holeyness_col | 22.40 | 199.36 | 0.95 |
| 15 | 47 | `horizontal_hole_edge_distance_row_1` | horizontal_hole_edge_distance | 24.80 | 33.67 | 0.62 |
| 16 | 80 | `horizontal_hole_clusteredness_row_14` | horizontal_hole_clusteredness | 29.80 | 157.63 | 0.91 |
| 17 | 63 | `horizontal_hole_edge_distance_row_17` | horizontal_hole_edge_distance | 31.40 | 179.15 | 0.17 |
| 18 | 16 | `holeyness_col_0` | holeyness_col | 35.00 | 159.41 | 0.28 |
| 19 | 24 | `holeyness_col_8` | holeyness_col | 35.00 | 177.18 | 1.54 |
| 20 | 12 | `mean_height` | height_summary | 37.20 | 150.90 | 0.88 |

## Other Meaningful Plots To Consider

- Retrain-ablation plot: train a new model after removing the top N features; this tests whether importance survives retraining.
- Cumulative top-N curve: remove/shuffle the top 1, 2, 5, 10 features together and plot performance drop.
- Stability plot across multiple trained checkpoints: compare ranks from several model seeds, not just episode seeds.
- Correlation heatmap of raw feature values: highly correlated features may share importance or mask each other.

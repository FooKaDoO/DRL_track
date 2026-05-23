"""
Analyze ShuffleGate feature-importance CSV outputs.

The evaluator's summary CSV is ranked by score drop. This script restores the
original execution order from the raw CSV, writes a combined table, and creates
large zoom-friendly plots for all features/conditions.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = SCRIPT_DIR / "shuffle_reports"
DEFAULT_SUMMARY_CSV = DEFAULT_REPORT_DIR / "individual_summary.csv"
DEFAULT_RAW_CSV = DEFAULT_REPORT_DIR / "individual_raw.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "analysis"


NUMERIC_COLUMNS = {
    "rank",
    "n_features",
    "runs",
    "mean_score",
    "std_score",
    "median_score",
    "mean_lines",
    "std_lines",
    "mean_pieces",
    "truncated_runs",
    "baseline_mean_score",
    "baseline_mean_lines",
    "mean_score_delta",
    "std_score_delta",
    "mean_score_delta_pct",
    "mean_lines_delta",
    "std_lines_delta",
}


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: dict[str, object] = dict(row)
            for column in NUMERIC_COLUMNS:
                if column in parsed:
                    parsed[column] = to_float(str(parsed[column]))
            if "rank" in parsed:
                parsed["score_drop_rank"] = to_int(str(parsed["rank"]))
            rows.append(parsed)
    return rows


def read_raw(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "condition": row["condition"],
                "seed": to_int(row["seed"]),
                "score": to_float(row["score"]),
                "lines": to_int(row["lines"]),
                "pieces": to_int(row["pieces"]),
                "truncated": str(row["truncated"]).lower() == "true",
            })
    return rows


def infer_execution_order(raw_rows: list[dict[str, object]]) -> OrderedDict[str, int]:
    order: OrderedDict[str, int] = OrderedDict()
    for row in raw_rows:
        condition = str(row["condition"])
        if condition == "baseline":
            continue
        if condition not in order:
            order[condition] = len(order) + 1
    return order


def fallback_execution_order(summary_rows: list[dict[str, object]]) -> OrderedDict[str, int]:
    def feature_index(row: dict[str, object]) -> int:
        raw_indices = str(row.get("feature_indices", ""))
        first = raw_indices.split(";")[0]
        return to_int(first, default=10**9)

    ordered = sorted(summary_rows, key=feature_index)
    return OrderedDict((str(row["condition"]), idx + 1) for idx, row in enumerate(ordered))


def family_for_condition(condition: str) -> str:
    if condition.startswith("fill_height_col_"):
        return "fill_height"
    if condition in {"mean_height", "height_deviation", "lowest_point", "highest_point"}:
        return "height_summary"
    if condition == "total_holeyness":
        return "total_holeyness"
    if condition.startswith("holeyness_col_"):
        return "holeyness_col"
    if condition.startswith("vertical_hole_depth_col_"):
        return "vertical_hole_depth"
    if condition.startswith("vertical_hole_clusteredness_col_"):
        return "vertical_hole_clusteredness"
    if condition.startswith("horizontal_hole_edge_distance_row_"):
        return "horizontal_hole_edge_distance"
    if condition.startswith("horizontal_hole_clusteredness_row_"):
        return "horizontal_hole_clusteredness"
    if "hole" in condition:
        return "hole_summary"
    return "other"


def natural_label_key(label: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", label)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def add_analysis_columns(
    summary_rows: list[dict[str, object]],
    execution_order: OrderedDict[str, int],
) -> list[dict[str, object]]:
    sorted_by_drop = sorted(
        summary_rows,
        key=lambda row: float(row.get("mean_score_delta", 0.0)),
        reverse=True,
    )
    rank_by_drop = {
        str(row["condition"]): rank
        for rank, row in enumerate(sorted_by_drop, start=1)
    }

    rows = []
    for row in summary_rows:
        condition = str(row["condition"])
        enriched = dict(row)
        enriched["execution_order"] = execution_order.get(condition, 0)
        enriched["score_drop_rank"] = rank_by_drop[condition]
        enriched["family"] = family_for_condition(condition)
        runs = max(float(enriched.get("runs", 0.0)), 1.0)
        std_delta = float(enriched.get("std_score_delta", 0.0))
        enriched["score_delta_sem"] = std_delta / math.sqrt(runs)
        enriched["score_delta_ci95"] = 1.96 * float(enriched["score_delta_sem"])
        rows.append(enriched)

    return rows


def compute_raw_deltas(raw_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    baseline_by_seed = {
        int(row["seed"]): row
        for row in raw_rows
        if row["condition"] == "baseline"
    }

    deltas_by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        condition = str(row["condition"])
        if condition == "baseline":
            continue

        seed = int(row["seed"])
        baseline = baseline_by_seed.get(seed)
        if baseline is None:
            continue

        deltas_by_condition[condition].append({
            "seed": seed,
            "score_delta": float(baseline["score"]) - float(row["score"]),
            "lines_delta": int(baseline["lines"]) - int(row["lines"]),
            "pieces_delta": int(baseline["pieces"]) - int(row["pieces"]),
        })

    return deltas_by_condition


def write_analysis_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "execution_order",
        "score_drop_rank",
        "condition",
        "family",
        "kind",
        "n_features",
        "runs",
        "mean_score_delta",
        "std_score_delta",
        "score_delta_sem",
        "score_delta_ci95",
        "mean_score_delta_pct",
        "mean_lines_delta",
        "std_lines_delta",
        "mean_score",
        "std_score",
        "median_score",
        "mean_lines",
        "mean_pieces",
        "baseline_mean_score",
        "baseline_mean_lines",
        "feature_indices",
        "feature_names",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def figure_size(count: int, width: float, row_height: float, min_height: float = 6.0) -> tuple[float, float]:
    return width, max(min_height, count * row_height)


def load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def save_bar_plot(
    rows: list[dict[str, object]],
    output_base: Path,
    title: str,
    width: float,
    row_height: float,
    label_column: str = "condition",
    value_column: str = "mean_score_delta",
    error_column: str = "score_delta_ci95",
) -> None:
    plt = load_pyplot()
    labels = [str(row[label_column]) for row in rows]
    values = [float(row[value_column]) for row in rows]
    errors = [float(row.get(error_column, 0.0)) for row in rows]
    colors = [FAMILY_COLORS.get(str(row.get("family", "other")), FAMILY_COLORS["other"]) for row in rows]
    y_positions = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=figure_size(len(rows), width, row_height))
    ax.barh(y_positions, values, xerr=errors, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean paired score drop vs baseline")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output_base)


def save_execution_scatter(
    rows: list[dict[str, object]],
    output_base: Path,
    width: float,
) -> None:
    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(width, 7.0))

    families = sorted({str(row["family"]) for row in rows})
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        ax.scatter(
            [float(row["execution_order"]) for row in family_rows],
            [float(row["mean_score_delta"]) for row in family_rows],
            label=family,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["other"]),
            s=45,
            alpha=0.85,
        )

    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xlabel("Execution order / feature index order")
    ax.set_ylabel("Mean paired score drop vs baseline")
    ax.set_title("Score Drop Across Original Feature Order")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def save_score_vs_lines_scatter(
    rows: list[dict[str, object]],
    output_base: Path,
    width: float,
    label_top_n: int,
) -> None:
    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(width, 8.0))

    families = sorted({str(row["family"]) for row in rows})
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        ax.scatter(
            [float(row["mean_lines_delta"]) for row in family_rows],
            [float(row["mean_score_delta"]) for row in family_rows],
            label=family,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["other"]),
            s=45,
            alpha=0.85,
        )

    top_rows = sorted(rows, key=lambda row: float(row["mean_score_delta"]), reverse=True)[:label_top_n]
    for row in top_rows:
        ax.annotate(
            str(row["condition"]),
            (float(row["mean_lines_delta"]), float(row["mean_score_delta"])),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_xlabel("Mean paired lines drop vs baseline")
    ax.set_ylabel("Mean paired score drop vs baseline")
    ax.set_title("Score Drop vs Lines Drop")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def save_family_boxplot(
    rows: list[dict[str, object]],
    output_base: Path,
    width: float,
) -> None:
    plt = load_pyplot()
    family_names = []
    values = []
    for family in sorted({str(row["family"]) for row in rows}):
        family_values = [
            float(row["mean_score_delta"])
            for row in rows
            if row["family"] == family
        ]
        if family_values:
            family_names.append(family)
            values.append(family_values)

    fig, ax = plt.subplots(figsize=(width, max(6.0, 0.45 * len(family_names))))
    ax.boxplot(values, vert=False, patch_artist=True)
    ax.set_yticks(np.arange(1, len(family_names) + 1))
    ax.set_yticklabels(family_names)
    ax.axvline(0.0, color="#444444", linewidth=0.9)
    ax.set_xlabel("Mean paired score drop vs baseline")
    ax.set_title("Score Drop Distribution By Feature Family")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_base)


def save_delta_heatmap(
    rows: list[dict[str, object]],
    deltas_by_condition: dict[str, list[dict[str, object]]],
    output_base: Path,
    width: float,
    row_height: float,
    max_seeds: int | None,
) -> None:
    plt = load_pyplot()
    if not deltas_by_condition:
        return

    seed_values = sorted({
        int(delta["seed"])
        for deltas in deltas_by_condition.values()
        for delta in deltas
    })
    if max_seeds is not None:
        seed_values = seed_values[:max_seeds]

    seed_to_index = {seed: idx for idx, seed in enumerate(seed_values)}
    matrix = np.full((len(rows), len(seed_values)), np.nan, dtype=np.float32)

    for row_idx, row in enumerate(rows):
        condition = str(row["condition"])
        for delta in deltas_by_condition.get(condition, []):
            seed = int(delta["seed"])
            if seed in seed_to_index:
                matrix[row_idx, seed_to_index[seed]] = float(delta["score_delta"])

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return

    limit = float(np.nanpercentile(np.abs(finite_values), 95))
    if limit <= 0.0:
        limit = float(np.nanmax(np.abs(finite_values))) or 1.0

    heatmap_width = max(width, min(36.0, max(10.0, len(seed_values) * 0.18)))
    fig, ax = plt.subplots(figsize=figure_size(len(rows), heatmap_width, row_height))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([str(row["condition"]) for row in rows], fontsize=8)
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Feature / condition")
    ax.set_title("Per-Seed Score Drop Heatmap")
    if len(seed_values) <= 120:
        tick_step = max(1, len(seed_values) // 20)
        tick_positions = np.arange(0, len(seed_values), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(seed_values[idx]) for idx in tick_positions], rotation=90, fontsize=7)
    fig.colorbar(image, ax=ax, label="Baseline score - shuffled score")
    fig.tight_layout()
    save_figure(fig, output_base)


def save_figure(fig, output_base: Path) -> None:
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    load_pyplot().close(fig)


def write_markdown_report(
    path: Path,
    rows: list[dict[str, object]],
    output_dir: Path,
    summary_csv: Path,
    raw_csv: Path | None,
    analysis_csv: Path,
    top_n: int,
) -> None:
    top_rows = sorted(rows, key=lambda row: float(row["mean_score_delta"]), reverse=True)[:top_n]
    negative_rows = sorted(rows, key=lambda row: float(row["mean_score_delta"]))[:top_n]

    lines = [
        "# Shuffle Result Analysis",
        "",
        f"Summary CSV: `{summary_csv}`",
        f"Raw CSV: `{raw_csv}`" if raw_csv else "Raw CSV: not provided",
        f"Combined analysis CSV: `{analysis_csv}`",
        f"Feature/condition count: `{len(rows)}`",
        "",
        "## Plots",
        "",
        "- `features_execution_order.png` / `.svg`: all features in original execution order.",
        "- `features_score_drop_order.png` / `.svg`: all features sorted by score drop.",
        "- `execution_order_scatter.png` / `.svg`: quick view of where important features sit in the feature vector.",
        "- `score_vs_lines_drop.png` / `.svg`: checks whether score sensitivity matches line-clear sensitivity.",
        "- `family_score_drop_boxplot.png` / `.svg`: compares feature families.",
        "- `per_seed_score_delta_heatmap.png` / `.svg`: shows whether each feature is consistently important or only matters for some seeds.",
        "",
        "## Top Score Drops",
        "",
        "| Rank | Execution | Feature | Family | Score drop | 95% CI | Lines drop |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(top_rows, start=1):
        lines.append(
            f"| {rank} | {int(row['execution_order'])} | `{row['condition']}` | "
            f"{row['family']} | {float(row['mean_score_delta']):.2f} | "
            f"{float(row['score_delta_ci95']):.2f} | {float(row['mean_lines_delta']):.2f} |"
        )

    lines.extend([
        "",
        "## Lowest Or Negative Score Drops",
        "",
        "| Rank | Execution | Feature | Family | Score drop | 95% CI | Lines drop |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ])

    for rank, row in enumerate(negative_rows, start=1):
        lines.append(
            f"| {rank} | {int(row['execution_order'])} | `{row['condition']}` | "
            f"{row['family']} | {float(row['mean_score_delta']):.2f} | "
            f"{float(row['score_delta_ci95']):.2f} | {float(row['mean_lines_delta']):.2f} |"
        )

    lines.extend([
        "",
        "## Other Meaningful Plots To Consider",
        "",
        "- Retrain-ablation plot: train a new model after removing the top N features; this tests whether importance survives retraining.",
        "- Cumulative top-N curve: remove/shuffle the top 1, 2, 5, 10 features together and plot performance drop.",
        "- Stability plot across multiple trained checkpoints: compare ranks from several model seeds, not just episode seeds.",
        "- Correlation heatmap of raw feature values: highly correlated features may share importance or mask each other.",
    ])

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ShuffleGate result CSV files.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=float, default=16.0)
    parser.add_argument(
        "--row-height",
        type=float,
        default=0.28,
        help="Inches of figure height per feature/condition for long all-feature plots.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--heatmap-order",
        choices=["execution", "score_drop"],
        default="score_drop",
    )
    parser.add_argument(
        "--max-heatmap-seeds",
        type=int,
        default=0,
        help="Limit heatmap columns. Use 0 for all seeds.",
    )
    return parser.parse_args()


FAMILY_COLORS = {
    "fill_height": "#2F6F73",
    "height_summary": "#8F5FBF",
    "total_holeyness": "#D18B2F",
    "holeyness_col": "#C94943",
    "vertical_hole_depth": "#3F7DBF",
    "vertical_hole_clusteredness": "#5F9B50",
    "horizontal_hole_edge_distance": "#B85C8E",
    "horizontal_hole_clusteredness": "#81733F",
    "hole_summary": "#586E75",
    "other": "#777777",
}


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    summary_rows = read_summary(args.summary_csv)
    raw_rows = read_raw(args.raw_csv) if args.raw_csv.exists() else []

    if raw_rows:
        execution_order = infer_execution_order(raw_rows)
    else:
        execution_order = fallback_execution_order(summary_rows)

    rows = add_analysis_columns(summary_rows, execution_order)
    rows_by_execution = sorted(rows, key=lambda row: (int(row["execution_order"]), natural_label_key(str(row["condition"]))))
    rows_by_score_drop = sorted(rows, key=lambda row: float(row["mean_score_delta"]), reverse=True)

    deltas_by_condition = compute_raw_deltas(raw_rows) if raw_rows else {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv = args.output_dir / "shuffle_analysis_combined.csv"
    report_md = args.output_dir / "shuffle_analysis_report.md"

    write_analysis_csv(analysis_csv, rows_by_execution)
    save_bar_plot(
        rows_by_execution,
        args.output_dir / "features_execution_order",
        "All Features In Execution Order",
        args.width,
        args.row_height,
    )
    save_bar_plot(
        rows_by_score_drop,
        args.output_dir / "features_score_drop_order",
        "All Features Ordered By Score Drop",
        args.width,
        args.row_height,
    )
    save_execution_scatter(
        rows_by_execution,
        args.output_dir / "execution_order_scatter",
        args.width,
    )
    save_score_vs_lines_scatter(
        rows_by_score_drop,
        args.output_dir / "score_vs_lines_drop",
        args.width,
        args.top_n,
    )
    save_family_boxplot(
        rows_by_execution,
        args.output_dir / "family_score_drop_boxplot",
        args.width,
    )
    heatmap_rows = rows_by_execution if args.heatmap_order == "execution" else rows_by_score_drop
    save_delta_heatmap(
        heatmap_rows,
        deltas_by_condition,
        args.output_dir / "per_seed_score_delta_heatmap",
        args.width,
        args.row_height,
        None if args.max_heatmap_seeds == 0 else args.max_heatmap_seeds,
    )
    write_markdown_report(
        report_md,
        rows_by_score_drop,
        args.output_dir,
        args.summary_csv,
        args.raw_csv if raw_rows else None,
        analysis_csv,
        args.top_n,
    )

    print(f"Analyzed {len(rows)} features/conditions")
    print(f"Wrote combined CSV: {analysis_csv}")
    print(f"Wrote report: {report_md}")
    print(f"Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

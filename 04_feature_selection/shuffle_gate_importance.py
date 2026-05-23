"""
ShuffleGate-style feature sensitivity analysis for the trained Tetris DQN.

This is a post-hoc evaluator: it does not retrain gates. For each candidate
placement batch, selected feature columns are shuffled across candidate states,
which preserves the feature's marginal values while breaking its association
with each action. Important features should cause larger score drops when
shuffled.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import math
import random
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from environment import TetrisEnv
from model import DQN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "tetris_dqn.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "shuffle_reports"


@dataclass(frozen=True)
class Condition:
    name: str
    feature_indices: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    seed: int
    score: float
    lines: int
    pieces: int
    truncated: bool


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_feature_groups(feature_names: list[str]) -> OrderedDict[str, tuple[int, ...]]:
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}

    def exact(names: list[str]) -> tuple[int, ...]:
        return tuple(name_to_index[name] for name in names if name in name_to_index)

    def prefix(prefix_: str) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, name in enumerate(feature_names)
            if name.startswith(prefix_)
        )

    groups: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    groups["fill_heights"] = prefix("fill_height_col_")
    groups["height_summary"] = exact([
        "mean_height",
        "height_deviation",
        "lowest_point",
        "highest_point",
    ])
    groups["total_holeyness"] = exact(["total_holeyness"])
    groups["holeyness_by_column"] = prefix("holeyness_col_")
    groups["vertical_hole_depths"] = prefix("vertical_hole_depth_col_")
    groups["vertical_hole_clusteredness"] = prefix("vertical_hole_clusteredness_col_")
    groups["horizontal_hole_edge_distances"] = prefix("horizontal_hole_edge_distance_row_")
    groups["horizontal_hole_clusteredness"] = prefix("horizontal_hole_clusteredness_row_")
    groups["hole_summary"] = exact([
        "mean_hole_depth",
        "mean_hole_vertical_clusteredness",
        "hole_depth_deviation",
        "hole_vertical_instability",
        "mean_hole_edge_distance",
        "mean_hole_horizontal_clusteredness",
        "hole_edge_distance_deviation",
        "hole_horizontal_instability",
        "mean_hole_distance",
        "general_hole_clusteredness",
    ])

    all_height = groups["fill_heights"] + groups["height_summary"]
    all_holes = (
        groups["total_holeyness"]
        + groups["holeyness_by_column"]
        + groups["vertical_hole_depths"]
        + groups["vertical_hole_clusteredness"]
        + groups["horizontal_hole_edge_distances"]
        + groups["horizontal_hole_clusteredness"]
        + groups["hole_summary"]
    )
    groups["all_height_features"] = all_height
    groups["all_hole_features"] = all_holes
    groups["all_features"] = tuple(range(len(feature_names)))

    return groups


def dedupe_indices(indices: list[int]) -> tuple[int, ...]:
    seen = set()
    deduped = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            deduped.append(index)
    return tuple(deduped)


def resolve_token(
    token: str,
    feature_names: list[str],
    groups: OrderedDict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    token = token.strip()
    if not token:
        return tuple()

    if token in groups:
        return groups[token]

    if token.isdigit():
        index = int(token)
        if 0 <= index < len(feature_names):
            return (index,)
        raise ValueError(f"Feature index out of range: {token}")

    if token in feature_names:
        return (feature_names.index(token),)

    if any(ch in token for ch in "*?[]"):
        matches = [
            idx
            for idx, name in enumerate(feature_names)
            if fnmatch.fnmatch(name, token)
        ]
        if matches:
            return tuple(matches)

    raise ValueError(f"Unknown feature, group, index, or glob: {token}")


def parse_condition_spec(
    spec: str,
    feature_names: list[str],
    groups: OrderedDict[str, tuple[int, ...]],
) -> Condition:
    indices: list[int] = []
    for token in spec.split(","):
        indices.extend(resolve_token(token, feature_names, groups))

    feature_indices = dedupe_indices(indices)
    if not feature_indices:
        raise ValueError(f"Condition has no features: {spec}")

    return Condition(name=spec, feature_indices=feature_indices, kind="custom")


def default_conditions(
    selection: str,
    feature_names: list[str],
    groups: OrderedDict[str, tuple[int, ...]],
) -> list[Condition]:
    conditions: list[Condition] = []

    if selection in {"individual", "all"}:
        conditions.extend(
            Condition(name=name, feature_indices=(idx,), kind="individual")
            for idx, name in enumerate(feature_names)
        )

    if selection in {"groups", "all"}:
        conditions.extend(
            Condition(name=name, feature_indices=indices, kind="group")
            for name, indices in groups.items()
            if indices
        )

    return conditions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(base_seed: int, condition_name: str, episode_seed: int) -> int:
    digest = hashlib.sha256(condition_name.encode("utf-8")).digest()
    condition_offset = int.from_bytes(digest[:4], "little")
    return (base_seed * 1_000_003 + episode_seed * 97 + condition_offset) % (2**32)


def shuffle_feature_columns(
    states: np.ndarray,
    feature_indices: tuple[int, ...],
    rng: np.random.Generator,
    shuffle_mode: str,
) -> np.ndarray:
    if not feature_indices or states.shape[0] <= 1:
        return states

    shuffled = states.copy()

    if shuffle_mode == "joint":
        permutation = rng.permutation(states.shape[0])
        shuffled[:, feature_indices] = shuffled[permutation][:, feature_indices]
        return shuffled

    for index in feature_indices:
        shuffled[:, index] = shuffled[rng.permutation(states.shape[0]), index]

    return shuffled


def evaluate_episode(
    model: DQN,
    env: TetrisEnv,
    device: torch.device,
    episode_seed: int,
    max_pieces: int,
    feature_indices: tuple[int, ...] = tuple(),
    shuffle_seed: int | None = None,
    shuffle_mode: str = "independent",
) -> EpisodeResult:
    seed_everything(episode_seed)
    rng = np.random.default_rng(shuffle_seed)

    env.reset()
    env.spawn_piece()
    done = False
    pieces = 0
    info = {"score": env.score, "lines_cleared": env.lines_cleared}

    while not done and pieces < max_pieces:
        actions = env.get_possible_actions()
        if not actions:
            break

        states = []
        valid_actions = []
        for action in actions:
            state = env.get_state_for_action(action)
            if state is not None:
                states.append(state)
                valid_actions.append(action)

        if not valid_actions:
            break

        states_array = np.array(states, dtype=np.float32)
        if feature_indices:
            states_array = shuffle_feature_columns(
                states_array,
                feature_indices,
                rng,
                shuffle_mode,
            )

        with torch.no_grad():
            states_tensor = torch.tensor(states_array, dtype=torch.float32, device=device)
            q_values = model(states_tensor).squeeze(1)
            best_idx = torch.argmax(q_values).item()

        _, _, done, info = env.step(valid_actions[best_idx])
        pieces += 1

    return EpisodeResult(
        condition="baseline" if not feature_indices else "",
        seed=episode_seed,
        score=float(info.get("score", env.score)),
        lines=int(info.get("lines_cleared", env.lines_cleared)),
        pieces=pieces,
        truncated=pieces >= max_pieces and not done,
    )


def summarize_condition(
    condition: Condition,
    results: list[EpisodeResult],
    baseline_by_seed: dict[int, EpisodeResult],
    feature_names: list[str],
) -> dict[str, object]:
    scores = [result.score for result in results]
    lines = [float(result.lines) for result in results]
    pieces = [float(result.pieces) for result in results]
    baseline_scores = [baseline_by_seed[result.seed].score for result in results]
    baseline_lines = [float(baseline_by_seed[result.seed].lines) for result in results]
    paired_score_deltas = [
        baseline_by_seed[result.seed].score - result.score
        for result in results
    ]
    paired_line_deltas = [
        float(baseline_by_seed[result.seed].lines - result.lines)
        for result in results
    ]
    baseline_score_mean = mean(baseline_scores)
    score_delta = mean(paired_score_deltas)
    score_delta_pct = (
        100.0 * score_delta / abs(baseline_score_mean)
        if abs(baseline_score_mean) > 1e-9
        else 0.0
    )

    return {
        "condition": condition.name,
        "kind": condition.kind,
        "n_features": len(condition.feature_indices),
        "feature_indices": ";".join(str(index) for index in condition.feature_indices),
        "feature_names": ";".join(feature_names[index] for index in condition.feature_indices),
        "runs": len(results),
        "mean_score": mean(scores),
        "std_score": std(scores),
        "median_score": float(statistics.median(scores)) if scores else 0.0,
        "mean_lines": mean(lines),
        "std_lines": std(lines),
        "mean_pieces": mean(pieces),
        "truncated_runs": sum(1 for result in results if result.truncated),
        "baseline_mean_score": baseline_score_mean,
        "baseline_mean_lines": mean(baseline_lines),
        "mean_score_delta": score_delta,
        "std_score_delta": std(paired_score_deltas),
        "mean_score_delta_pct": score_delta_pct,
        "mean_lines_delta": mean(paired_line_deltas),
        "std_lines_delta": std(paired_line_deltas),
    }


def write_raw_csv(path: Path, results: list[EpisodeResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "score",
                "lines",
                "pieces",
                "truncated",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "condition": result.condition,
                "seed": result.seed,
                "score": result.score,
                "lines": result.lines,
                "pieces": result.pieces,
                "truncated": result.truncated,
            })


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "condition",
        "kind",
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
        "feature_indices",
        "feature_names",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, summary in enumerate(summaries, start=1):
            row = {"rank": rank}
            row.update(summary)
            writer.writerow(row)


def fmt(value: object, precision: int = 2) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown_report(
    path: Path,
    summaries: list[dict[str, object]],
    baseline_summary: dict[str, object],
    args: argparse.Namespace,
    feature_names: list[str],
    groups: OrderedDict[str, tuple[int, ...]],
    csv_path: Path,
    raw_path: Path,
    plot_path: Path | None,
) -> None:
    top_n = min(args.top_n, len(summaries))
    lines = [
        "# ShuffleGate Feature Importance Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: `{Path(args.model_path).resolve()}`",
        f"Runs per condition: `{args.runs}`",
        f"Episode seeds: `{args.seed}` to `{args.seed + args.runs - 1}`",
        f"Max pieces per episode: `{args.max_pieces}`",
        f"Shuffle mode: `{args.shuffle_mode}`",
        "",
        "This report uses ShuffleGate-style batch-wise feature shuffling during "
        "action evaluation. Larger positive deltas mean the agent performed worse "
        "when that feature set was shuffled, so the feature set appears more "
        "important to the current trained policy.",
        "",
        "## Baseline",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean score | {fmt(baseline_summary['mean_score'])} |",
        f"| Score std | {fmt(baseline_summary['std_score'])} |",
        f"| Median score | {fmt(baseline_summary['median_score'])} |",
        f"| Mean lines | {fmt(baseline_summary['mean_lines'])} |",
        f"| Mean pieces | {fmt(baseline_summary['mean_pieces'])} |",
        f"| Truncated runs | {baseline_summary['truncated_runs']} |",
        "",
        f"## Top {top_n} Most Sensitive Conditions",
        "",
        "| Rank | Condition | Kind | Features | Score drop | Lines drop | Shuffled mean score |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for rank, summary in enumerate(summaries[:top_n], start=1):
        lines.append(
            "| "
            + " | ".join([
                str(rank),
                f"`{summary['condition']}`",
                str(summary["kind"]),
                str(summary["n_features"]),
                fmt(summary["mean_score_delta"]),
                fmt(summary["mean_lines_delta"]),
                fmt(summary["mean_score"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Least Sensitive Or Helpful When Shuffled",
        "",
        "| Rank | Condition | Kind | Features | Score drop | Lines drop | Shuffled mean score |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ])

    least = list(reversed(summaries[-top_n:]))
    for rank, summary in enumerate(least, start=1):
        lines.append(
            "| "
            + " | ".join([
                str(rank),
                f"`{summary['condition']}`",
                str(summary["kind"]),
                str(summary["n_features"]),
                fmt(summary["mean_score_delta"]),
                fmt(summary["mean_lines_delta"]),
                fmt(summary["mean_score"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Files",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Raw per-run CSV: `{raw_path}`",
    ])
    if plot_path is not None:
        lines.append(f"- Plot: `{plot_path}`")

    lines.extend([
        "",
        "## Feature Groups",
        "",
        "| Group | Feature count |",
        "| --- | ---: |",
    ])
    for name, indices in groups.items():
        lines.append(f"| `{name}` | {len(indices)} |")

    lines.extend([
        "",
        "## Feature Index Map",
        "",
        "| Index | Feature |",
        "| ---: | --- |",
    ])
    for index, name in enumerate(feature_names):
        lines.append(f"| {index} | `{name}` |")

    path.write_text("\n".join(lines) + "\n")


def write_plot(path: Path, summaries: list[dict[str, object]], top_n: int) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not summaries:
        return None

    selected = summaries[: min(top_n, len(summaries))]
    labels = [str(summary["condition"]) for summary in selected]
    values = [float(summary["mean_score_delta"]) for summary in selected]
    errors = [float(summary["std_score_delta"]) for summary in selected]

    height = max(4.0, 0.35 * len(selected))
    plt.figure(figsize=(12, height))
    y_positions = np.arange(len(selected))
    plt.barh(y_positions, values, xerr=errors, color="#2F6F73")
    plt.yticks(y_positions, labels)
    plt.xlabel("Mean paired score drop vs baseline")
    plt.title("ShuffleGate-style Feature Sensitivity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def make_baseline_summary(results: list[EpisodeResult]) -> dict[str, object]:
    scores = [result.score for result in results]
    lines = [float(result.lines) for result in results]
    pieces = [float(result.pieces) for result in results]
    return {
        "mean_score": mean(scores),
        "std_score": std(scores),
        "median_score": float(statistics.median(scores)) if scores else 0.0,
        "mean_lines": mean(lines),
        "std_lines": std(lines),
        "mean_pieces": mean(pieces),
        "truncated_runs": sum(1 for result in results if result.truncated),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ShuffleGate-style feature shuffling against a trained DQN.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=5000)
    parser.add_argument(
        "--selection",
        choices=["individual", "groups", "all"],
        default="groups",
        help="Default conditions to run when --condition is not supplied.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help=(
            "Feature condition to shuffle. Use feature names, indices, group names, "
            "globs, or comma-separated combinations. May be repeated."
        ),
    )
    parser.add_argument(
        "--shuffle-mode",
        choices=["independent", "joint"],
        default="independent",
        help="Independent shuffles each selected column separately; joint uses one permutation for the set.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="shuffle_gate")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.max_pieces <= 0:
        raise ValueError("--max-pieces must be positive")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = TetrisEnv()
    feature_names = list(env.state_feature_names)
    groups = build_feature_groups(feature_names)

    model = DQN(env.state_size).to(device)
    model.load_state_dict(load_state_dict(model_path, device))
    model.eval()

    if args.condition:
        conditions = [
            parse_condition_spec(spec, feature_names, groups)
            for spec in args.condition
        ]
    else:
        conditions = default_conditions(args.selection, feature_names, groups)

    if not conditions:
        raise ValueError("No shuffle conditions selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.output_dir / f"{args.prefix}_summary.csv"
    raw_csv_path = args.output_dir / f"{args.prefix}_raw.csv"
    markdown_path = args.output_dir / f"{args.prefix}_report.md"
    plot_path = args.output_dir / f"{args.prefix}_top.png"

    episode_seeds = [args.seed + run_idx for run_idx in range(args.runs)]
    print(f"Using device: {device}")
    print(f"Features: {len(feature_names)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Runs per condition: {args.runs}")

    baseline_results = []
    for run_idx, episode_seed in enumerate(episode_seeds, start=1):
        result = evaluate_episode(
            model=model,
            env=env,
            device=device,
            episode_seed=episode_seed,
            max_pieces=args.max_pieces,
        )
        baseline_results.append(EpisodeResult(
            condition="baseline",
            seed=result.seed,
            score=result.score,
            lines=result.lines,
            pieces=result.pieces,
            truncated=result.truncated,
        ))
        if run_idx % max(1, args.runs // 10) == 0 or run_idx == args.runs:
            print(f"Baseline {run_idx}/{args.runs}")

    baseline_by_seed = {result.seed: result for result in baseline_results}
    raw_results = list(baseline_results)
    summaries = []

    for condition_idx, condition in enumerate(conditions, start=1):
        condition_results = []
        for run_idx, episode_seed in enumerate(episode_seeds, start=1):
            shuffle_seed = stable_seed(args.seed, condition.name, episode_seed)
            result = evaluate_episode(
                model=model,
                env=env,
                device=device,
                episode_seed=episode_seed,
                max_pieces=args.max_pieces,
                feature_indices=condition.feature_indices,
                shuffle_seed=shuffle_seed,
                shuffle_mode=args.shuffle_mode,
            )
            result = EpisodeResult(
                condition=condition.name,
                seed=result.seed,
                score=result.score,
                lines=result.lines,
                pieces=result.pieces,
                truncated=result.truncated,
            )
            condition_results.append(result)
            raw_results.append(result)

        summary = summarize_condition(
            condition,
            condition_results,
            baseline_by_seed,
            feature_names,
        )
        summaries.append(summary)
        print(
            f"[{condition_idx}/{len(conditions)}] {condition.name}: "
            f"score drop {summary['mean_score_delta']:.2f}, "
            f"lines drop {summary['mean_lines_delta']:.2f}"
        )

    summaries.sort(key=lambda row: float(row["mean_score_delta"]), reverse=True)
    baseline_summary = make_baseline_summary(baseline_results)

    write_summary_csv(summary_csv_path, summaries)
    write_raw_csv(raw_csv_path, raw_results)
    written_plot = write_plot(plot_path, summaries, args.top_n)
    write_markdown_report(
        markdown_path,
        summaries,
        baseline_summary,
        args,
        feature_names,
        groups,
        summary_csv_path,
        raw_csv_path,
        written_plot,
    )

    print("")
    print(f"Baseline mean score: {baseline_summary['mean_score']:.2f}")
    print(f"Best positive score drop: {summaries[0]['condition']} ({summaries[0]['mean_score_delta']:.2f})")
    print(f"Wrote summary CSV: {summary_csv_path}")
    print(f"Wrote raw CSV: {raw_csv_path}")
    print(f"Wrote report: {markdown_path}")
    if written_plot is not None:
        print(f"Wrote plot: {written_plot}")


if __name__ == "__main__":
    main()

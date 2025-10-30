"""Utility for aggregating task decomposition evaluation CSV outputs.

This script expects CSV files that match the schema emitted by
`task_decomposition_evaluation.py`. It validates the columns, combines
all supplied files, and prints a compact textual summary so further
analysis or visualisation steps can be added later.

For evaluation, call: python evaluation/evaluation_summary.py "('results-3_2b.csv', 'llama 3.2 (3.2 B)')" "('results-8b.csv', 'llama 3.1 (8 B)')" "('results-12_2b.csv', 'mistral-nemo (12.2 B)')" "('results-20_9b.csv', 'gpt-oss (20.9 B)')"
"""

import argparse
import ast
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass

EXPECTED_COLUMNS: List[str] = [
    "model_id",
    "rule_count",
    "example_count",
    "scenario",
    "correct_output_format",
    "tool_coverage",
    "edge_coverage",
    "false_tool_count",
    "false_edge_count",
    "dag_graph_edit_distance",
]

SCENARIO_LABELS = {
    1: "XAI (SHAP) Task",
    2: "EDA TASK",
}

TREND_LABELS = {
    "rule_count": "rule count",
    "example_count": "example count",
}

VALUE_LABELS = {
    "dag_graph_edit_distance": "DAG Graph Edit Distance",
    "false_tool_count": "False Tool Count",
    "false_edge_count": "False Dependency Count",
    "correct_output_format": "Correct Output Format",
    "tool_coverage": "Tool Coverage (Recall)",
    "edge_coverage": "Dependency Coverage (Recall)",
    "tool_precision": "Tool Precision",
    "edge_precision": "Dependency Precision",
    "tool_f1": "Tool F1",
    "edge_f1": "Dependency F1",
}

JT_PERMUTATIONS = 2000
JT_RANDOM_SEED = 42


@dataclass
class StratifiedSpearmanResult:
    combined_z_stat: float
    combined_p_two_tailed: float
    one_sided_p_value: float
    total_strata: int
    supports_expected: bool
    expected_direction: str


@dataclass
class MetricPlotConfig:
    column: str
    display_name: str
    base_filename: str
    high_good: bool
    vmin: float
    vmax: float


@dataclass
class TrendMetricConfig:
    column: str
    display_name: str
    output_slug: str
    fillna_value: Optional[float] = None
    expect_negative: bool = True
    min_unique_rules: int = 3
    min_unique_examples: int = 2


def parse_csv_argument(entry: str) -> Tuple[str, Optional[str]]:
    candidate = entry.strip()
    if candidate.startswith("(") and candidate.endswith(")"):
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Unable to interpret CSV argument as tuple <path, model_name>: {entry}"
            ) from exc

        if (
            isinstance(parsed, (tuple, list))
            and len(parsed) == 2
            and all(isinstance(item, str) and item.strip() for item in parsed)
        ):
            csv_path, model_name = parsed
            return csv_path.strip(), model_name.strip()

        raise ValueError(
            "Tuple CSV arguments must provide two non-empty strings: (path, model_name)"
        )

    return candidate, None


def resolve_paths(raw_paths: Iterable[str]) -> List[Path]:
    base = Path(__file__).parent
    resolved: List[Path] = []
    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved.append(candidate.resolve())
    return resolved


def load_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    df = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file {path} is missing expected columns: {', '.join(missing)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    tool_numerator = np.where(
        df["scenario"] == 1,
        2.0 * df["tool_coverage"],
        8.0 * df["tool_coverage"],
    )
    tool_denominator = np.where(
        df["scenario"] == 1,
        2.0 + df["false_tool_count"],
        8.0 + df["false_tool_count"],
    )
    df["tool_precision"] = tool_numerator / tool_denominator

    edge_numerator = np.where(
        df["scenario"] == 1,
        1.0 * df["edge_coverage"],
        4.0 * df["edge_coverage"],
    )
    edge_denominator = np.where(
        df["scenario"] == 1,
        1.0 + df["false_edge_count"],
        4.0 + df["false_edge_count"],
    )
    df["edge_precision"] = edge_numerator / edge_denominator

    with np.errstate(divide="ignore", invalid="ignore"):
        tool_sum = df["tool_precision"] + df["tool_coverage"]
        df["tool_f1"] = np.where(
            tool_sum > 0,
            2.0 * df["tool_precision"] * df["tool_coverage"] / tool_sum,
            0.0,
        )

        edge_sum = df["edge_precision"] + df["edge_coverage"]
        df["edge_f1"] = np.where(
            edge_sum > 0,
            2.0 * df["edge_precision"] * df["edge_coverage"] / edge_sum,
            0.0,
        )

    return df


def attach_model_name(df: pd.DataFrame, model_name: Optional[str]) -> pd.DataFrame:
    labelled = df.copy()
    labelled["model_id"] = labelled["model_id"].astype(str)
    if model_name:
        labelled["model_name"] = model_name
    else:
        labelled["model_name"] = labelled["model_id"]
    return labelled


def summarise_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    return combined


def print_summary_table(df: pd.DataFrame) -> None:
    print("\nLoaded evaluation results summary")
    print("--------------------------------")
    print(f"Rows total          : {len(df)}")
    print(f"Unique models       : {sorted(df['model_name'].unique())}")
    print(f"Rule counts covered : {sorted(df['rule_count'].unique())}")
    print(f"Example counts      : {sorted(df['example_count'].unique())}")
    print("Scenarios           : {}".format(sorted(df["scenario"].unique())))
    print("\nPreview of combined results (first 5 rows):")
    preview_columns = EXPECTED_COLUMNS + [
        "tool_precision",
        "edge_precision",
        "tool_f1",
        "edge_f1",
        "model_name",
    ]
    print(df[preview_columns].head().to_string(index=False))


def plot_metric_heatmaps(
    df: pd.DataFrame,
    model_order: List[str],
    output_dir: Path,
    base_filename: str,
    metric_column: str,
    metric_display_name: str,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
) -> Tuple[List[Path], Optional[Path]]:
    example_counts = sorted(df["example_count"].unique())
    scenarios = sorted(df["scenario"].unique())
    if not example_counts:
        raise ValueError("No rows available for metric visualisations.")

    ordered_models = [name for name in model_order if name in df["model_name"].unique()]

    output_dir.mkdir(parents=True, exist_ok=True)

    pivot_cache: Dict[int, Dict[int, Dict[str, pd.DataFrame]]] = {}
    saved_paths: List[Path] = []
    mask_with_format = metric_column != "correct_output_format"

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]
        if scenario_df.empty:
            continue

        pivot_cache.setdefault(scenario, {})

        for example_count in example_counts:
            filtered = scenario_df[scenario_df["example_count"] == example_count].copy()
            if filtered.empty:
                continue

            filtered[metric_column] = pd.to_numeric(filtered[metric_column], errors="coerce")
            filtered["model_name"] = pd.Categorical(
                filtered["model_name"], categories=ordered_models, ordered=True
            )
            filtered.sort_values(["model_name", "rule_count"], inplace=True)

            pivot = filtered.pivot_table(
                index="rule_count",
                columns="model_name",
                values=metric_column,
                aggfunc="mean",
            )

            format_pivot = filtered.pivot_table(
                index="rule_count",
                columns="model_name",
                values="correct_output_format",
                aggfunc="mean",
            )

            if mask_with_format:
                pivot = pivot.where(format_pivot == 1)

            if pivot.empty:
                continue

            pivot_cache[scenario][example_count] = {"metric": pivot, "format": format_pivot}

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_xlabel("Model")

            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_ylabel("Rule Count")

            label = SCENARIO_LABELS.get(scenario, f"Scenario {scenario}")
            ax.set_title(
                f"{metric_display_name} | {label} | {example_count} Examples"
            )

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if mask_with_format:
                        allowed = format_pivot.iloc[i, j] == 1
                    else:
                        allowed = True
                    value = pivot.iloc[i, j]
                    text = "NA" if not allowed or pd.isna(value) else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

            fig.colorbar(im, ax=ax, label=metric_display_name)
            fig.tight_layout()

            output_path = output_dir / f"{base_filename}_sc_{scenario}_ex_{example_count}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)

            saved_paths.append(output_path)

    grid_path = plot_metric_grid(
        pivot_cache,
        example_counts,
        cmap,
        vmin,
        vmax,
        metric_display_name,
        base_filename,
        output_dir,
        mask_with_format,
    )

    return saved_paths, grid_path


def plot_metric_grid(
    pivot_cache: Dict[int, Dict[int, Dict[str, pd.DataFrame]]],
    example_counts: List[int],
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
    metric_display_name: str,
    base_filename: str,
    output_dir: Path,
    mask_with_format: bool,
) -> Optional[Path]:
    scenario_order = [2, 1]
    scenario_titles = [SCENARIO_LABELS.get(sc, f"Scenario {sc}") for sc in scenario_order]
    example_order = [0, 1, 2, 3]

    fig, axes = plt.subplots(len(example_order), len(scenario_order), figsize=(4.2 * len(scenario_order), 3.5 * len(example_order)))
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])

    last_im = None

    for row_idx, example_count in enumerate(example_order):
        row_axes = axes[row_idx]
        if example_count not in example_counts:
            for ax in np.atleast_1d(row_axes):
                ax.axis("off")
            continue

        for col_idx, scenario in enumerate(scenario_order):
            ax = row_axes[col_idx]
            cached = pivot_cache.get(scenario, {}).get(example_count)
            if cached is None:
                ax.axis("off")
                continue

            pivot = cached.get("metric")
            format_pivot = cached.get("format")
            if pivot is None or pivot.empty:
                ax.axis("off")
                continue

            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            last_im = im

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
            if row_idx == len(example_order) - 1:
                ax.set_xlabel("Model")
            else:
                ax.set_xlabel("")

            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{example_count} Examples", fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == 0:
                ax.set_title(scenario_titles[col_idx], fontsize=12)

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    allowed = True
                    if mask_with_format and format_pivot is not None:
                        allowed = format_pivot.iloc[i, j] == 1
                    value = pivot.iloc[i, j]
                    text = "NA" if not allowed or pd.isna(value) else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    if last_im is None:
        plt.close(fig)
        return None

    fig.suptitle(f"{metric_display_name} Overview", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel(metric_display_name)

    output_path = output_dir / f"{base_filename}_grid.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_pastel_cmap(high_good: bool) -> mcolors.LinearSegmentedColormap:
    base_colors = ["#f18c8e", "#f6c667", "#4fb79a"]  # red → amber → teal
    if not high_good:
        base_colors = list(reversed(base_colors))
    return mcolors.LinearSegmentedColormap.from_list("balanced_pastel", base_colors, N=256)


def compute_stratified_spearman(
    df: pd.DataFrame,
    group_cols: List[str],
    x_col: str,
    y_col: str,
    min_unique_x: int = 3,
    expect_negative: bool = True,
) -> Optional[StratifiedSpearmanResult]:
    from scipy import stats

    strata_effects = []
    grouped = df.dropna(subset=[y_col]).groupby(group_cols)

    for _, group in grouped:
        n = len(group)
        if n < 4:
            continue

        if group[x_col].nunique() < min_unique_x:
            continue
        if group[y_col].nunique() <= 1:
            continue

        rho, _ = stats.spearmanr(group[x_col], group[y_col])
        if np.isnan(rho):
            continue

        rho = float(np.clip(rho, -0.999999, 0.999999))
        fisher_z = np.arctanh(rho)
        variance = 1.0 / (n - 3)
        weight = 1.0 / variance

        strata_effects.append((fisher_z, weight))

    if not strata_effects:
        return None

    weights = np.array([weight for (_, weight) in strata_effects])
    fisher_values = np.array([fisher for (fisher, _) in strata_effects])

    weighted_mean_fisher = np.sum(weights * fisher_values) / np.sum(weights)
    combined_variance = 1.0 / np.sum(weights)
    combined_std = math.sqrt(combined_variance)
    combined_z_stat = weighted_mean_fisher / combined_std

    combined_p_two_tailed = 2 * stats.norm.sf(abs(combined_z_stat))

    if expect_negative:
        one_sided_p_value = stats.norm.cdf(combined_z_stat)
        supports_expected = combined_z_stat < 0 and one_sided_p_value < 0.05
        expected_direction = "negative"
    else:
        one_sided_p_value = stats.norm.sf(combined_z_stat)
        supports_expected = combined_z_stat > 0 and one_sided_p_value < 0.05
        expected_direction = "positive"

    return StratifiedSpearmanResult(
        combined_z_stat=combined_z_stat,
        combined_p_two_tailed=combined_p_two_tailed,
        one_sided_p_value=one_sided_p_value,
        total_strata=len(strata_effects),
        supports_expected=supports_expected,
        expected_direction=expected_direction,
    )


def jonckheere_statistic(samples: List[np.ndarray]) -> float:
    statistic = 0.0
    for i in range(len(samples) - 1):
        current = samples[i]
        if current.size == 0:
            continue
        for j in range(i + 1, len(samples)):
            comparison = samples[j]
            if comparison.size == 0:
                continue
            diff = np.subtract.outer(comparison, current)
            statistic += float(np.sum(diff > 0))
            statistic += 0.5 * float(np.sum(diff == 0))
    return statistic


def jonckheere_permutation_pvalue(
    samples: List[np.ndarray],
    observed_statistic: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    if not samples:
        return float("nan")

    sizes = [sample.size for sample in samples]
    if any(size == 0 for size in sizes):
        return float("nan")

    combined = np.concatenate(samples)
    exceedances = 0
    for _ in range(permutations):
        shuffled = np.array(combined, copy=True)
        rng.shuffle(shuffled)
        start = 0
        permuted: List[np.ndarray] = []
        for size in sizes:
            permuted.append(shuffled[start : start + size])
            start += size
        perm_statistic = jonckheere_statistic(permuted)
        if perm_statistic <= observed_statistic + 1e-12:
            exceedances += 1
    return (exceedances + 1) / (permutations + 1)


def build_jonckheere_trend_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    ordering_col: str,
    value_col: str,
    min_levels: int = 2,
    expect_negative: bool = True,
) -> Optional[pd.DataFrame]:
    filtered = df.dropna(subset=[value_col])
    if filtered.empty:
        return None

    direction_word = "decreasing" if expect_negative else "increasing"

    rows: List[Dict[str, object]] = []
    grouped = filtered.groupby(group_cols)
    rng = np.random.default_rng(JT_RANDOM_SEED)

    for keys, group in grouped:
        ordered = group.groupby(ordering_col)[value_col].apply(list).sort_index()
        if len(ordered) < min_levels:
            continue

        samples = [np.array(values, dtype=float) for values in ordered]
        if any(sample.size == 0 for sample in samples):
            continue

        combined = np.concatenate(samples)
        if np.allclose(combined, combined[0]):
            continue

        if not expect_negative:
            samples_for_test = [(-1.0) * sample for sample in samples]
        else:
            samples_for_test = samples

        statistic = jonckheere_statistic(samples_for_test)
        p_value = jonckheere_permutation_pvalue(
            samples_for_test, statistic, JT_PERMUTATIONS, rng
        )
        if np.isnan(p_value):
            continue

        supports_expected = p_value < 0.05
        finding = (
            f"Evidence for {direction_word} {VALUE_LABELS.get(value_col, value_col)} with higher {TREND_LABELS.get(ordering_col, ordering_col)} (p={p_value:.3g})."
            if supports_expected
            else f"No significant monotonic {direction_word} trend detected (p={p_value:.3g})."
        )

        summary_row: Dict[str, object] = {
            "ordering_levels": ",".join(str(level) for level in ordered.index.tolist()),
            "total_samples": int(sum(sample.size for sample in samples)),
            "jt_statistic": statistic,
            "jt_pvalue": p_value,
            "supports_expected": supports_expected,
            "expected_direction": direction_word,
            "finding": finding,
            "trend_variable": ordering_col,
            "trend_label": TREND_LABELS.get(ordering_col, ordering_col),
            "value_variable": value_col,
            "value_label": VALUE_LABELS.get(value_col, value_col),
        }

        if len(group_cols) == 1:
            summary_row[group_cols[0]] = keys
        else:
            for col_name, key_value in zip(group_cols, keys):
                summary_row[col_name] = key_value

        rows.append(summary_row)

    if not rows:
        return None

    summary = pd.DataFrame(rows)
    sort_cols = [col for col in group_cols if col in summary.columns]
    if ordering_col in summary.columns:
        sort_cols.append(ordering_col)
    summary.sort_values(sort_cols, inplace=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 1-4 task decomposition evaluation CSV files",
        epilog=(
            "CSV arguments can optionally be provided as tuples to declare a "
            "display name, e.g. (\"results-3_2b.csv\", \"EINHORN-3.2B\")."
        ),
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Paths (or tuple strings) to CSV files produced by task_decomposition_evaluation.py",
    )
    parser.add_argument(
        "--metric-output-dir",
        type=str,
        default="",
        help="Directory to save metric heatmaps.",
    )
    parser.add_argument(
        "--tool-coverage-base",
        type=str,
        default="tool_coverage",
        help="Base filename prefix for tool coverage heatmaps.",
    )
    parser.add_argument(
        "--correct-output-base",
        type=str,
        default="correct_output_format",
        help="Base filename prefix for correct_output_format heatmaps.",
    )
    parser.add_argument(
        "--edge-coverage-base",
        type=str,
        default="edge_coverage",
        help="Base filename prefix for edge_coverage heatmaps.",
    )
    parser.add_argument(
        "--tool-precision-base",
        type=str,
        default="tool_precision",
        help="Base filename prefix for tool_precision heatmaps.",
    )
    parser.add_argument(
        "--edge-precision-base",
        type=str,
        default="edge_precision",
        help="Base filename prefix for edge_precision heatmaps.",
    )
    parser.add_argument(
        "--tool-f1-base",
        type=str,
        default="tool_f1",
        help="Base filename prefix for tool F1 heatmaps.",
    )
    parser.add_argument(
        "--edge-f1-base",
        type=str,
        default="edge_f1",
        help="Base filename prefix for dependency F1 heatmaps.",
    )
    parser.add_argument(
        "--false-tool-base",
        type=str,
        default="false_tool_count",
        help="Base filename prefix for false_tool_count heatmaps.",
    )
    parser.add_argument(
        "--false-edge-base",
        type=str,
        default="false_edge_count",
        help="Base filename prefix for false_edge_count heatmaps.",
    )
    parser.add_argument(
        "--dag-ged-base",
        type=str,
        default="dag_graph_edit_distance",
        help="Base filename prefix for DAG graph edit distance heatmaps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_specs = [parse_csv_argument(entry) for entry in args.csv_files]

    if not (1 <= len(csv_specs) <= 4):
        raise ValueError("Please supply between 1 and 4 evaluation CSV files.")

    raw_paths = [spec[0] for spec in csv_specs]
    resolved_paths = resolve_paths(raw_paths)

    frames: List[pd.DataFrame] = []
    model_order: List[str] = []
    for (raw_path, model_name), resolved in zip(csv_specs, resolved_paths):
        df = load_results_csv(resolved)
        labelled = attach_model_name(df, model_name)
        frames.append(labelled)
        display_name = labelled["model_name"].iat[0]
        if display_name not in model_order:
            model_order.append(display_name)

    combined = summarise_frames(frames)
    print_summary_table(combined)

    trend_metrics = [
        TrendMetricConfig(
            "dag_graph_edit_distance",
            "DAG Graph Edit Distance",
            "dag_ged",
            fillna_value=10.0,
            expect_negative=True,
            min_unique_rules=3,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "false_tool_count",
            "False Tool Count",
            "false_tool",
            expect_negative=True,
            min_unique_rules=3,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "false_edge_count",
            "False Dependency Count",
            "false_edge",
            expect_negative=True,
            min_unique_rules=3,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "correct_output_format",
            "Correct Output Format",
            "correct_output",
            expect_negative=False,
            min_unique_rules=2,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "tool_coverage",
            "Tool Coverage",
            "tool_coverage",
            expect_negative=False,
            min_unique_rules=2,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "edge_coverage",
            "Dependency Coverage",
            "dependency_coverage",
            expect_negative=False,
            min_unique_rules=2,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "tool_precision",
            "Tool Precision",
            "tool_precision",
            fillna_value=0.0,
            expect_negative=False,
            min_unique_rules=2,
            min_unique_examples=2,
        ),
        TrendMetricConfig(
            "edge_precision",
            "Dependency Precision",
            "edge_precision",
            fillna_value=0.0,
            expect_negative=False,
            min_unique_rules=2,
            min_unique_examples=2,
        ),
    ]

    metric_configs = [
        MetricPlotConfig(
            "tool_coverage",
            "Tool Coverage (Recall)",
            args.tool_coverage_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "correct_output_format",
            "Correct Output Format",
            args.correct_output_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "edge_coverage",
            "Dependency Coverage (Recall)",
            args.edge_coverage_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "tool_precision",
            "Tool Precision",
            args.tool_precision_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "edge_precision",
            "Dependency Precision",
            args.edge_precision_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "tool_f1",
            "Tool F1",
            args.tool_f1_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "edge_f1",
            "Dependency F1",
            args.edge_f1_base,
            True,
            0.0,
            1.0,
        ),
        MetricPlotConfig(
            "false_tool_count",
            "False Tool Count",
            args.false_tool_base,
            False,
            0.0,
            4.0,
        ),
        MetricPlotConfig(
            "false_edge_count",
            "False Dependency Count",
            args.false_edge_base,
            False,
            0.0,
            4.0,
        ),
        MetricPlotConfig(
            "dag_graph_edit_distance",
            "DAG Graph Edit Distance",
            args.dag_ged_base,
            False,
            0.0,
            4.0,
        ),
    ]

    metric_dir = Path(args.metric_output_dir) if args.metric_output_dir else Path("evaluation visualizations")
    if not metric_dir.is_absolute():
        metric_dir = Path(__file__).parent / metric_dir
    metric_dir.mkdir(parents=True, exist_ok=True)

    for config in metric_configs:
        cmap = build_pastel_cmap(high_good=config.high_good)
        saved_paths, grid_path = plot_metric_heatmaps(
            combined,
            model_order,
            metric_dir,
            config.base_filename,
            config.column,
            config.display_name,
            cmap,
            config.vmin,
            config.vmax,
        )

        if saved_paths:
            print(f"\n{config.display_name} heatmaps saved:")
            for path in saved_paths:
                print(f"  - {path}")
        else:
            print(f"\nNo {config.display_name} heatmaps were generated (no matching data).")

        if grid_path:
            print(f"  Grid overview: {grid_path}")

    results_output_dir = metric_dir

    stats_output_dir = Path("statistical evaluation results")
    if not stats_output_dir.is_absolute():
        stats_output_dir = Path(__file__).parent / stats_output_dir
    stats_output_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: List[Dict[str, object]] = []
    spearman_rows: List[Dict[str, object]] = []

    for trend_config in trend_metrics:
        working_df = combined.copy()
        if trend_config.fillna_value is not None:
            working_df[trend_config.column] = working_df[trend_config.column].fillna(trend_config.fillna_value)

        rule_meta_result = compute_stratified_spearman(
            working_df,
            group_cols=["scenario", "example_count", "model_id"],
            x_col="rule_count",
            y_col=trend_config.column,
            min_unique_x=trend_config.min_unique_rules,
            expect_negative=trend_config.expect_negative,
        )

        label = trend_config.display_name
        metric_column = trend_config.column

        direction_word = "decreasing" if trend_config.expect_negative else "increasing"

        if rule_meta_result is not None:
            direction_text = (
                f"Combined correlation supports {direction_word} {label} with more rules"
                if rule_meta_result.supports_expected
                else f"No significant evidence of {direction_word} {label} with more rules"
            )
        else:
            direction_text = (
                f"Insufficient data for stratified Spearman analysis (rule count vs. {label})"
            )

        spearman_entry = {
            "metric": metric_column,
            "metric_display": label,
            "test": "spearman_meta",
            "trend": "rule_count",
            "aspect": f"rule_count_vs_{metric_column}",
            "total_strata": getattr(rule_meta_result, "total_strata", None),
            "combined_z_stat": getattr(rule_meta_result, "combined_z_stat", None),
            "two_tailed_p": getattr(rule_meta_result, "combined_p_two_tailed", None),
            "one_sided_p_value": getattr(rule_meta_result, "one_sided_p_value", None),
            "supports_expected": getattr(rule_meta_result, "supports_expected", None),
            "expected_direction": direction_word,
            "finding": direction_text,
        }
        stats_rows.append(spearman_entry)
        spearman_rows.append({**spearman_entry, "metric_slug": trend_config.output_slug})

        if rule_meta_result is not None:
            print(
                f"\nStratified Spearman correlation analysis (rule_count vs. {label}):"
            )
            print(
                "  Total strata analysed : {total}\n"
                "  Combined z-stat        : {z:.4f}\n"
                "  Two-tailed p-value     : {p:.4g}\n"
                "  One-sided p-value     : {pn:.4g}\n"
                "  Supports {direction_word} trend? : {support}".format(
                    total=rule_meta_result.total_strata,
                    z=rule_meta_result.combined_z_stat,
                    p=rule_meta_result.combined_p_two_tailed,
                    pn=rule_meta_result.one_sided_p_value,
                    direction_word=direction_word,
                    support="yes" if rule_meta_result.supports_expected else "no",
                )
            )
        else:
            print(
                f"\nStratified Spearman correlation analysis (rule_count vs. {label}): not enough data"
            )
            fallback_entry = {
                "metric": metric_column,
                "metric_display": label,
                "test": "spearman_meta",
                "trend": "rule_count",
                "aspect": f"rule_count_vs_{metric_column}",
                "total_strata": 0,
                "combined_z_stat": None,
                "two_tailed_p": None,
                "one_sided_p_value": None,
                "supports_expected": False,
                "expected_direction": direction_word,
                "finding": (
                    f"Insufficient data for stratified Spearman analysis (rule count vs. {label})"
                ),
            }
            stats_rows.append(fallback_entry)
            spearman_rows.append({**fallback_entry, "metric_slug": trend_config.output_slug})

        example_meta_result = compute_stratified_spearman(
            working_df,
            group_cols=["scenario", "rule_count", "model_id"],
            x_col="example_count",
            y_col=trend_config.column,
            min_unique_x=trend_config.min_unique_examples,
            expect_negative=trend_config.expect_negative,
        )

        if example_meta_result is not None:
            direction_text_example = (
                f"Combined correlation supports {direction_word} {label} with more examples"
                if example_meta_result.supports_expected
                else f"No significant evidence of {direction_word} {label} with more examples"
            )
        else:
            direction_text_example = (
                f"Insufficient data for stratified Spearman analysis (example count vs. {label})"
            )

        spearman_example_entry = {
            "metric": metric_column,
            "metric_display": label,
            "test": "spearman_meta",
            "trend": "example_count",
            "aspect": f"example_count_vs_{metric_column}",
            "total_strata": getattr(example_meta_result, "total_strata", None),
            "combined_z_stat": getattr(example_meta_result, "combined_z_stat", None),
            "two_tailed_p": getattr(example_meta_result, "combined_p_two_tailed", None),
            "one_sided_p_value": getattr(example_meta_result, "one_sided_p_value", None),
            "supports_expected": getattr(example_meta_result, "supports_expected", None),
            "expected_direction": direction_word,
            "finding": direction_text_example,
        }
        stats_rows.append(spearman_example_entry)
        spearman_rows.append({**spearman_example_entry, "metric_slug": trend_config.output_slug})

        if example_meta_result is not None:
            print(
                f"\nStratified Spearman correlation analysis (example_count vs. {label}):"
            )
            print(
                "  Total strata analysed : {total}\n"
                "  Combined z-stat        : {z:.4f}\n"
                "  Two-tailed p-value     : {p:.4g}\n"
                "  One-sided p-value     : {pn:.4g}\n"
                "  Supports {direction_word} trend? : {support}".format(
                    total=example_meta_result.total_strata,
                    z=example_meta_result.combined_z_stat,
                    p=example_meta_result.combined_p_two_tailed,
                    pn=example_meta_result.one_sided_p_value,
                    direction_word=direction_word,
                    support="yes" if example_meta_result.supports_expected else "no",
                )
            )
        else:
            print(
                f"\nStratified Spearman correlation analysis (example_count vs. {label}): not enough data"
            )
            fallback_example_entry = {
                "metric": metric_column,
                "metric_display": label,
                "test": "spearman_meta",
                "trend": "example_count",
                "aspect": f"example_count_vs_{metric_column}",
                "total_strata": 0,
                "combined_z_stat": None,
                "two_tailed_p": None,
                "one_sided_p_value": None,
                "supports_expected": False,
                "expected_direction": direction_word,
                "finding": (
                    f"Insufficient data for stratified Spearman analysis (example count vs. {label})"
                ),
            }
            stats_rows.append(fallback_example_entry)
            spearman_rows.append({**fallback_example_entry, "metric_slug": trend_config.output_slug})

        jt_summary_rules = build_jonckheere_trend_summary(
            working_df,
            group_cols=["scenario", "example_count", "model_id", "model_name"],
            ordering_col="rule_count",
            value_col=trend_config.column,
            expect_negative=trend_config.expect_negative,
        )

        jt_summary_example = build_jonckheere_trend_summary(
            working_df,
            group_cols=["scenario", "rule_count", "model_id", "model_name"],
            ordering_col="example_count",
            value_col=trend_config.column,
            min_levels=trend_config.min_unique_examples,
            expect_negative=trend_config.expect_negative,
        )

        if jt_summary_rules is not None:
            jt_rules_csv = (
                stats_output_dir
                / f"jonckheere_trend_rules_summary_{trend_config.output_slug}.csv"
            )
            jt_summary_rules.to_csv(jt_rules_csv, index=False)
            print(
                f"Rule-count trend summary for {label} saved to {jt_rules_csv}"
            )

            display_rules = jt_summary_rules.copy()
            display_rules["scenario_label"] = display_rules["scenario"].map(
                SCENARIO_LABELS
            )
            expected_word = "decreasing" if trend_config.expect_negative else "increasing"
            print(
                f"\nJonckheere–Terpstra monotonic trend test (rule_count vs. {label}, {expected_word} hypothesis):"
            )
            print(
                display_rules[
                    [
                        "scenario_label",
                        "example_count",
                        "model_name",
                        "jt_statistic",
                        "jt_pvalue",
                        "finding",
                    ]
                ].to_string(index=False)
            )

            for _, row in jt_summary_rules.iterrows():
                stats_rows.append(
                    {
                        "metric": metric_column,
                        "metric_display": label,
                        "test": "jonckheere_terpstra",
                        "trend": "rule_count",
                        "aspect": f"rule_count_vs_{metric_column}",
                        "scenario": row["scenario"],
                        "example_count": row["example_count"],
                        "model_id": row["model_id"],
                        "model_name": row["model_name"],
                        "jt_statistic": row["jt_statistic"],
                        "jt_pvalue": row["jt_pvalue"],
                        "supports_expected": row["supports_expected"],
                        "expected_direction": row["expected_direction"],
                        "finding": row["finding"],
                    }
                )
        else:
            print(
                f"\nJonckheere–Terpstra monotonic trend test (rule_count vs. {label}): not enough variation for testing"
            )
            stats_rows.append(
                {
                    "metric": metric_column,
                    "metric_display": label,
                    "test": "jonckheere_terpstra",
                    "trend": "rule_count",
                    "aspect": f"rule_count_vs_{metric_column}",
                    "finding": (
                        f"Insufficient variation for Jonckheere–Terpstra test (rule count vs. {label})"
                    ),
                    "expected_direction": "decreasing" if trend_config.expect_negative else "increasing",
                }
            )

        if jt_summary_example is not None:
            jt_example_csv = (
                stats_output_dir
                / f"jonckheere_trend_examples_summary_{trend_config.output_slug}.csv"
            )
            jt_summary_example.to_csv(jt_example_csv, index=False)
            print(
                f"Example-count trend summary for {label} saved to {jt_example_csv}"
            )

            display_example = jt_summary_example.copy()
            display_example["scenario_label"] = display_example["scenario"].map(
                SCENARIO_LABELS
            )
            expected_word = "decreasing" if trend_config.expect_negative else "increasing"
            print(
                f"\nJonckheere–Terpstra monotonic trend test (example_count vs. {label}, {expected_word} hypothesis):"
            )
            print(
                display_example[
                    [
                        "scenario_label",
                        "rule_count",
                        "model_name",
                        "jt_statistic",
                        "jt_pvalue",
                        "finding",
                    ]
                ].to_string(index=False)
            )

            for _, row in jt_summary_example.iterrows():
                stats_rows.append(
                    {
                        "metric": metric_column,
                        "metric_display": label,
                        "test": "jonckheere_terpstra",
                        "trend": "example_count",
                        "aspect": f"example_count_vs_{metric_column}",
                        "scenario": row["scenario"],
                        "rule_count": row["rule_count"],
                        "model_id": row["model_id"],
                        "model_name": row["model_name"],
                        "jt_statistic": row["jt_statistic"],
                        "jt_pvalue": row["jt_pvalue"],
                        "supports_expected": row["supports_expected"],
                        "expected_direction": row["expected_direction"],
                        "finding": row["finding"],
                    }
                )
        else:
            print(
                f"\nJonckheere–Terpstra monotonic trend test (example_count vs. {label}): not enough variation for testing"
            )
            stats_rows.append(
                {
                    "metric": metric_column,
                    "metric_display": label,
                    "test": "jonckheere_terpstra",
                    "trend": "example_count",
                    "aspect": f"example_count_vs_{metric_column}",
                    "finding": (
                        f"Insufficient variation for Jonckheere–Terpstra test (example count vs. {label})"
                    ),
                    "expected_direction": "decreasing" if trend_config.expect_negative else "increasing",
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    combined_stats_path = stats_output_dir / "statistical_trend_summary.csv"
    stats_df.to_csv(combined_stats_path, index=False)
    print(f"Statistical summaries saved to {combined_stats_path}")

    if spearman_rows:
        spearman_df = pd.DataFrame(spearman_rows)
        spearman_csv_path = stats_output_dir / "spearman_trend_summary.csv"
        spearman_df.to_csv(spearman_csv_path, index=False)
        print(f"Spearman trend summary saved to {spearman_csv_path}")


if __name__ == "__main__":
    main()

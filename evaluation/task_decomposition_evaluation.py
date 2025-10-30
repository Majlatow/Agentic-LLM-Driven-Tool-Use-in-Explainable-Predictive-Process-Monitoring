# Evaluation of AsyncPlannerAgent task decomposition across local LLM models.
import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import csv

import networkx as nx

from einhorn.model_manager import AsyncLLMTokenManager
from einhorn.planner_agent_async import AsyncPlannerAgent
from time import perf_counter
from pathlib import Path


@dataclass
class EvaluationResult:
    scenario: str
    scenario_description: str
    model_id: int
    rule_iteration: int
    example_iteration: int
    prompt_rules: List[str] = field(default_factory=list)
    prompt_examples: List[str] = field(default_factory=list)
    rule_count: int = 0
    example_count: int = 0
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    process_plan: List[List[Dict[str, Any]]] = field(default_factory=list)
    error: Optional[str] = None
    correct_output_format: Optional[int] = None
    tool_coverage: Optional[float] = None
    edge_coverage: Optional[float] = None
    false_tool_count: Optional[int] = None
    false_edge_count: Optional[int] = None
    dag_graph_edit_distance: Optional[float] = None


TRACE_A = "178864"
TRACE_B = "181247"

TEST_DATA_X = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 100.0]
TEST_DATA_Y = [2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 113.0]


def build_trace_prompt(trace_a: str, trace_b: str) -> str:
    return (
        f"For the traces {trace_a} and {trace_b} find the event with the longest predicted processing time "
        "and perform a SHAP analysis for this event."
    )


def build_list_prompt() -> str:
    return (
        "Analyze these two numerical lists:\n"
        f"List X: {TEST_DATA_X}\n"
        f"List Y: {TEST_DATA_Y}\n\n"
        "Please perform the following analyses:\n"
        "1. Calculate basic statistics (mean, median, std) for both lists\n"
        "2. Check for outliers in both lists\n"
        "3. Calculate basic statistics (mean, median, std) for both lists without outliers\n"
        "4. Calculate the 25th, 50th, and 75th percentiles for both lists without outliers"
    )


TRACE_PROMPT = build_trace_prompt(TRACE_A, TRACE_B)
LIST_PROMPT = build_list_prompt()

def build_expected_graph_trace(trace_a: str, trace_b: str) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": 1,
                "tool_name": "get_event_processing_time_predictions",
                "description": f"Get prediction intervals for traces {trace_a} and {trace_b}",
                "required_inputs": ["trace_event_requests: Sequence"],
                "input_mapping": f"trace_event_requests <- [('" + trace_a + "',), ('" + trace_b + "',)]",
                "missing_inputs": "None",
            },
            {
                "id": 2,
                "tool_name": "get_event_SHAP_explanation",
                "description": "Run SHAP explanation for the event with the longest predicted duration",
                "required_inputs": [
                    "trace_id: str",
                    "event_position: int",
                    "show_plots: bool",
                    "save_plots: bool",
                ],
                "input_mapping": "trace_id <- task 1 output['max_time_trace_id']; event_position <- task 1 output['max_time_event_index']; show_plots=False; save_plots=False",
                "missing_inputs": "trace_id <- task 1 output['max_time_trace_id']; event_position <- task 1 output['max_time_event_index']",
            },
        ],
        "edges": [
            {
                "source": 1,
                "target": 2,
                "forwarded_inputs": ["trace_id", "event_position"],
            },
        ],
    }


TYPE_HINTS = {
    "data": "List[float]",
    "x": "List[float]",
    "y": "List[float]",
    "percentiles": "List[float]",
    "threshold": "float",
}


def build_expected_graph_lists() -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": 1,
                "tool_name": "calculate_summary_stats",
                "description": "Summary stats for List X",
                "required_inputs": ["data: List[float]"],
                "input_mapping": "data <- List X",
                "missing_inputs": "None",
            },
            {
                "id": 2,
                "tool_name": "calculate_summary_stats",
                "description": "Summary stats for List Y",
                "required_inputs": ["data: List[float]"],
                "input_mapping": "data <- List Y",
                "missing_inputs": "None",
            },
            {
                "id": 3,
                "tool_name": "detect_outliers",
                "description": "Outliers for List X",
                "required_inputs": ["data: List[float]", "threshold: float"],
                "input_mapping": "data <- List X; threshold <- default 1.5",
                "missing_inputs": "None",
            },
            {
                "id": 4,
                "tool_name": "detect_outliers",
                "description": "Outliers for List Y",
                "required_inputs": ["data: List[float]", "threshold: float"],
                "input_mapping": "data <- List Y; threshold <- default 1.5",
                "missing_inputs": "None",
            },
            {
                "id": 5,
                "tool_name": "calculate_summary_stats",
                "description": "Summary stats for List X without outliers",
                "required_inputs": ["data: List[float]"],
                "input_mapping": "data <- task 3 output['filtered_data']",
                "missing_inputs": "data <- task 3 output['filtered_data']",
            },
            {
                "id": 6,
                "tool_name": "calculate_summary_stats",
                "description": "Summary stats for List Y without outliers",
                "required_inputs": ["data: List[float]"],
                "input_mapping": "data <- task 4 output['filtered_data']",
                "missing_inputs": "data <- task 4 output['filtered_data']",
            },
            {
                "id": 7,
                "tool_name": "calculate_percentiles",
                "description": "Percentiles for List X without outliers",
                "required_inputs": ["data: List[float]", "percentiles: List[float]"],
                "input_mapping": "data <- task 3 output['filtered_data']; percentiles=[25, 50, 75]",
                "missing_inputs": "data <- task 3 output['filtered_data']",
            },
            {
                "id": 8,
                "tool_name": "calculate_percentiles",
                "description": "Percentiles for List Y without outliers",
                "required_inputs": ["data: List[float]", "percentiles: List[float]"],
                "input_mapping": "data <- task 4 output['filtered_data']; percentiles=[25, 50, 75]",
                "missing_inputs": "data <- task 4 output['filtered_data']",
            },
        ],
        "edges": [
            {"source": 3, "target": 5, "forwarded_inputs": ["data"]},
            {"source": 4, "target": 6, "forwarded_inputs": ["data"]},
            {"source": 3, "target": 7, "forwarded_inputs": ["data"]},
            {"source": 4, "target": 8, "forwarded_inputs": ["data"]},
        ],
    }


LOCAL_MODEL_IDS = [-1, -2, -3, -4]

PLANNER_PROMPT_RULES: List[str] = [
    "1. Represent every task as a node with fields: {\"id\":int, \"tool_name\":str, \"description\":str (optional free text), \"required_inputs\": [str,...], \"input_mapping\":str, \"missing_inputs\":str}.",
    "2. Provide dependencies between tasks as directed edges with structure {\"source\":int, \"target\":int, \"forwarded_inputs\": [str,...] }, where each item in forwarded_inputs is a parameter name of the target tool to be fulfilled from the source task’s output.",
    "3. Use the exact JSON schema: {\"nodes\":[{...},...], \"edges\":[{...},...]}",
    "4. Each task must map to at least one available tool.",
    "5. If any tool must be applied to multiple datasets or contexts, create a dedicated node for each invocation (no combining multiple uses into one node).",
    "6. 'tool_name' MUST be the exact function name (e.g., 'calculate_summary_stats').",
    "7. 'required_inputs' MUST list every parameter for the tool with the format '<name>: <type or class>' (e.g., 'data: List[float]').",
    "8. 'missing_inputs' MUST describe, using the format '<parameter> <- task <id> output[<path>]' for each unmet requirement, where <path> is optional and may be an attribute, key, or index accessor (e.g., output['foo'] or .foo). Use 'None' when all inputs are satisfied.",
    "9. 'input_mapping' MUST be free text describing how available inputs are bound to the required parameters.",
    "10. For every task, create edges ONLY from tasks explicitly cited in its 'missing_inputs'. Do not create edges for tasks whose 'missing_inputs' is 'None'. Ensure each cited task appears exactly once in the incoming edges.",
    "11. Ensure the dependency graph between tasks is acyclic and ids start at 1 with no gaps.",
    "12. Among tasks that can run in parallel, preserve the order implied by the user request when serialising them (unless a dependency requires otherwise).",
    "13. DO NOT include comments, placeholder/dummy nodes, or extra fields in the JSON output. Only provide valid JSON.",
]

def build_prompt_rule_iterations(desired_counts: Optional[List[int]] = None) -> List[List[str]]:
    total_rules = len(PLANNER_PROMPT_RULES)
    counts = desired_counts or list(range(1, total_rules + 1))
    iterations: List[List[str]] = []
    for count in counts:
        if count < 1 or count > total_rules:
            continue
        iterations.append(PLANNER_PROMPT_RULES[:count])
    return iterations


EXAMPLE_PROMPT_VARIANTS: List[str] = [
    "Example 1 (User request: 'Summarize A and B, then correlate them') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for dataset A\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_A as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for dataset B\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_B as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 3, \"tool_name\": \"pearson_correlation\", \"description\": \"Correlation on A vs B\", \"required_inputs\": [\"x: List[float]\", \"y: List[float]\"], \"input_mapping\": \"Use list_A for x and list_B for y as data\", \"missing_inputs\": \"None\"}\n  ],\n  \"edges\": [\n  ]\n}\n",
    "Example 2 (User request: 'Calculate summary stats for two lists, find outliers, then recompute stats and percentiles without outliers') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List X\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_X as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List Y\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_Y as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 3, \"tool_name\": \"detect_outliers\", \"description\": \"Detect outliers in List X\", \"required_inputs\": [\"data: List[float]\", \"threshold: float\"], \"input_mapping\": \"data <- task 1 output; threshold=1.5\", \"missing_inputs\": \"data <- task 1 output\"},\n    {\"id\": 4, \"tool_name\": \"detect_outliers\", \"description\": \"Detect outliers in List Y\", \"required_inputs\": [\"data: List[float]\", \"threshold: float\"], \"input_mapping\": \"data <- task 2 output; threshold=1.5\", \"missing_inputs\": \"data <- task 2 output\"},\n    {\"id\": 5, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List X without outliers\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use filtered data from task 3\", \"missing_inputs\": \"data <- task 3 output\"},\n    {\"id\": 6, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List Y without outliers\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use filtered data from task 4\", \"missing_inputs\": \"data <- task 4 output\"},\n    {\"id\": 7, \"tool_name\": \"calculate_percentiles\", \"description\": \"Percentiles for List X without outliers\", \"required_inputs\": [\"data: List[float]\", \"percentiles: List[float]\"], \"input_mapping\": \"data <- task 3 output; percentiles=[25, 50, 75]\", \"missing_inputs\": \"data <- task 3 output\"},\n    {\"id\": 8, \"tool_name\": \"calculate_percentiles\", \"description\": \"Percentiles for List Y without outliers\", \"required_inputs\": [\"data: List[float]\", \"percentiles: List[float]\"], \"input_mapping\": \"data <- task 4 output; percentiles=[25, 50, 75]\", \"missing_inputs\": \"data <- task 4 output\"}\n  ],\n  \"edges\": [\n    {\"source\": 3, \"target\": 5, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 4, \"target\": 6, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 3, \"target\": 7, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 4, \"target\": 8, \"forwarded_inputs\": [\"data\"]}\n  ]\n}\n",
    "Example 3 (User request: 'For the traces with IDs 234235 and 795463, find the event amongst them with the shortest processing time and explain the prediction for this event via SHAP.') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"get_event_processing_time_predictions\", \"description\": \"Collect predicted processing times for traces\", \"required_inputs\": [\"trace_event_requests: Sequence\"], \"input_mapping\": \"trace_event_requests <- [(\\\"234235\\\",), (\\\"795463\\\",)]\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"get_event_SHAP_explanation\", \"description\": \"Run SHAP on shortest predicted event\", \"required_inputs\": [\"trace_id: str\", \"event_position: int\", \"show_plots: bool\", \"save_plots: bool\"], \"input_mapping\": \"trace_id <- task 1 output['min_time_trace_id']; event_position <- task 1 output['min_time_event_index']; show_plots=False; save_plots=False\", \"missing_inputs\": \"trace_id <- task 1 output['min_time_trace_id']; event_position <- task 1 output['min_time_event_index']\"}\n  ],\n  \"edges\": [\n    {\"source\": 1, \"target\": 2, \"forwarded_inputs\": [\"trace_id\", \"event_position\"]}\n  ]\n}\n",
]


def build_prompt_example_iterations(desired_counts: Optional[List[int]] = None) -> List[List[str]]:
    base_options: List[List[str]] = [
        [],
        [EXAMPLE_PROMPT_VARIANTS[0]],
        [EXAMPLE_PROMPT_VARIANTS[0], EXAMPLE_PROMPT_VARIANTS[1]],
        [EXAMPLE_PROMPT_VARIANTS[0], EXAMPLE_PROMPT_VARIANTS[1], EXAMPLE_PROMPT_VARIANTS[2]],
    ]
    counts = desired_counts or list(range(len(base_options)))
    iterations: List[List[str]] = []
    for count in counts:
        if 0 <= count < len(base_options):
            iterations.append(base_options[count])
    return iterations


# def compute_precision_recall(expected_nodes: List[Dict], actual_nodes: List[Dict]):
# def cosine_similarity_edges(expected_graph: nx.DiGraph, actual_graph: nx.DiGraph) -> float:
# def graph_edit_distance_stats(expected_graph: nx.DiGraph, actual_graph: nx.DiGraph) -> Tuple[float, float, float]:

def format_examples_for_prompt(examples: List[str]) -> List[str]:
    return list(examples)


def build_planner(model_manager: AsyncLLMTokenManager, model_id: int) -> AsyncPlannerAgent:
    return AsyncPlannerAgent(model_manager, model_id)


def debug_print_tasks(model_id: int, rule_count: int, example_count: int, tasks: List[Any]) -> None:
    separator = "-" * 72
    print(f"\n{separator}")
    print(
        f"Decompose Output | model={model_id} | rules={rule_count} | examples={example_count}"
    )
    if not tasks:
        print("  (no tasks returned)")
        print(separator)
        return

    for task in tasks:
        base = f"Task {getattr(task, 'index', '?')}: {getattr(task, 'tool_name', 'Unknown tool')}"
        print(base)
        description = getattr(task, "description", None)
        if description:
            print(f"  description     : {description}")
        depends_on = getattr(task, "depends_on", None)
        if depends_on:
            print(f"  depends_on      : {depends_on}")
        required_inputs = getattr(task, "required_inputs", None)
        if required_inputs:
            print(f"  required_inputs : {required_inputs}")
        input_mapping = getattr(task, "input_mapping", None)
        if input_mapping:
            print(f"  input_mapping   : {input_mapping}")
        missing_inputs = getattr(task, "missing_inputs", None)
        if missing_inputs:
            print(f"  missing_inputs  : {missing_inputs}")
        forwarded_inputs = getattr(task, "forwarded_inputs", None)
        if forwarded_inputs:
            print(f"  forwarded_inputs: {forwarded_inputs}")
    print(separator)


def build_actual_graph(tasks: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for task in tasks:
        nodes.append(
            {
                "id": getattr(task, "index", None),
                "tool_name": getattr(task, "tool_name", None),
                "description": getattr(task, "description", None),
                "required_inputs": getattr(task, "required_inputs", None),
                "depends_on": getattr(task, "depends_on", None),
                "input_mapping": getattr(task, "input_mapping", None),
                "missing_inputs": getattr(task, "missing_inputs", None),
            }
        )

    for task in tasks:
        depends_on = getattr(task, "depends_on", None) or []
        forwarded_inputs = getattr(task, "forwarded_inputs", None) or {}
        for dependency in depends_on:
            edges.append(
                {
                    "source": dependency,
                    "target": getattr(task, "index", None),
                    "forwarded_inputs": forwarded_inputs.get(dependency, []),
                }
            )

    return nodes, edges


def compute_tool_coverage(expected_nodes: List[Dict[str, Any]], actual_nodes: List[Dict[str, Any]]) -> float:
    if not expected_nodes:
        return 1.0 if not actual_nodes else 0.0

    expected_counts = {}
    for node in expected_nodes:
        tool = node.get("tool_name")
        if tool is None:
            continue
        expected_counts[tool] = expected_counts.get(tool, 0) + 1

    matched = 0
    actual_counts = {}
    for node in actual_nodes:
        tool = node.get("tool_name")
        if tool is None:
            continue
        actual_counts[tool] = actual_counts.get(tool, 0) + 1

    for tool, count in expected_counts.items():
        available = actual_counts.get(tool, 0)
        matched += min(count, available)

    total = sum(expected_counts.values())
    return matched / total if total else 0.0


def compute_edge_coverage(expected_edges: List[Dict[str, Any]], actual_edges: List[Dict[str, Any]]) -> float:
    if not expected_edges:
        return 1.0 if not actual_edges else 0.0

    expected_counts = {}
    for edge in expected_edges:
        key = (edge.get("source"), edge.get("target"))
        if None in key:
            continue
        expected_counts[key] = expected_counts.get(key, 0) + 1

    actual_counts = {}
    for edge in actual_edges:
        key = (edge.get("source"), edge.get("target"))
        if None in key:
            continue
        actual_counts[key] = actual_counts.get(key, 0) + 1

    matched = 0
    for key, count in expected_counts.items():
        available = actual_counts.get(key, 0)
        matched += min(count, available)

    total = sum(expected_counts.values())
    return matched / total if total else 0.0


def count_false_tools(expected_nodes: List[Dict[str, Any]], actual_nodes: List[Dict[str, Any]]) -> int:
    expected_counts = {}
    for node in expected_nodes:
        tool = node.get("tool_name")
        if tool is None:
            continue
        expected_counts[tool] = expected_counts.get(tool, 0) + 1

    false_count = 0
    actual_counts = {}
    for node in actual_nodes:
        tool = node.get("tool_name")
        if tool is None:
            continue
        actual_counts[tool] = actual_counts.get(tool, 0) + 1

    for tool, count in actual_counts.items():
        allowed = expected_counts.get(tool, 0)
        if count > allowed:
            false_count += count - allowed

    return false_count


def count_false_edges(expected_edges: List[Dict[str, Any]], actual_edges: List[Dict[str, Any]]) -> int:
    expected_counts = {}
    for edge in expected_edges:
        key = (edge.get("source"), edge.get("target"))
        if None in key:
            continue
        expected_counts[key] = expected_counts.get(key, 0) + 1

    false_count = 0
    actual_counts = {}
    for edge in actual_edges:
        key = (edge.get("source"), edge.get("target"))
        if None in key:
            continue
        actual_counts[key] = actual_counts.get(key, 0) + 1

    for key, count in actual_counts.items():
        allowed = expected_counts.get(key, 0)
        if count > allowed:
            false_count += count - allowed

    return false_count


def compute_graph_edit_distance(
    expected_nodes: List[Dict[str, Any]],
    expected_edges: List[Dict[str, Any]],
    actual_nodes: List[Dict[str, Any]],
    actual_edges: List[Dict[str, Any]],
) -> Optional[float]:
    try:
        expected_graph = nx.DiGraph()
        actual_graph = nx.DiGraph()

        for node in expected_nodes:
            expected_graph.add_node(node.get("id"))
        for edge in expected_edges:
            expected_graph.add_edge(edge.get("source"), edge.get("target"))

        for node in actual_nodes:
            actual_graph.add_node(node.get("id"))
        for edge in actual_edges:
            actual_graph.add_edge(edge.get("source"), edge.get("target"))

        if not nx.is_directed_acyclic_graph(expected_graph):
            return None
        if not nx.is_directed_acyclic_graph(actual_graph):
            return None

        ged = nx.algorithms.similarity.graph_edit_distance(expected_graph, actual_graph)
        if ged is None:
            return None
        return float(ged)
    except Exception:
        return None


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if not value.is_integer() else f"{int(value)}"
    return str(value)


async def evaluate_configuration(
    model_manager: AsyncLLMTokenManager,
    model_id: int,
    rules: List[str],
    examples: List[str],
    prompt: str,
    verbose: bool,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[List[Dict[str, Any]]],
    Optional[str],
]:
    planner = build_planner(model_manager, model_id)
    await planner.init()

    planner.set_prompt_rules(rules)
    planner.set_prompt_examples(format_examples_for_prompt(examples))

    try:
        tasks = await planner.decompose(prompt)
        if verbose:
            debug_print_tasks(model_id, len(rules), len(examples), tasks)
        nodes, edges = build_actual_graph(tasks)
        process_plan: List[List[Dict[str, Any]]] = []
        try:
            subtasks = planner._parse_sub_tasks(json.dumps({"nodes": nodes, "edges": edges}))
            plan_levels = AsyncPlannerAgent.build_process_plan(subtasks)
            for level in plan_levels:
                process_plan.append(
                    [
                        {
                            "id": task.index,
                            "tool_name": task.tool_name,
                            "description": task.description,
                            "required_inputs": task.required_inputs,
                            "depends_on": task.depends_on,
                            "forwarded_inputs": task.forwarded_inputs,
                            "input_mapping": task.input_mapping,
                            "missing_inputs": task.missing_inputs,
                        }
                        for task in level
                    ]
                )
        except ValueError as parse_error:
            print(f"Warning: failed to parse tasks into process plan: {parse_error}")
        return (
            [task.to_dict() for task in tasks],
            nodes,
            edges,
            process_plan,
            None,
        )
    except Exception as exc:
        return [], [], [], [], str(exc)
    finally:
        await planner.shutdown()


def evaluate_task_decomposition(
    scenarios: List[str],
    model_ids: List[int],
    prompt_rule_iterations: List[List[str]],
    prompt_example_iterations: List[List[str]],
    config_path: str,
    verbose: bool = True,
    results_csv: Optional[str] = None,
) -> Tuple[List[EvaluationResult], float]:
    loop = asyncio.get_event_loop()

    manager = AsyncLLMTokenManager(config_path=config_path)

    results: List[EvaluationResult] = []

    start_time = perf_counter()

    for scenario_key in scenarios:
        scenario = SCENARIOS[scenario_key]
        prompt = scenario["prompt_builder"]()
        expected_graph = scenario["expected_graph_builder"]()
        expected_nodes = expected_graph.get("nodes", [])
        expected_edges = expected_graph.get("edges", [])

        if verbose:
            print(f"\n=== Scenario: {scenario_key} ({scenario['description']}) ===")

        for model_id in model_ids:
            for rule_idx, rules in enumerate(prompt_rule_iterations):
                for example_idx, examples in enumerate(prompt_example_iterations):
                    tasks, nodes, edges, process_plan, error = loop.run_until_complete(
                        evaluate_configuration(manager, model_id, rules, examples, prompt, verbose)
                    )

                    if error:
                        correct_output_format = 0
                        tool_coverage = 0.0 if expected_nodes else 1.0
                        edge_coverage = 0.0 if expected_edges else 1.0
                        false_tools = len(nodes)
                        false_edges = len(edges)
                        dag_ged = None
                    else:
                        correct_output_format = 1
                        tool_coverage = compute_tool_coverage(expected_nodes, nodes)
                        edge_coverage = compute_edge_coverage(expected_edges, edges)
                        false_tools = count_false_tools(expected_nodes, nodes)
                        false_edges = count_false_edges(expected_edges, edges)
                        dag_ged = compute_graph_edit_distance(
                            expected_nodes,
                            expected_edges,
                            nodes,
                            edges,
                        )

                    results.append(
                        EvaluationResult(
                            scenario=scenario_key,
                            scenario_description=scenario["description"],
                            model_id=model_id,
                            rule_iteration=rule_idx,
                            example_iteration=example_idx,
                            prompt_rules=list(rules),
                            prompt_examples=format_examples_for_prompt(examples),
                            rule_count=len(rules),
                            example_count=len(examples),
                            tasks=tasks,
                            nodes=nodes,
                            edges=edges,
                            process_plan=process_plan,
                            error=error,
                            correct_output_format=correct_output_format,
                            tool_coverage=tool_coverage,
                            edge_coverage=edge_coverage,
                            false_tool_count=false_tools,
                            false_edge_count=false_edges,
                            dag_graph_edit_distance=dag_ged,
                        )
                    )

    elapsed_seconds = perf_counter() - start_time
    minutes, seconds = divmod(elapsed_seconds, 60)

    resolved_csv = resolve_results_path(results_csv)
    if resolved_csv:
        write_results_csv(resolved_csv, results)

    if verbose:
        print(f"\nEvaluation completed in {int(minutes)}m {seconds:0.1f}s")

    return results, elapsed_seconds


def print_detailed_results(results: List[EvaluationResult], verbose: bool = True) -> None:
    if not results:
        return

    if not verbose:
        print("\n=== Planner Metrics ===")
        for result in results:
            header = (
                f"Scenario {result.scenario} ({result.scenario_description}) | Model {result.model_id} | rules={result.rule_count} | examples={result.example_count}"
            )
            print(header)
            if result.error:
                print(f"  Error   : {result.error}")
            print("  Metrics:")
            print(f"    correct_output_format: {format_metric(result.correct_output_format)}")
            print(f"    tool_coverage        : {format_metric(result.tool_coverage)}")
            print(f"    edge_coverage        : {format_metric(result.edge_coverage)}")
            print(f"    false_tool_count     : {format_metric(result.false_tool_count)}")
            print(f"    false_edge_count     : {format_metric(result.false_edge_count)}")
            print(f"    dag_graph_edit_distance: {format_metric(result.dag_graph_edit_distance)}")
        print("=== End Planner Metrics ===\n")
        return

    print("\n=== Planner Decomposition Records ===")
    for result in results:
        header = (
            f"Scenario {result.scenario} ({result.scenario_description}) | Model {result.model_id} | rules={result.rule_count} | examples={result.example_count}"
        )
        print(header)
        if result.error:
            print(f"  Error   : {result.error}")
            continue

        print("  Prompt rules:")
        for rule in result.prompt_rules:
            print(f"    - {rule}")

        if result.prompt_examples:
            print("  Prompt examples provided.")
        else:
            print("  No prompt examples provided.")

        print("  Captured tasks:")
        if not result.tasks:
            print("    (none)")
        for task in result.tasks:
            print(
                f"    Task {task.get('index')} | tool={task.get('tool_name')} | desc={task.get('description')}"
            )
            if task.get("depends_on"):
                print(f"      depends_on: {task['depends_on']}")
            if task.get("required_inputs"):
                print(f"      required_inputs: {task['required_inputs']}")
            if task.get("input_mapping"):
                print(f"      input_mapping: {task['input_mapping']}")
            if task.get("missing_inputs"):
                print(f"      missing_inputs: {task['missing_inputs']}")
            forwarded_inputs = task.get("forwarded_inputs")
            if forwarded_inputs:
                print(f"      forwarded_inputs: {forwarded_inputs}")

        print("  Metrics:")
        print(f"    correct_output_format: {format_metric(result.correct_output_format)}")
        print(f"    tool_coverage        : {format_metric(result.tool_coverage)}")
        print(f"    edge_coverage        : {format_metric(result.edge_coverage)}")
        print(f"    false_tool_count     : {format_metric(result.false_tool_count)}")
        print(f"    false_edge_count     : {format_metric(result.false_edge_count)}")
        print(f"    dag_graph_edit_distance: {format_metric(result.dag_graph_edit_distance)}")

        print("  Nodes/Edges JSON:")
        print(
            json.dumps(
                {
                    "nodes": result.nodes,
                    "edges": result.edges,
                },
                indent=2,
            )
        )

        if result.process_plan:
            print("  Hierarchical process plan:")
            for level_idx, level in enumerate(result.process_plan, start=1):
                entries = ", ".join(
                    f"id={task['id']} tool={task['tool_name']}" for task in level
                )
                print(f"    Level {level_idx}: {entries}")
        else:
            print("  Process plan could not be derived.")

    print("=== End Planner Decomposition Records ===\n")


def print_metric_tables(results: List[EvaluationResult], verbose: bool = True) -> None:
    if not results:
        return

    if not verbose:
        return

    grouped: Dict[str, Dict[int, Dict[int, Dict[int, EvaluationResult]]]] = {}
    for result in results:
        grouped.setdefault(result.scenario, {}).setdefault(result.model_id, {}).setdefault(result.rule_count, {})[result.example_count] = result

    headers = ["Examples\\Rules"] + [str(rules) for rules in sorted({res.rule_count for res in results})]

    col_widths = [max(len(header), 12) for header in headers]

    def format_row(values: List[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, col_widths))

    for scenario_key, model_blocks in grouped.items():
        scenario_description = SCENARIOS[scenario_key]["description"]
        print(f"\n=== Scenario {scenario_key}: {scenario_description} ===")
        for model_id, rule_blocks in model_blocks.items():
            print(f"\n=== Model {model_id} ===")
            example_counts = sorted({
                res.example_count for res in results if res.model_id == model_id and res.scenario == scenario_key
            })

            print("\nConfigurations")
            print(format_row(headers))
            print("-" * sum(width + 3 for width in col_widths))
            for examples in example_counts:
                row: List[str] = [str(examples)]
                for rules in sorted(rule_blocks.keys()):
                    result = rule_blocks[rules].get(examples)
                    if result is None:
                        value = "NA"
                    elif result.error:
                        value = "Error"
                    else:
                        value = "OK"
                    row.append(value)
                print(format_row(row))

            print("\nCaptured Graphs:")
            for rules in sorted(rule_blocks.keys()):
                for examples in example_counts:
                    result = rule_blocks[rules].get(examples)
                    if not result:
                        continue
                    status = "ERROR" if result.error else "SUCCESS"
                    if result.error:
                        print(f"- Rules={rules}, Examples={examples}, Status={status}")
                        print(f"  Error: {result.error}")
                        continue

                    if verbose:
                        print(f"- Rules={rules}, Examples={examples}, Status={status}")
                        print(
                            json.dumps(
                                {
                                    "nodes": result.nodes,
                                    "edges": result.edges,
                                },
                                indent=2,
                            )
                        )
                        print("    Metrics:")
                        print(f"      correct_output_format: {format_metric(result.correct_output_format)}")
                        print(f"      tool_coverage        : {format_metric(result.tool_coverage)}")
                        print(f"      edge_coverage        : {format_metric(result.edge_coverage)}")
                        print(f"      false_tool_count     : {format_metric(result.false_tool_count)}")
                        print(f"      false_edge_count     : {format_metric(result.false_edge_count)}")
                        print(f"      dag_graph_edit_distance: {format_metric(result.dag_graph_edit_distance)}")
                    else:
                        print(
                            f"- Rules={rules}, Examples={examples}, Status={status} | metrics: "
                            f"correct_output_format={format_metric(result.correct_output_format)}, "
                            f"tool_coverage={format_metric(result.tool_coverage)}, edge_coverage={format_metric(result.edge_coverage)}, "
                            f"false_tool_count={format_metric(result.false_tool_count)}, false_edge_count={format_metric(result.false_edge_count)}, "
                            f"dag_graph_edit_distance={format_metric(result.dag_graph_edit_distance)}"
                        )


SCENARIOS = {
    "trace": {
        "description": "Analyze manufacturing traces for longest event and SHAP explanation",
        "prompt_builder": lambda: build_trace_prompt(TRACE_A, TRACE_B),
        "expected_graph_builder": lambda: build_expected_graph_trace(TRACE_A, TRACE_B),
    },
    "lists": {
        "description": "Analyze two numerical lists for stats, outliers, and percentiles",
        "prompt_builder": build_list_prompt,
        "expected_graph_builder": build_expected_graph_lists,
    },
}


SCENARIO_CODES = {
    "trace": 1,
    "lists": 2,
}


def build_results_table(results: List[EvaluationResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for res in results:
        rows.append(
            {
                "model_id": res.model_id,
                "rule_count": res.rule_count,
                "example_count": res.example_count,
                "scenario": SCENARIO_CODES.get(res.scenario, res.scenario),
                "correct_output_format": format_metric(res.correct_output_format),
                "tool_coverage": format_metric(res.tool_coverage),
                "edge_coverage": format_metric(res.edge_coverage),
                "false_tool_count": format_metric(res.false_tool_count),
                "false_edge_count": format_metric(res.false_edge_count),
                "dag_graph_edit_distance": format_metric(res.dag_graph_edit_distance),
            }
        )
    return rows


def write_results_csv(path: Union[str, Path], results: List[EvaluationResult]) -> None:
    rows = build_results_table(results)
    fieldnames = [
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
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_results_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    target = Path(path)
    if not target.is_absolute():
        target = Path(__file__).parent / target
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate planner task decomposition")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
        help="Scenario keys to evaluate",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        type=int,
        default=LOCAL_MODEL_IDS,
        help="Model IDs to evaluate (default: local models)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.json",
        help="Path to Einhorn configuration JSON",
    )
    parser.add_argument(
        "--rule-counts",
        nargs="+",
        type=int,
        default=None,
        help="Specific rule counts to evaluate (default: all counts)",
    )
    parser.add_argument(
        "--example-counts",
        nargs="+",
        type=int,
        default=None,
        help="Specific example counts to evaluate (0:none,1:first,2:first+second)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print detailed planner outputs; otherwise show condensed metrics.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Optional path to write aggregated evaluation metrics as CSV.",
    )

    args = parser.parse_args()

    prompt_rule_iterations = build_prompt_rule_iterations(args.rule_counts)
    prompt_example_iterations = build_prompt_example_iterations(args.example_counts)

    if args.results_csv:
        resolved_results_path = resolve_results_path(args.results_csv)
    else:
        resolved_results_path = resolve_results_path("results.csv")

    results, elapsed_seconds = evaluate_task_decomposition(
        scenarios=args.scenarios,
        model_ids=args.model_ids,
        prompt_rule_iterations=prompt_rule_iterations,
        prompt_example_iterations=prompt_example_iterations,
        config_path=args.config_path,
        verbose=args.verbose,
        results_csv=str(resolved_results_path) if resolved_results_path else None,
    )

    if resolved_results_path:
        write_results_csv(resolved_results_path, results)

    print_detailed_results(results, verbose=args.verbose)
    print_metric_tables(results, verbose=args.verbose)

    total_minutes, total_seconds = divmod(elapsed_seconds, 60)
    print(
        f"Evaluation processing time: {int(total_minutes)}m {total_seconds:0.1f}s"
    )


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

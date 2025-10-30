"""
Planner agent for decomposing tasks based on available tools.
"""
import re
import inspect
import json
from typing import List, Dict, Any, Optional, Sequence
from .data_models import SubTask
from . import internal_tools
from . import external_tools
class AsyncPlannerAgent:
    def __init__(self, model_interface, model_id: int):
        self.model_interface = model_interface
        self.model_id = model_id
        self.token = None
        self.available_tools = self._gather_tools()
        self._last_prompt: Optional[str] = None
        self._last_edges: List[Dict[str, Any]] = []
        self._last_parsed_edges: List[Dict[str, Any]] = []
        self.prompt_rules: List[str] = [
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
        self.prompt_examples: List[str] = [
           "Example 1 (User request: 'Summarize A and B, then correlate them') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for dataset A\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_A as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for dataset B\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_B as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 3, \"tool_name\": \"pearson_correlation\", \"description\": \"Correlation on A vs B\", \"required_inputs\": [\"x: List[float]\", \"y: List[float]\"], \"input_mapping\": \"Use list_A for x and list_B for y as data\", \"missing_inputs\": \"None\"}\n  ],\n  \"edges\": [\n  ]\n}\n",
            "Example 2 (User request: 'Calculate summary stats for two lists, find outliers, then recompute stats and percentiles without outliers') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List X\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_X as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List Y\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use list_Y as data\", \"missing_inputs\": \"None\"},\n    {\"id\": 3, \"tool_name\": \"detect_outliers\", \"description\": \"Detect outliers in List X\", \"required_inputs\": [\"data: List[float]\", \"threshold: float\"], \"input_mapping\": \"data <- task 1 output; threshold=1.5\", \"missing_inputs\": \"data <- task 1 output\"},\n    {\"id\": 4, \"tool_name\": \"detect_outliers\", \"description\": \"Detect outliers in List Y\", \"required_inputs\": [\"data: List[float]\", \"threshold: float\"], \"input_mapping\": \"data <- task 2 output; threshold=1.5\", \"missing_inputs\": \"data <- task 2 output\"},\n    {\"id\": 5, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List X without outliers\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use filtered data from task 3\", \"missing_inputs\": \"data <- task 3 output\"},\n    {\"id\": 6, \"tool_name\": \"calculate_summary_stats\", \"description\": \"Summary stats for List Y without outliers\", \"required_inputs\": [\"data: List[float]\"], \"input_mapping\": \"Use filtered data from task 4\", \"missing_inputs\": \"data <- task 4 output\"},\n    {\"id\": 7, \"tool_name\": \"calculate_percentiles\", \"description\": \"Percentiles for List X without outliers\", \"required_inputs\": [\"data: List[float]\", \"percentiles: List[float]\"], \"input_mapping\": \"data <- task 3 output; percentiles=[25, 50, 75]\", \"missing_inputs\": \"data <- task 3 output\"},\n    {\"id\": 8, \"tool_name\": \"calculate_percentiles\", \"description\": \"Percentiles for List Y without outliers\", \"required_inputs\": [\"data: List[float]\", \"percentiles: List[float]\"], \"input_mapping\": \"data <- task 4 output; percentiles=[25, 50, 75]\", \"missing_inputs\": \"data <- task 4 output\"}\n  ],\n  \"edges\": [\n    {\"source\": 3, \"target\": 5, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 4, \"target\": 6, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 3, \"target\": 7, \"forwarded_inputs\": [\"data\"]},\n    {\"source\": 4, \"target\": 8, \"forwarded_inputs\": [\"data\"]}\n  ]\n}\n",
            "Example 3 (User request: 'For the traces with IDs 234235 and 795463, find the event amongst them with the shortest processing time and explain the prediction for this event via SHAP.') Expected JSON:\n{\n  \"nodes\": [\n    {\"id\": 1, \"tool_name\": \"get_event_processing_time_predictions\", \"description\": \"Collect predicted processing times for traces\", \"required_inputs\": [\"trace_event_requests: Sequence\"], \"input_mapping\": \"trace_event_requests <- [(\\\"234235\\\",), (\\\"795463\\\",)]\", \"missing_inputs\": \"None\"},\n    {\"id\": 2, \"tool_name\": \"get_event_SHAP_explanation\", \"description\": \"Run SHAP on shortest predicted event\", \"required_inputs\": [\"trace_id: str\", \"event_position: int\", \"show_plots: bool\", \"save_plots: bool\"], \"input_mapping\": \"trace_id <- task 1 output['min_time_trace_id']; event_position <- task 1 output['min_time_event_index']; show_plots=False; save_plots=False\", \"missing_inputs\": \"trace_id <- task 1 output['min_time_trace_id']; event_position <- task 1 output['min_time_event_index']\"}\n  ],\n  \"edges\": [\n    {\"source\": 1, \"target\": 2, \"forwarded_inputs\": [\"trace_id\", \"event_position\"]}\n  ]\n}\n",
        ]

    def set_prompt_rules(self, rules: Sequence[str]) -> None:
        self.prompt_rules = list(rules)

    def add_prompt_rule(self, rule: str) -> None:
        self.prompt_rules.append(rule)

    def remove_prompt_rule(self, rule: str) -> None:
        if rule in self.prompt_rules:
            self.prompt_rules.remove(rule)

    def set_prompt_examples(self, examples: Sequence[str]) -> None:
        self.prompt_examples = list(examples)

    def add_prompt_example(self, example: str) -> None:
        self.prompt_examples.append(example)

    def clear_prompt_examples(self) -> None:
        self.prompt_examples.clear()

    def _build_prompt(self, user_prompt: str) -> str:
        tool_info = self._format_tool_info()

        rules_block = "Rules:\n" + "\n".join(self.prompt_rules) + "\n" if self.prompt_rules else ""
        examples_block = "\n".join(self.prompt_examples) + ("\n" if self.prompt_examples else "")

        prompt_parts = [
            "You are a task planning AI with access to only the following tools:\n\n",
            f"{tool_info}\n\n",
            "Using only these accessible tools, decompose the following request into a dependency-aware task graph.\n",
            "Return ONLY a JSON object that can be consumed by networkx (nodes + directed edges).\n",
            rules_block,
            examples_block,
            f"Request to decompose: {user_prompt}\n\n",
            "Return only the JSON DAG:"
        ]

        prompt = "".join(prompt_parts)
        self._last_prompt = prompt
        return prompt

    def _gather_tools(self) -> Dict[str, Dict[str, Any]]:
        """Gather all available tools and their documentation."""
        tools = {}
        
        # Gather internal tools
        for name, func in inspect.getmembers(internal_tools, inspect.isfunction):
            tools[name] = {
                "description": inspect.getdoc(func),
                "parameters": inspect.signature(func).parameters,
                "source": "internal_tools"
            }
            
        # Gather external tools
        for name, func in inspect.getmembers(external_tools, inspect.isfunction):
            tools[name] = {
                "description": inspect.getdoc(func),
                "parameters": inspect.signature(func).parameters,
                "source": "external_tools"
            }
            
        return tools

    def _format_tool_info(self) -> str:
        """Format tool information for the LLM prompt."""
        sections = []
        
        # Format internal tools
        internal = [name for name, info in self.available_tools.items() 
                   if info["source"] == "internal_tools"]
        if internal:
            sections.append("Internal Statistical Tools:")
            for name in internal:
                tool = self.available_tools[name]
                params = ", ".join(str(p) for p in tool["parameters"])
                sections.append(f"- {name}({params})")
                # Split description and take first line safely
                desc_lines = tool["description"].split("\n")
                first_line = desc_lines[0] if desc_lines else ""
                sections.append(f"  Description: {first_line}")
        
        # Format external tools
        external = [name for name, info in self.available_tools.items() 
                   if info["source"] == "external_tools"]
        if external:
            sections.append("\nExternal ML and Geographic Tools:")
            for name in external:
                tool = self.available_tools[name]
                params = ", ".join(str(p) for p in tool["parameters"])
                sections.append(f"- {name}({params})")
                # Split description and take first line safely
                desc_lines = tool["description"].split("\n")
                first_line = desc_lines[0] if desc_lines else ""
                sections.append(f"  Description: {first_line}")
                
        return "\n".join(sections)

    async def init(self):
        """Initialize the planner agent."""
        self.token = await self.model_interface.init_agent(self.model_id)

    async def decompose(self, user_prompt: str) -> List[SubTask]:
        """
        Decompose a user request into subtasks based on available tools.
        
        Args:
            user_prompt: The user's request to decompose
            
        Returns:
            List of SubTask objects representing the decomposed tasks
        """
        prompt = self._build_prompt(user_prompt)

        response = await self.model_interface.ask_agent(self.token, prompt)
        subtasks = self._parse_sub_tasks(response)
        self._last_edges = self._last_parsed_edges
        return subtasks

    def _parse_sub_tasks(self, response: str) -> List[SubTask]:
        """Parse the JSON DAG response into SubTask objects."""
        if not response or "{" not in response:
            raise ValueError("Planner response did not contain JSON content.")

        json_start = response.find("{")
        json_end = response.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            raise ValueError("Unable to locate JSON object in planner response.")

        json_str = response[json_start:json_end + 1]

        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode planner JSON: {exc}") from exc

        nodes: List[Dict[str, Any]] = payload.get("nodes", [])
        edges: List[Dict[str, Any]] = payload.get("edges", [])
        self._last_parsed_edges = edges

        if not nodes:
            return []

        task_data: Dict[int, Dict[str, Any]] = {}
        for node in nodes:
            node_id = int(node["id"])
            task_data[node_id] = {
                "description": node.get("description"),
                "tool_name": node["tool_name"],
                "required_inputs": node.get("required_inputs"),
                "input_mapping": node.get("input_mapping"),
                "missing_inputs": node.get("missing_inputs"),
                "depends_on": set(),
                "forwarded": {}
            }

        for edge in edges:
            source = int(edge["source"])
            target = int(edge["target"])
            forwarded_inputs = edge.get("forwarded_inputs", []) or []

            if target in task_data:
                task_data[target]["depends_on"].add(source)
                if forwarded_inputs:
                    task_data[target]["forwarded"][source] = forwarded_inputs

        subtasks: List[SubTask] = []
        for node_id in sorted(task_data.keys()):
            info = task_data[node_id]
            depends_on = sorted(info["depends_on"]) if info["depends_on"] else None
            forwarded_inputs = info["forwarded"] or None
            subtasks.append(SubTask(
                index=node_id,
                description=info.get("description"),
                depends_on=depends_on,
                tool_name=info.get("tool_name"),
                required_inputs=info.get("required_inputs"),
                forwarded_inputs=forwarded_inputs,
                input_mapping=info.get("input_mapping"),
                missing_inputs=info.get("missing_inputs")
            ))

        return subtasks

    @staticmethod
    def compute_execution_levels(nodes: List[Dict[str, Any]], edges: List[Any]) -> List[List[int]]:
        """Return execution levels (lists of node ids) derived from DAG nodes/edges."""
        if not nodes:
            return []

        node_ids = [node["id"] for node in nodes]
        in_degree = {node_id: 0 for node_id in node_ids}
        adjacency: Dict[int, List[int]] = {node_id: [] for node_id in node_ids}

        for edge in edges or []:
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source, target = edge[0], edge[1]
            else:
                continue

            if source not in adjacency or target not in in_degree:
                raise ValueError(f"Edge references unknown node ids: {edge}")

            adjacency[source].append(target)
            in_degree[target] += 1

        levels: List[List[int]] = []
        ready = sorted([node_id for node_id, deg in in_degree.items() if deg == 0])
        processed_count = 0

        while ready:
            current_level = ready
            levels.append(current_level)
            next_ready: List[int] = []

            for node_id in current_level:
                processed_count += 1
                for neighbor in adjacency[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)

            ready = sorted(next_ready)

        if processed_count != len(node_ids):
            raise ValueError("Cycle detected in planner DAG - cannot compute execution levels")

        return levels

    @staticmethod
    def build_process_plan(subtasks: List[SubTask]) -> List[List[SubTask]]:
        """Convert subtasks into hierarchical execution levels."""
        if not subtasks:
            return []

        nodes = [{
            "id": subtask.index,
            "tool_name": subtask.tool_name,
            "description": subtask.description,
            "required_inputs": subtask.required_inputs
        } for subtask in subtasks]

        edges = []
        for subtask in subtasks:
            if not subtask.depends_on:
                continue
            for dependency in subtask.depends_on:
                forwarded = []
                if subtask.forwarded_inputs and dependency in subtask.forwarded_inputs:
                    forwarded = subtask.forwarded_inputs[dependency]
                edges.append({
                    "source": dependency,
                    "target": subtask.index,
                    "forwarded_inputs": forwarded
                })

        level_ids = AsyncPlannerAgent.compute_execution_levels(nodes, edges)
        subtask_lookup = {subtask.index: subtask for subtask in subtasks}

        process_plan: List[List[SubTask]] = []
        for level in level_ids:
            process_plan.append([subtask_lookup[node_id] for node_id in level])

        return process_plan

    async def shutdown(self):
        """Shutdown the planner agent."""
        await self.model_interface.shutdown_agent(self.token)

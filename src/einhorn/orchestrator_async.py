import asyncio
import json
import re
from typing import List, Dict, Any
from collections import defaultdict, deque

from .model_manager import AsyncLLMTokenManager
from .planner_agent_async import AsyncPlannerAgent
from .filler_agent_async import AsyncFillerAgent
from .matcher_llm_agent import LLMBasedMatcherAgent
from .function_executor import FunctionExecutor
from .template_builder import TemplateBuilder
from .data_models import SubTask, ToolTemplate

class AsyncOrchestrator:
    def __init__(self, role_model_map: Dict[str, int]):
        self.role_model_map = role_model_map
        self.llm = AsyncLLMTokenManager()
        self.executor = FunctionExecutor()
        self.matcher = LLMBasedMatcherAgent(self.llm, self.role_model_map.get("matcher", 1))
        self.builder = TemplateBuilder()

    async def __aenter__(self):
        await self.matcher.init()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.matcher.shutdown()
        await self.llm.shutdown_all()

    async def run(self, user_prompt: str, tool_docs_json: List[str], save_log_path: str = "subtask_log.json") -> str:
        templates: List[ToolTemplate] = []
        for doc in tool_docs_json:
            templates.extend(self.builder.build_templates_from_json(doc))

        # Step 1: Decompose prompt
        planner = AsyncPlannerAgent(self.llm, self.role_model_map["planner"])
        await planner.init()
        sub_tasks: List[SubTask] = await planner.decompose(user_prompt)
        await planner.shutdown()

        # Step 2: Organize tasks into levels based on dependencies
        levels = self.get_execution_levels(sub_tasks)

        # Step 3: Execute level by level
        result_map: Dict[int, Any] = {}
        task_log: List[Dict[str, Any]] = []

        for level in levels:
            # For each sub-task in this level, prepare a coroutine
            coroutines = [
                self._run_subtask(task, templates, result_map)
                for task in level
            ]
            level_results = await asyncio.gather(*coroutines)
            for r in level_results:
                result_map[r["subtask_index"]] = r.get("result")
                task_log.append(r)

        await self.llm.shutdown_all()
        self._save_task_log(task_log, save_log_path)

        return "\n\n".join([
            f"Sub-task {r['subtask_index']}: {r['description']}\nResult: {r.get('result', r.get('error_message', 'N/A'))}"
            for r in task_log
        ])

    async def _run_subtask(self, task: SubTask, templates: List[ToolTemplate], result_map: Dict[int, Any]) -> Dict[str, Any]:
        entry = {
            "subtask_index": task.index,
            "description": task.description,
            "status": "pending"
        }

        # Inject result-based substitutions
        enriched_input = self.substitute_dependencies(task, result_map)

        # Match tool
        template = self.matcher.match(task, templates)
        if not template:
            entry["status"] = "no_matching_tool"
            return entry

        try:
            filler = AsyncFillerAgent(self.llm, self.role_model_map["filler"])
            params = await filler.extract_params(enriched_input, template)

            result = await self.executor.execute_async(
                module_path=template.module_path,
                function_name=template.function_name,
                parameters=params
            )

            entry.update({
                "function_name": template.function_name,
                "module_path": template.module_path,
                "parameters": params,
                "result": result,
                "status": "success"
            })
        except Exception as e:
            entry.update({"status": "error", "error_message": str(e)})
        return entry

    def substitute_dependencies(self, task: SubTask, result_map: Dict[int, Any]) -> str:
        text = task.description
        for dep in (task.depends_on or []):
            result = result_map.get(dep)
            if result is not None:
                text = re.sub(
                    fr"\b(result\s+of\s+(step|task)\s*{dep})\b",
                    json.dumps(result),
                    text,
                    flags=re.IGNORECASE
                )
        return text

    def get_execution_levels(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        indegree = defaultdict(int)
        graph = defaultdict(list)
        index_map = {t.index: t for t in subtasks}

        for task in subtasks:
            for dep in (task.depends_on or []):
                indegree[task.index] += 1
                graph[dep].append(task.index)

        ready = deque([t.index for t in subtasks if indegree[t.index] == 0])
        visited = set()
        levels = []

        while ready:
            current_level = []
            for _ in range(len(ready)):
                idx = ready.popleft()
                if idx in visited:
                    continue
                visited.add(idx)
                task = index_map[idx]
                current_level.append(task)
                for neighbor in graph[idx]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        ready.append(neighbor)
            levels.append(current_level)
        return levels

    def _save_task_log(self, log: List[Dict[str, Any]], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

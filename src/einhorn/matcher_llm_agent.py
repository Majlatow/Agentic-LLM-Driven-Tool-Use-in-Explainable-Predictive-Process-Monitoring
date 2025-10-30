import re
from typing import List, Optional
from .data_models import SubTask, ToolTemplate

class LLMBasedMatcherAgent:
    def __init__(self, model_interface, model_id: int):
        self.model_interface = model_interface
        self.model_id = model_id
        self.token = None

    async def init(self):
        self.token = await self.model_interface.init_agent(self.model_id)

    async def match(self, subtask: SubTask, tools: List[ToolTemplate]) -> Optional[ToolTemplate]:
        if not tools:
            return None

        prompt = self._build_prompt(subtask, tools)

        print(f"Built prompt: {prompt}")
        
        response = await self.model_interface.ask_agent(self.token, prompt)
        
        print(f"Yielded response: {response}")
        
        matched_name = self._parse_function_name(response)
        return next((t for t in tools if t.function_name == matched_name), None)

    def _build_prompt(self, subtask: SubTask, tools: List[ToolTemplate]) -> str:
        lines = [
            f"Given the following sub-task:",
            f"{subtask.description}\n",
            "Choose the single best matching function from the list below and only provide the name of the function as a single string as your answer. If no function fits, your answer should be 'None'.\n"
        ]
        for i, t in enumerate(tools, start=1):
            param_list = ", ".join(t.parameters.keys())
            lines.append(f"{i}. {t.function_name}: {t.description} (Parameters: {param_list})")

        lines.append("\nReturn only the name of the best-matching function.")
        return "\n".join(lines)

    def _parse_function_name(self, response: str) -> Optional[str]:
        # Accept first word-like token as candidate
        match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", response)
        return match.group(1) if match else None

    async def shutdown(self):
        await self.model_interface.shutdown_agent(self.token)

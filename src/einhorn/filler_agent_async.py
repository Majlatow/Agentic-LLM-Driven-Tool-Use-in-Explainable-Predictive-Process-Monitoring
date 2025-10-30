import re
import json
from typing import Dict
from .data_models import ToolTemplate

class AsyncFillerAgent:
    def __init__(self, model_interface, model_id: int):
        self.model_interface = model_interface
        self.model_id = model_id

    async def extract_params(self, subtask_text: str, template: ToolTemplate) -> Dict[str, str]:
        token = await self.model_interface.init_agent(self.model_id)

        prompt = (
            f"You are given a task: \"{subtask_text}\"\n"
            f"Please extract the following parameters from this task description:\n"
            f"{', '.join(template.parameters)}\nReturn the output strictly as a JSON object."
        )

        response = await self.model_interface.ask_agent(token, prompt)
        await self.model_interface.shutdown_agent(token)

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in: {response}")
        return json.loads(match.group(0))

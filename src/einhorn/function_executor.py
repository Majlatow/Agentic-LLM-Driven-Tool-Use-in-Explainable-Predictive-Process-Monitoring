import sys
import os

import importlib
import asyncio
from functools import partial
from typing import Any, Dict

class FunctionExecutor:
    def _load_function(self, module_path, function_name):
        module = importlib.import_module(module_path)
        return getattr(module, function_name)

    def execute_serial(self, module_path: str, function_name: str, parameters: Dict[str, Any]) -> Any:
        # print(f"sys.path = {sys.path}")
        func = self._load_function(module_path, function_name)
        return func(**parameters)

    async def execute_async(self, module_path: str, function_name: str, parameters: Dict[str, Any]) -> Any:
        loop = asyncio.get_event_loop()
        func = self._load_function(module_path, function_name)
        return await loop.run_in_executor(None, partial(func, **parameters))

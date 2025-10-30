"""
Einhorn - A Python package developed iteratively

This package provides an asynchronous interface to the AIMAN API for managing AI models
and conversations, with support for task decomposition and orchestration.
"""

from .model_manager import AsyncLLMTokenManager
from .orchestrator_async import AsyncOrchestrator
from .data_models import SubTask, ToolTemplate
from .template_builder import TemplateBuilder
from .function_executor import FunctionExecutor

__version__ = "0.1.0"

__all__ = [
    "AsyncLLMTokenManager",
    "AsyncOrchestrator",
    "SubTask",
    "ToolTemplate",
    "TemplateBuilder",
    "FunctionExecutor"
] 
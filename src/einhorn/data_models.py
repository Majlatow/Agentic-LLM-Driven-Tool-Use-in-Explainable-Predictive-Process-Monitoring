from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime

@dataclass
class SubTask:
    index: int
    description: str
    depends_on: Optional[List[int]] = None
    resolved_inputs: Optional[Dict[str, str]] = None
    tool_name: Optional[str] = None
    required_inputs: Optional[List[str]] = None
    forwarded_inputs: Optional[Dict[int, List[str]]] = None
    input_mapping: Optional[str] = None
    missing_inputs: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)

@dataclass
class ToolTemplate:
    function_name: str
    module_path: str
    description: str
    parameters: Dict[str, str]
    template: str
    created_at: str = datetime.utcnow().isoformat()
    authored_by: Optional[str] = "unknown"
    tags: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)

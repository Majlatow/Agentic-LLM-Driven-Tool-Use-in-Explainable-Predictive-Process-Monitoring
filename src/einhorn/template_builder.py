import json
import hashlib
import os
from typing import List, Dict
from datetime import datetime
from .data_models import ToolTemplate

class TemplateBuilder:
    def __init__(self, cache_path: str = ".template_cache.json"):
        self.cache_path = cache_path
        self._cache: Dict[str, ToolTemplate] = {}
        self._load_cache()

    def _doc_hash(self, doc_json: str) -> str:
        return hashlib.sha256(doc_json.encode()).hexdigest()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                self._cache = {
                    h: ToolTemplate(**t) for h, t in raw.items()
                }

    def _save_cache(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump({h: t.to_dict() for h, t in self._cache.items()}, f, indent=2)

    def build_templates_from_json(self, doc_json: str, authored_by: str = "unknown", tags: List[str] = None) -> List[ToolTemplate]:
        doc_hash = self._doc_hash(doc_json)
        if doc_hash in self._cache:
            return [self._cache[doc_hash]]

        parsed = json.loads(doc_json)
        function_name = parsed["function_name"]
        module_path = parsed["module_path"]
        description = parsed["description"]
        parameters = parsed["parameters"]

        template_str = f"Call function `{function_name}` with: " + ", ".join(f"{k}={{{k}}}" for k in parameters)

        template = ToolTemplate(
            function_name=function_name,
            module_path=module_path,
            description=description,
            parameters=parameters,
            template=template_str,
            created_at=datetime.utcnow().isoformat(),
            authored_by=authored_by,
            tags=tags or []
        )

        self._cache[doc_hash] = template
        self._save_cache()

        return [template]

    def list_templates(self) -> Dict[str, ToolTemplate]:
        return self._cache

    def delete_template(self, doc_hash: str) -> bool:
        if doc_hash in self._cache:
            del self._cache[doc_hash]
            self._save_cache()
            return True
        return False

    def update_template(self, doc_hash: str, updates: Dict) -> bool:
        if doc_hash in self._cache:
            template = self._cache[doc_hash]
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            self._save_cache()
            return True
        return False

"""ToolRegistry — central registry for all tools available to agents."""
from __future__ import annotations
import logging
from typing import Iterator
from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
class ToolRegistry:
    def __init__(self): self._tools = {}
    def register(self, tool): self._tools[self._normalise(tool.name)] = tool
    def get(self, name): return self._tools.get(self._normalise(name))
    def all_tools(self): return iter(self._tools.values())
    def list_names(self): return sorted(self._tools.keys())
    @staticmethod
    def _normalise(name): return name.strip().lower().replace(" ","_").replace("-","_")
    def __len__(self): return len(self._tools)

"""Unit tests for ToolRegistry."""
import pytest
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.database import DatabaseTool
def make_registry():
    r = ToolRegistry(); r.register(CalculatorTool()); r.register(DatabaseTool()); return r
def test_register_get(): r = make_registry(); assert r.get("calculator") is not None
def test_missing(): return make_registry().get("nonexistent") is None
def test_normalization(): r = make_registry(); assert r.get("Calculator") is not None
def test_sorted(): r = make_registry(); n = r.list_names(); assert n == sorted(n)
def test_len(): assert len(make_registry()) == 2

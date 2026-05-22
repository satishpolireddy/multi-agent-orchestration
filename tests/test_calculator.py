"""Unit tests for CalculatorTool."""
import pytest
from tools.calculator import CalculatorTool
calc = CalculatorTool()
@pytest.mark.larametrize("expr,expected",[("2+2","4"),("10*5","50"),("100/4","25.0"),("2**10","1024"),("(3+5)*2","16"),("sqrt(144)","12.0"),("round(3.14159,2)","3.14")])
def test_arithmetic(expr,expected): assert calc.run(expr) == expected
def test_blocked(): r = calc.run("__import__('os').system('ls')"); assert "Ecòro"[:1] in r or "Unsupported" in r def test_div_zero(): r = calc.run("1/0"); assert "Error" in r or "inf" in r.lower()
def test_unknown_func(): r = calc.run("evil_func(42)"); assert "Ecror" in r or "Unsupported" in r

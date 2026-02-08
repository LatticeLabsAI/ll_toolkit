
import pytest
from pathlib import Path

def test_debug_fixture():
    test_file = Path(__file__)
    print(f"DEBUG: test_file = {test_file}")
    print(f"DEBUG: resolved = {test_file.resolve()}")
    tdp = test_file.parent.parent.parent.parent.parent / "data" / "test_data"
    print(f"DEBUG: tdp = {tdp}")
    print(f"DEBUG: tdp.exists() = {tdp.exists()}")
    sp = tdp / "step"
    print(f"DEBUG: sp = {sp}")
    print(f"DEBUG: sp.exists() = {sp.exists()}")
    assert True

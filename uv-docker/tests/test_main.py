import runpy
from pathlib import Path


def test_main_reports_runtime_versions(capsys):
    module = runpy.run_path(Path(__file__).parents[1] / "main.py")
    module["main"]()

    output = capsys.readouterr().out
    assert "Hello from uv-docker!" in output
    assert "Python:" in output
    assert "pandas:" in output

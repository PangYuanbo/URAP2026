import importlib.util
from pathlib import Path


def load_launcher():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("advanced_flow_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_registered_scripts_exist():
    launcher = load_launcher()
    missing = [str(path) for path in launcher.METHODS.values() if not path.exists()]
    assert not missing


def test_complete_method_progression_is_registered():
    launcher = load_launcher()
    assert set(launcher.METHODS) == {
        "baseline-homography",
        "nps-tvl1",
        "sea-raft",
        "double-stage",
        "parallax-robust",
        "evaluate-parallax",
    }

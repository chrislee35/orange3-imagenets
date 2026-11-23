import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_uses_hatchling_and_keeps_license():
    data = load()
    assert data["build-system"]["build-backend"] == "hatchling.build"
    assert data["project"]["license"] == "CC-BY-NC-SA-4.0"


def test_dev_dependencies():
    dev = {d.split()[0].split(">")[0] for d in load()["project"]["optional-dependencies"]["dev"]}
    assert dev >= {"black", "flake8", "build", "pytest", "twine", "pylint", "ruff"}


def test_line_length_is_140():
    tool = load()["tool"]
    assert tool["ruff"]["line-length"] == tool["black"]["line-length"] == 140
    assert tool["pylint"]["format"]["max-line-length"] == 140


def test_entry_points_resolve():
    import importlib

    for group in ("orange.widgets", "orange3.addon"):
        for target in load()["project"]["entry-points"][group].values():
            importlib.import_module(target)

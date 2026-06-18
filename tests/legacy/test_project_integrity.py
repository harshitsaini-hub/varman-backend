# type: ignore
# pyright: reportMissingImports=false
# pyright: reportUndefinedVariable=false
import ast
from pathlib import Path


def test_python_sources_parse():
    ignored_parts = {".git", ".venv", "venv", "datasets", "weights"}
    for path in Path(".").rglob("*.py"):
        if ignored_parts.intersection(path.parts):
            continue
        ast.parse(path.read_text(), filename=str(path))


def test_runtime_configuration_creates_storage_directories():
    import config

    assert Path(config.STORAGE_DIR).is_dir()
    assert Path(config.DB_DIR).is_dir()
    assert Path(config.TEMP_STORAGE_PATH).is_dir()
    assert config.REDIS_URL.startswith("redis://")


def test_requirements_are_unique_and_pinned():
    requirements = Path("requirements.txt").read_text().splitlines()
    package_names = []
    for line in requirements:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        package = stripped.split("==", maxsplit=1)[0].split("[", maxsplit=1)[0].lower()
        package_names.append(package)
        assert "==" in stripped, f"Unpinned requirement: {stripped}"

    assert len(package_names) == len(set(package_names))

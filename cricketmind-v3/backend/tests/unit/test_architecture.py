import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "cricketmind_v3"


def imports_in(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                names.append("(relative import)")  # we only use absolute imports
            else:
                names.append(node.module)
    return names


def imports_in_layer(layer: str) -> list[tuple[str, str]]:
    found = []
    for py_file in sorted((PACKAGE / layer).rglob("*.py")):
        for name in imports_in(py_file):
            found.append((py_file.name, name))
    return found


def test_domain_uses_only_the_standard_library():
    bad = []
    for file_name, name in imports_in_layer("domain"):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names:
            continue
        if name.startswith("cricketmind_v3.domain"):
            continue
        bad.append(file_name + " imports " + name)
    assert bad == []


def test_application_never_imports_outer_layers():
    outer = ["cricketmind_v3.infrastructure", "cricketmind_v3.pipeline", "cricketmind_v3.api"]
    bad = []
    for file_name, name in imports_in_layer("application"):
        if name == "(relative import)":
            bad.append(file_name + " uses a relative import")
            continue
        for layer in outer:
            if name.startswith(layer):
                bad.append(file_name + " imports " + name)
    assert bad == []


def test_every_layer_exists():
    for layer in ["domain", "application", "infrastructure", "pipeline", "api"]:
        assert (PACKAGE / layer / "__init__.py").exists(), layer

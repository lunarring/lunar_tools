from __future__ import annotations

import argparse
import ast
import textwrap
from pathlib import Path
from typing import Iterable, Tuple

BASE_DIR = Path(__file__).resolve().parent


def iter_example_files() -> Iterable[Path]:
    for path in sorted(BASE_DIR.rglob("*.py")):
        if path.name in {"__init__.py", "__main__.py"}:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def extract_docstring(path: Path) -> Tuple[str, str]:
    try:
        module = ast.parse(path.read_text())
    except SyntaxError as exc:  # pragma: no cover - should not happen
        return "", f"(failed to parse: {exc})"

    doc = ast.get_docstring(module)
    if not doc:
        return "", "(no module docstring)"

    clean = textwrap.dedent(doc).strip()
    summary = next((line.strip() for line in clean.splitlines() if line.strip()), "")
    return summary, clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List runnable Lunar Tools examples straight from their module docstrings."
    )
    parser.add_argument("--full", action="store_true", help="Print full docstrings instead of just the summary line.")
    parser.add_argument(
        "--filter",
        metavar="TEXT",
        help="Substring filter applied to file paths and docstrings (case-insensitive).",
    )
    args = parser.parse_args()

    filter_text = (args.filter or "").lower()

    for path in iter_example_files():
        summary, full_doc = extract_docstring(path)
        haystack = f"{path.relative_to(BASE_DIR)}\n{full_doc}".lower()
        if filter_text and filter_text not in haystack:
            continue

        rel_path = path.relative_to(BASE_DIR)
        if args.full:
            print(rel_path)
            print(textwrap.indent(full_doc, "  "))
            print()
        else:
            print(f"{rel_path}: {summary or full_doc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run P3 target/model tests without requiring pytest on the GPU server."""

from __future__ import annotations

import inspect
import runpy
from pathlib import Path


def main() -> None:
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "tests" / "test_model.py"))
    tests = [
        value
        for name, value in namespace.items()
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    for test in tests:
        test()
    print(f"P3 model/target smoke tests passed: {len(tests)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable P4 freeze artifacts")
    parser.add_argument("--freeze", default="results/p4/freeze_manifest.json")
    args = parser.parse_args()
    freeze_path = Path(args.freeze)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks = [
        (Path(freeze["sealed_manifest"]), freeze["sealed_manifest_sha256"]),
        (Path(freeze["sealed_test_path"]), freeze["sealed_test_sha256"]),
    ]
    checks.extend((Path(item["path"]), item["sha256"]) for item in freeze["checkpoints"])
    failures = []
    for path, expected in checks:
        actual = sha256(path) if path.is_file() else "missing"
        if actual != expected:
            failures.append({"path": path.as_posix(), "expected": expected, "actual": actual})
    if failures:
        raise ValueError(f"P4 freeze verification failed: {failures}")
    print(json.dumps({"freeze": freeze_path.as_posix(), "verified_files": len(checks), "status": "passed"}))


if __name__ == "__main__":
    main()

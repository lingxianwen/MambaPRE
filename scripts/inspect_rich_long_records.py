from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rich long-message field tiling")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--minimum", type=int, default=513)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    for value in args.inputs:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                length = len(bytes.fromhex(row["payload_hex"]))
                if length < args.minimum:
                    continue
                fields = row.get("fields", [])
                if args.compact:
                    payload = bytes.fromhex(row["payload_hex"])
                    gaps = []
                    for index, field in enumerate(fields):
                        if not any(
                            marker in str(field.get("name", "")).lower()
                            for marker in ("undissected", "opaque", "unknown")
                        ):
                            continue
                        off, width = int(field["off"]), int(field["len"])
                        nxt = fields[index + 1] if index + 1 < len(fields) else {}
                        gaps.append(
                            {
                                "off": off,
                                "len": width,
                                "little_endian_value": int.from_bytes(
                                    payload[off : off + width], "little", signed=True
                                ),
                                "next_name": nxt.get("name"),
                                "next_len": nxt.get("len"),
                            }
                        )
                    print(
                        json.dumps(
                            {
                                "path": path.as_posix(),
                                "id": row["id"],
                                "capture_id": row.get("capture_id"),
                                "length": length,
                                "field_count": len(fields),
                                "role_counts": dict(
                                    Counter(str(field.get("role")) for field in fields)
                                ),
                                "gaps": gaps,
                            },
                            separators=(",", ":"),
                        )
                    )
                    continue
                print(
                    json.dumps(
                        {
                            "path": path.as_posix(),
                            "id": row["id"],
                            "capture_id": row.get("capture_id"),
                            "length": length,
                            "field_count": len(fields),
                            "fields": [
                                {
                                    key: field.get(key)
                                    for key in ("off", "len", "name", "role", "type")
                                }
                                for field in fields
                            ],
                        },
                        separators=(",", ":"),
                    )
                )


if __name__ == "__main__":
    main()

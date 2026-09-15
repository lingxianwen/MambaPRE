from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scapy.all import PcapReader, TCP, UDP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(path: Path, minimum: int) -> dict:
    lengths = []
    qualifying_payload_hashes = []
    with PcapReader(str(path)) as reader:
        for packet in reader:
            layer = packet.getlayer(TCP) or packet.getlayer(UDP)
            if layer is None:
                continue
            payload = bytes(layer.payload)
            lengths.append(len(payload))
            if len(payload) >= minimum:
                qualifying_payload_hashes.append(hashlib.sha256(payload).hexdigest())
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "transport_payload_packets": len(lengths),
        "max_transport_payload": max(lengths, default=0),
        "packets_at_least_minimum": sum(length >= minimum for length in lengths),
        "qualifying_transport_payload_sha256": qualifying_payload_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-screen PCAP transport payload lengths")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--minimum", type=int, default=513)
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = []
    for value in args.inputs:
        path = Path(value)
        paths.extend(sorted(path.rglob("*.pcap*"))) if path.is_dir() else paths.append(path)
    report = {
        "minimum": args.minimum,
        "note": "transport-segment pre-screen only; passing does not imply full-field eligibility",
        "captures": [inspect(path, args.minimum) for path in paths],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

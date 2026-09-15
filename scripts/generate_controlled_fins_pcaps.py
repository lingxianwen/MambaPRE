from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
from pathlib import Path


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def ethernet_ipv4_udp(
    payload: bytes,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int = 9600,
    identification: int = 0,
) -> bytes:
    dst_mac = bytes.fromhex("02 00 00 00 00 02")
    src_mac = bytes.fromhex("02 00 00 00 00 01")
    ethernet = dst_mac + src_mac + struct.pack("!H", 0x0800)
    udp_length = 8 + len(payload)
    udp = struct.pack("!HHHH", src_port, dst_port, udp_length, 0)
    src = socket.inet_aton(src_ip)
    dst = socket.inet_aton(dst_ip)
    total_length = 20 + udp_length
    ip_without_checksum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_length,
        identification & 0xFFFF,
        0x4000,
        64,
        17,
        0,
        src,
        dst,
    )
    ip_header = ip_without_checksum[:10] + struct.pack(
        "!H", checksum(ip_without_checksum)
    ) + ip_without_checksum[12:]
    return ethernet + ip_header + udp + payload


def pcap_bytes(frames: list[bytes], capture_index: int) -> bytes:
    output = bytearray(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    base_time = 1_800_000_000 + capture_index * 60
    for index, frame in enumerate(frames):
        output.extend(
            struct.pack("<IIII", base_time + index, index * 1000, len(frame), len(frame))
        )
        output.extend(frame)
    return bytes(output)


def mutate_template(source: dict, capture_index: int, message_index: int) -> bytes:
    payload = bytearray.fromhex(source["payload_hex"])
    for field_index, field in enumerate(source["fields"]):
        name = str(field.get("name", ""))
        offset = int(field["off"])
        width = int(field["len"])
        if name == "sid":
            payload[offset] = (17 * capture_index + message_index) % 256
        elif name == "da1":
            payload[offset] = 20 + capture_index
        elif name == "sa1":
            payload[offset] = 120 + capture_index
        elif name == "unit_address":
            payload[offset] = (field_index + capture_index + message_index) % 255 or 1
        elif name == "model_number":
            model = f"CJ2M-C{capture_index:02d}-M{message_index:02d}-{field_index:02d}"
            encoded = model.encode("ascii")[:width].ljust(width, b" ")
            payload[offset : offset + width] = encoded
    return bytes(payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate controlled, standards-valid long FINS/UDP captures"
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--captures", type=int, default=10)
    parser.add_argument("--messages-per-capture", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    with Path(args.template).open(encoding="utf-8") as handle:
        source = json.loads(next(line for line in handle if line.strip()))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    captures = []
    for capture_index in range(args.captures):
        frames = []
        for message_index in range(args.messages_per_capture):
            fins = mutate_template(source, capture_index, message_index)
            frames.append(
                ethernet_ipv4_udp(
                    fins,
                    src_ip=f"10.20.{capture_index}.10",
                    dst_ip=f"10.30.{capture_index}.20",
                    src_port=20000 + 100 * capture_index + message_index,
                    identification=args.seed + 100 * capture_index + message_index,
                )
            )
        path = output / f"controlled_fins_capture_{capture_index:02d}.pcap"
        path.write_bytes(pcap_bytes(frames, capture_index))
        captures.append(
            {
                "path": path.name,
                "sha256": sha256(path),
                "messages": len(frames),
                "message_length": len(source["payload_hex"]) // 2,
            }
        )

    manifest = {
        "source_kind": "controlled_valid_protocol_traffic",
        "not_operational_traffic": True,
        "protocol": "Omron FINS over UDP",
        "seed": args.seed,
        "template_record_id": source.get("id"),
        "template_capture_id": source.get("capture_id"),
        "template_capture_sha256": source.get("capture_sha256"),
        "mutation_scope": [
            "FINS service ID",
            "source/destination node",
            "unit addresses",
            "20-byte model-number fields",
        ],
        "captures": captures,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

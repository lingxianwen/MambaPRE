from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mambapre.neupre import stable_session_id


@dataclass(frozen=True)
class ProtocolConfig:
    pcap: str
    display_filter: str
    decode_as: tuple[str, ...]
    server_port: int


PROTOCOLS = {
    "modbus": ProtocolConfig(
        "libmodbus-bandwidth_server-rand_client.pcap",
        "mbtcp",
        ("tcp.port==1502,mbtcp",),
        1502,
    ),
    "dnp3": ProtocolConfig("dnp3.pcap", "dnp3", ("tcp.port==20000,dnp3",), 20000),
    "s7comm": ProtocolConfig("s7comm.pcap", "s7comm", (), 102),
    "iec104": ProtocolConfig(
        "iec104.pcap", "104apci", ("tcp.port==2404,104apci",), 2404
    ),
    "omron": ProtocolConfig("omron.pcap", "omron", ("tcp.port==9600,omron",), 9600),
    "delta": ProtocolConfig("delta.pcap", "mbtcp", ("tcp.port==502,mbtcp",), 502),
    "lon": ProtocolConfig("lon.pcap", "lon", (), 1628),
    "eip": ProtocolConfig("ethernet_ip.pcap", "enip || cip", (), 44818),
    "bacnet": ProtocolConfig("bacnet.pcap", "bacnet || bacapp", (), 47808),
    "opcua": ProtocolConfig(
        "opcua.pcap", "opcua", ("tcp.port==4840,opcua",), 4840
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_id(protocol: str, payload_hex: str) -> str:
    digest = hashlib.sha256(bytes.fromhex(payload_hex)).hexdigest()[:20]
    return f"neupre-{protocol}-{digest}"


def normalize_opcua_fields(payload_hex: str, source_fields: list[dict]) -> list[dict]:
    """Refine OPC UA PDML labels without guessing undisclosed payload spans.

    Wireshark exposes the bytes of a String/ByteString as a leaf but omits its
    four-byte Int32 length prefix from the leaf list.  We recover only prefixes
    whose little-endian value exactly equals the immediately following leaf
    width.  Every other unknown span stays unknown and is rejected by the
    strict full-field audit.
    """
    payload = bytes.fromhex(payload_hex)
    fields = [dict(field) for field in source_fields]
    for index, field in enumerate(fields):
        name = str(field.get("name", ""))
        lowered = name.lower()
        off, width = int(field["off"]), int(field["len"])
        if "undissected" in lowered and width == 4 and index + 1 < len(fields):
            following = fields[index + 1]
            value = int.from_bytes(payload[off : off + width], "little", signed=True)
            if int(following["off"]) == off + width and value == int(following["len"]):
                field.update(
                    {
                        "name": f"verified_length_prefix_{following.get('name', 'value')}",
                        "role": "length",
                        "type": "bytes",
                        "value": str(value),
                    }
                )
                lowered = str(field["name"]).lower()

        if "verified_length_prefix" in lowered or "transport_size" in lowered:
            field["role"] = "length"
        elif "arraysize" in lowered or "array_size" in lowered:
            field["role"] = "count"
        elif any(
            marker in lowered
            for marker in (
                "transport_scid",
                "security_tokenid",
                "security_seq",
                "security_rqid",
                "requesthandle",
                "subscriptionid",
                "sequencenumber",
                "clienthandle",
                "monitoreditemid",
            )
        ):
            field["role"] = "sequence"
        elif any(
            marker in lowered
            for marker in (
                "encodingmask",
                "_mask",
                "transport_chunk",
                "variant_has_value",
                "morenotifications",
            )
        ):
            field["role"] = "constant"
        elif any(
            marker in lowered
            for marker in ("nodeid", "nsindex", "namespaceindex")
        ):
            field["role"] = "address"
    return fields


def _load_pdml_module(module_dir: Path):
    sys.path.insert(0, str(module_dir))
    try:
        return importlib.import_module("pdml_dissector")
    finally:
        sys.path.pop(0)


def _record_from_pdml(
    protocol: str,
    config: ProtocolConfig,
    pcap: Path,
    result,
    capture_id: str | None = None,
    capture_sha256: str | None = None,
    source_kind: str = "real_capture",
) -> dict:
    meta = dict(result.meta)
    fields = result.schema["fields"]
    annotation_source = "Wireshark/tshark PDML via the local PDML parser"
    if protocol == "opcua":
        fields = normalize_opcua_fields(result.payload_hex, fields)
        annotation_source += " + verified OPC UA Int32 length-prefix reconstruction"
    return {
        "id": _sample_id(protocol, result.payload_hex),
        "protocol": protocol,
        "source_protocol": protocol,
        "direction": meta.get("direction", "?"),
        "payload_hex": result.payload_hex,
        "fields": fields,
        "capture_id": capture_id or pcap.name,
        "capture_sha256": capture_sha256 or _sha256(pcap),
        "source_kind": source_kind,
        "session_id": stable_session_id(meta),
        "packet_index": meta.get("pkt_idx"),
        "annotation_scope": "full_field_pdml",
        "annotation_source": annotation_source,
        "capture_meta": meta,
        "server_port": config.server_port,
    }


def generate_full_records(
    pdml,
    protocol: str,
    config: ProtocolConfig,
    pcap: Path,
    capture_id: str | None = None,
    source_kind: str = "real_capture",
) -> list[dict]:
    records = {}
    capture_sha256 = _sha256(pcap)
    for result in pdml.dissect_pcap(
        str(pcap),
        display_filter=config.display_filter,
        decode_as=list(config.decode_as),
        server_ports=[config.server_port],
        max_packets=None,
    ):
        record = _record_from_pdml(
            protocol, config, pcap, result, capture_id, capture_sha256, source_kind
        )
        records.setdefault(record["payload_hex"], record)
    return list(records.values())


def _tshark_rows(tshark: str, pcap: Path) -> list[list[str]]:
    command = [
        tshark,
        "-r",
        str(pcap),
        "-Y",
        "tcp.payload",
        "-T",
        "fields",
        "-E",
        "separator=|",
        "-E",
        "occurrence=f",
    ]
    for field in (
        "frame.number",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "tcp.payload",
    ):
        command.extend(("-e", field))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [line.split("|", 5) for line in result.stdout.splitlines() if line.strip()]


def generate_s7comm_plus_envelope(tshark: str, pcap: Path) -> list[dict]:
    records = {}
    for columns in _tshark_rows(tshark, pcap):
        if len(columns) != 6:
            continue
        packet_index, src_ip, dst_ip, src_port, dst_port, payload_hex = columns
        payload_hex = payload_hex.replace(":", "")
        try:
            message = bytes.fromhex(payload_hex)
        except ValueError:
            continue
        if not (
            len(message) >= 7
            and message[:2] == b"\x03\x00"
            and int.from_bytes(message[2:4], "big") == len(message)
            and message[4] >= 2
        ):
            continue
        meta = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(src_port),
            "dst_port": int(dst_port),
            "pkt_idx": int(packet_index),
            "proto": "tcp",
        }
        meta["direction"] = "c2s" if meta["dst_port"] == 102 else "s2c"
        fields = [
            {"off": 0, "len": 1, "name": "tpkt_version", "role": "constant", "type": "uint8"},
            {"off": 1, "len": 1, "name": "tpkt_reserved", "role": "constant", "type": "uint8"},
            {"off": 2, "len": 2, "name": "tpkt_length", "role": "length", "type": "uint16_be"},
            {"off": 4, "len": 1, "name": "cotp_length_indicator", "role": "length", "type": "uint8"},
            {"off": 5, "len": 1, "name": "cotp_pdu_type", "role": "opcode", "type": "uint8"},
            {"off": 6, "len": 1, "name": "cotp_tpdu_number", "role": "sequence", "type": "uint8"},
            {
                "off": 7,
                "len": len(message) - 7,
                "name": "opaque_s7comm_plus_payload",
                "role": "payload",
                "type": "bytes",
            },
        ]
        record = {
            "id": _sample_id("s7comm_plus", payload_hex),
            "protocol": "s7comm_plus",
            "source_protocol": "s7comm_plus",
            "direction": meta["direction"],
            "payload_hex": payload_hex,
            "fields": fields,
            "capture_id": pcap.name,
            "session_id": stable_session_id(meta),
            "packet_index": meta["pkt_idx"],
            "annotation_scope": "rfc1006_cotp_envelope_only",
            "annotation_source": "RFC 1006 TPKT/COTP envelope over real NeuPRE PCAP",
            "capture_meta": meta,
            "server_port": 102,
        }
        records.setdefault(payload_hex, record)
    return list(records.values())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate NeuPRE PCAP labels with tshark PDML")
    parser.add_argument("--pcap-dir", required=True)
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--pdml-module-dir", required=True)
    parser.add_argument("--tshark", default=r"C:\Program Files\Wireshark\tshark.exe")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument(
        "--pcap-override",
        action="append",
        default=[],
        metavar="PROTOCOL=PATH",
        help="override a configured PCAP; repeat for multiple protocols",
    )
    parser.add_argument(
        "--capture-id-prefix",
        default="",
        help="provenance prefix used to distinguish captures with the same basename",
    )
    parser.add_argument(
        "--source-kind",
        default="real_capture",
        choices=("real_capture", "public_real_capture", "controlled_valid_protocol_traffic"),
    )
    parser.add_argument("--include-s7comm-plus-envelope", action="store_true")
    args = parser.parse_args()
    os.environ["TSHARK_EXE"] = args.tshark

    pcap_dir = Path(args.pcap_dir)
    records_dir = Path(args.records_dir)
    pdml = _load_pdml_module(Path(args.pdml_module_dir))
    selected = [item.strip() for item in args.protocols.split(",") if item.strip()]
    unknown = set(selected) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocols: {sorted(unknown)}")

    overrides = {}
    for value in args.pcap_override:
        if "=" not in value:
            raise ValueError(f"invalid --pcap-override {value!r}; expected PROTOCOL=PATH")
        protocol, path = value.split("=", 1)
        protocol = protocol.strip()
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown override protocol: {protocol}")
        overrides[protocol] = Path(path)

    manifest = {"protocols": {}, "pcaps": {}, "tshark": args.tshark}
    for protocol in selected:
        config = PROTOCOLS[protocol]
        pcap = overrides.get(protocol, pcap_dir / config.pcap)
        capture_id = f"{args.capture_id_prefix}{pcap.name}"
        rows = generate_full_records(
            pdml, protocol, config, pcap, capture_id, args.source_kind
        )
        _write_jsonl(records_dir / f"{protocol}.jsonl", rows)
        manifest["protocols"][protocol] = {
            "records": len(rows),
            "max_length": max((len(bytes.fromhex(row["payload_hex"])) for row in rows), default=0),
            "annotation_scope": "full_field_pdml",
            "capture_id": capture_id,
            "source_path": str(pcap),
            "source_kind": args.source_kind,
        }
        manifest["pcaps"][pcap.name] = {"bytes": pcap.stat().st_size, "sha256": _sha256(pcap)}
        print(json.dumps({"protocol": protocol, **manifest["protocols"][protocol]}))

    if args.include_s7comm_plus_envelope:
        pcap = pcap_dir / "s7comm_plus.pcap"
        rows = generate_s7comm_plus_envelope(args.tshark, pcap)
        _write_jsonl(records_dir / "s7comm_plus_envelope.jsonl", rows)
        manifest["protocols"]["s7comm_plus"] = {
            "records": len(rows),
            "max_length": max((len(bytes.fromhex(row["payload_hex"])) for row in rows), default=0),
            "long_over_512": sum(len(bytes.fromhex(row["payload_hex"])) > 512 for row in rows),
            "annotation_scope": "rfc1006_cotp_envelope_only",
        }
        manifest["pcaps"][pcap.name] = {"bytes": pcap.stat().st_size, "sha256": _sha256(pcap)}
        print(json.dumps({"protocol": "s7comm_plus", **manifest["protocols"]["s7comm_plus"]}))

    try:
        version = subprocess.run(
            [args.tshark, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        version = "unavailable"
    manifest["tshark_version"] = version
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .constants import length_bucket
from .data import load_jsonl, normalized_row

UNKNOWN_LABEL = -100

PROTOCOL_ALIASES = {
    "modbus": "modbus",
    "delta": "modbus_delta",
    "dnp3": "dnp3",
    "s7comm": "s7comm",
    "s7comm_plus": "s7comm_plus",
    "iec104": "iec104",
    "omron": "omron_fins",
    "lon": "lon",
    "eip": "ethernet_ip",
    "bacnet": "bacnet",
}


def stable_session_id(meta: dict) -> str:
    """Return a direction-independent identifier for one captured transport flow."""
    first = f"{meta.get('src_ip', '?')}:{meta.get('src_port', '?')}"
    second = f"{meta.get('dst_ip', '?')}:{meta.get('dst_port', '?')}"
    canonical = "|".join(sorted((first, second)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def boundary_only_row(
    sample_id: str,
    protocol: str,
    byte_values: bytes,
    boundaries: Iterable[int],
    direction: str = "?",
) -> dict:
    """Normalize boundary-only labels without inventing role or type targets."""
    n = len(byte_values)
    points = sorted({int(value) for value in boundaries})
    if not n:
        raise ValueError(f"{sample_id}: empty message")
    if any(value < 0 or value > n for value in points):
        raise ValueError(f"{sample_id}: boundary outside [0, {n}]")
    internal = [value for value in points if 0 < value < n]
    return {
        "id": sample_id,
        "protocol": protocol,
        "direction": direction,
        "bytes": list(byte_values),
        "boundaries": internal,
        "role_ids": [UNKNOWN_LABEL] * n,
        "type_ids": [UNKNOWN_LABEL] * n,
    }


def _copy_metadata(row: dict, source: dict) -> None:
    for key in (
        "capture_id",
        "session_id",
        "packet_index",
        "annotation_scope",
        "annotation_source",
        "source_protocol",
        "capture_sha256",
        "source_kind",
    ):
        if key in source:
            row[key] = source[key]


def convert_rich_record(source: dict) -> dict:
    protocol = PROTOCOL_ALIASES.get(
        str(source.get("protocol", source.get("source_protocol", "unknown"))).lower(),
        str(source.get("protocol", source.get("source_protocol", "unknown"))).lower(),
    )
    payload_hex = str(source["payload_hex"]).replace(":", "").replace(" ", "")
    byte_values = bytes.fromhex(payload_hex)
    fields = source.get("fields")
    if fields:
        row = normalized_row(
            str(source["id"]),
            protocol,
            str(source.get("direction", "?")),
            byte_values,
            fields,
        )
        row["field_spans"] = [
            {
                key: field[key]
                for key in ("off", "len", "name", "role", "type")
                if key in field
            }
            for field in sorted(fields, key=lambda item: int(item["off"]))
        ]
    else:
        row = boundary_only_row(
            str(source["id"]),
            protocol,
            byte_values,
            source["boundaries"],
            str(source.get("direction", "?")),
        )
    _copy_metadata(row, source)
    return row


def load_rich_records(records_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(records_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(convert_rich_record(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no JSONL records found in {records_dir}")
    return rows


def load_boundary_maps(pdml_dir: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(pdml_dir).glob("*.json")):
        source_protocol = path.stem.lower()
        protocol = PROTOCOL_ALIASES.get(source_protocol, source_protocol)
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for index, (payload_hex, boundaries) in enumerate(mapping.items()):
            row = boundary_only_row(
                f"neupre-map-{source_protocol}-{index:06d}",
                protocol,
                bytes.fromhex(payload_hex),
                boundaries,
            )
            row.update(
                capture_id=f"{source_protocol}.pcap",
                annotation_scope="boundary_only_pdml_map",
                annotation_source="NeuPRE gen_pdml_gt.py via Wireshark/tshark",
                source_protocol=source_protocol,
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"no JSON maps found in {pdml_dir}")
    return rows


def _identity(row: dict) -> tuple[str, bytes]:
    return str(row["protocol"]), bytes(row["bytes"])


def _deduplicate(rows: Sequence[dict]) -> list[dict]:
    seen = set()
    output = []
    for row in rows:
        identity = _identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(row)
    return output


def _reference_identities(paths: Sequence[str | Path]) -> tuple[set[tuple[str, bytes]], set[bytes]]:
    protocol_bytes: set[tuple[str, bytes]] = set()
    byte_values: set[bytes] = set()
    for path in paths:
        for sample in load_jsonl(path):
            value = bytes(sample.byte_values)
            protocol_bytes.add((sample.protocol, value))
            byte_values.add(value)
    return protocol_bytes, byte_values


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def session_disjoint_calibration_test(
    rows: Sequence[dict], calibration_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Partition complete sessions, never individual messages, for calibration."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        session = row.get("session_id")
        if not session:
            raise ValueError("session-disjoint split requires session_id on every row")
        groups.setdefault(str(session), []).append(row)
    if len(groups) < 2:
        raise ValueError("session-disjoint split requires at least two sessions")
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )
    target = len(rows) * calibration_fraction
    selected = set()
    selected_rows = 0
    for session in ordered[:-1]:
        if selected_rows >= target:
            break
        selected.add(session)
        selected_rows += len(groups[session])
    calibration = [row for row in rows if str(row["session_id"]) in selected]
    test = [row for row in rows if str(row["session_id"]) not in selected]
    if not calibration or not test:
        raise RuntimeError("failed to produce non-empty session-disjoint partitions")
    return calibration, test


def _stats(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "protocols": {},
            "annotation_scopes": {},
            "length_buckets": {},
            "max_length": 0,
            "unique_messages": 0,
            "captures": 0,
            "sessions": 0,
        }
    return {
        "n": len(rows),
        "protocols": dict(Counter(row["protocol"] for row in rows)),
        "annotation_scopes": dict(
            Counter(row.get("annotation_scope", "unknown") for row in rows)
        ),
        "length_buckets": dict(
            Counter(length_bucket(len(row["bytes"])) for row in rows)
        ),
        "max_length": max(len(row["bytes"]) for row in rows),
        "unique_messages": len({_identity(row) for row in rows}),
        "captures": len({row.get("capture_id") for row in rows if row.get("capture_id")}),
        "sessions": len({row.get("session_id") for row in rows if row.get("session_id")}),
    }


def prepare_neupre_dataset(
    output_dir: str | Path,
    records_dir: str | Path | None = None,
    pdml_dir: str | Path | None = None,
    reference_paths: Sequence[str | Path] = (),
    calibration_fraction: float = 0.3,
    seed: int = 1337,
) -> dict:
    """Prepare scope-separated external tests from rich records or boundary maps.

    Full-field PDML and coarse S7comm+ envelope labels are intentionally written
    to different files. They must never be pooled into one headline metric.
    """
    if bool(records_dir) == bool(pdml_dir):
        raise ValueError("provide exactly one of records_dir or pdml_dir")
    rows = load_rich_records(records_dir) if records_dir else load_boundary_maps(pdml_dir)
    rows = _deduplicate(rows)
    reference_protocol_bytes, reference_bytes = _reference_identities(reference_paths)

    envelope = [
        row for row in rows if row.get("annotation_scope") == "rfc1006_cotp_envelope_only"
    ]
    full = [
        row for row in rows if row.get("annotation_scope") != "rfc1006_cotp_envelope_only"
    ]
    long_envelope_novel = [
        row
        for row in envelope
        if len(row["bytes"]) > 512 and bytes(row["bytes"]) not in reference_bytes
    ]
    envelope_calibration, envelope_test = ([], [])
    if long_envelope_novel:
        envelope_calibration, envelope_test = session_disjoint_calibration_test(
            long_envelope_novel, calibration_fraction, seed
        )
    splits = {
        "test_external_full": full,
        "test_external_full_novel": [
            row for row in full if bytes(row["bytes"]) not in reference_bytes
        ],
        "test_external_long_full": [row for row in full if len(row["bytes"]) > 512],
        "test_external_long_full_novel": [
            row
            for row in full
            if len(row["bytes"]) > 512 and bytes(row["bytes"]) not in reference_bytes
        ],
        "test_external_envelope": envelope,
        "test_external_envelope_novel": [
            row for row in envelope if bytes(row["bytes"]) not in reference_bytes
        ],
        "test_external_long_envelope": [
            row for row in envelope if len(row["bytes"]) > 512
        ],
        "test_external_long_envelope_novel": long_envelope_novel,
        "calibration_long_envelope_novel": envelope_calibration,
        "test_long_envelope_session_disjoint_novel": envelope_test,
    }
    output = Path(output_dir)
    for name, split_rows in splits.items():
        _write_jsonl(output / f"{name}.jsonl", split_rows)

    stats = {name: _stats(split_rows) for name, split_rows in splits.items()}
    stats["overlap_audit"] = {
        "reference_paths": [str(path) for path in reference_paths],
        "external_exact_protocol_and_bytes": sum(
            _identity(row) in reference_protocol_bytes for row in rows
        ),
        "external_exact_bytes_regardless_of_protocol": sum(
            bytes(row["bytes"]) in reference_bytes for row in rows
        ),
        "full_envelope_exact_identity_overlap": len(
            {_identity(row) for row in full} & {_identity(row) for row in envelope}
        ),
        "long_envelope_calibration_test_session_overlap": len(
            {row.get("session_id") for row in envelope_calibration}
            & {row.get("session_id") for row in envelope_test}
        ),
        "long_envelope_calibration_test_exact_identity_overlap": len(
            {_identity(row) for row in envelope_calibration}
            & {_identity(row) for row in envelope_test}
        ),
    }
    stats["evidence_boundary"] = {
        "full_field_long_messages_over_512": len(splits["test_external_long_full"]),
        "full_field_novel_long_messages_over_512": len(
            splits["test_external_long_full_novel"]
        ),
        "envelope_only_long_messages_over_512": len(
            splits["test_external_long_envelope"]
        ),
        "envelope_only_novel_long_messages_over_512": len(
            splits["test_external_long_envelope_novel"]
        ),
        "warning": (
            "Envelope-only S7comm+ labels cover TPKT/COTP and opaque payload; "
            "do not pool them with full-field PDML metrics. Exact-byte novelty "
            "does not prove capture- or session-level independence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


FORBIDDEN_FIELD_MARKERS = (
    "undissected",
    "opaque",
    "unknown_bytes",
    "unparsed",
)

# These names can describe a transport envelope but do not demonstrate that the
# application protocol itself was dissected.  The list is intentionally narrow:
# an unfamiliar field counts as application evidence rather than being rejected.
ENVELOPE_FIELD_PREFIXES = ("tpkt_", "cotp_")
ENVELOPE_FIELD_NAMES = {
    "tpkt.version",
    "tpkt.reserved",
    "tpkt.length",
    "cotp.length",
    "cotp.type",
    "cotp.tpdu-number",
}


@dataclass(frozen=True)
class AuditDecision:
    accepted: bool
    reasons: tuple[str, ...]
    length: int
    field_count: int
    structural_field_count: int
    application_evidence_field_count: int
    largest_field_fraction: float
    exact_reference_overlap: bool


def _payload_bytes(row: dict) -> bytes:
    if "payload_hex" in row:
        value = str(row["payload_hex"]).replace(":", "").replace(" ", "")
        return bytes.fromhex(value)
    if "bytes" in row:
        return bytes(int(value) for value in row["bytes"])
    raise ValueError("record has neither payload_hex nor bytes")


def _fields(row: dict) -> list[dict]:
    value = row.get("fields", row.get("field_spans", []))
    return sorted(value, key=lambda field: int(field["off"]))


def _is_forbidden_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(marker in normalized for marker in FORBIDDEN_FIELD_MARKERS)


def _is_envelope_field(name: str) -> bool:
    normalized = name.lower()
    underscored = normalized.replace(".", "_").replace("-", "_")
    return normalized in ENVELOPE_FIELD_NAMES or underscored.startswith(
        ENVELOPE_FIELD_PREFIXES
    )


def audit_record(
    row: dict,
    reference_bytes: set[bytes] | None = None,
    min_length: int = 513,
    min_fields: int = 5,
    min_structural_fields: int = 3,
) -> AuditDecision:
    """Apply the evidence gate for a genuinely long, fully labelled message.

    Acceptance requires exact, non-overlapping byte tiling by explicit dissector
    fields.  Gap-fill/opaque fields, envelope-only annotations, and exact byte
    duplicates of the training pool are rejected.
    """

    reasons: list[str] = []
    try:
        payload = _payload_bytes(row)
    except (TypeError, ValueError) as exc:
        return AuditDecision(False, (f"invalid_payload:{exc}",), 0, 0, 0, 0, 0.0, False)

    fields = _fields(row)
    if len(payload) < min_length:
        reasons.append("too_short")
    if row.get("annotation_scope") != "full_field_pdml":
        reasons.append("not_full_field_pdml")
    if len(fields) < min_fields:
        reasons.append("too_few_fields")

    cursor = 0
    largest_width = 0
    forbidden = False
    structural = 0
    application_evidence = 0
    for index, field in enumerate(fields):
        try:
            offset = int(field["off"])
            width = int(field["len"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"invalid_field_{index}")
            continue
        if offset != cursor or width <= 0 or offset + width > len(payload):
            reasons.append(f"non_contiguous_field_{index}")
        cursor = max(cursor, offset + max(width, 0))
        largest_width = max(largest_width, width)
        name = str(field.get("name", ""))
        if _is_forbidden_name(name):
            forbidden = True
        if str(field.get("role", "payload")) != "payload":
            structural += 1
        if name and not _is_envelope_field(name) and not _is_forbidden_name(name):
            application_evidence += 1

    if cursor != len(payload):
        reasons.append("incomplete_byte_coverage")
    if forbidden:
        reasons.append("gap_or_opaque_field")
    if structural < min_structural_fields:
        reasons.append("too_few_structural_fields")
    if application_evidence == 0:
        reasons.append("envelope_only_fields")

    overlap = payload in (reference_bytes or set())
    if overlap:
        reasons.append("exact_reference_overlap")

    reasons = list(dict.fromkeys(reasons))
    return AuditDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        length=len(payload),
        field_count=len(fields),
        structural_field_count=structural,
        application_evidence_field_count=application_evidence,
        largest_field_fraction=(largest_width / len(payload) if payload else 0.0),
        exact_reference_overlap=overlap,
    )


def load_reference_bytes(paths: Sequence[str | Path]) -> set[bytes]:
    values: set[bytes] = set()
    for source in paths:
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    values.add(_payload_bytes(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return values


def audit_paths(
    paths: Iterable[str | Path],
    reference_paths: Sequence[str | Path] = (),
    min_length: int = 513,
    min_fields: int = 5,
    min_structural_fields: int = 3,
) -> tuple[list[dict], dict]:
    reference = load_reference_bytes(reference_paths)
    accepted: list[dict] = []
    decisions: list[tuple[dict, AuditDecision, str, int]] = []
    seen_payloads: set[bytes] = set()

    for source in paths:
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                decision = audit_record(
                    row,
                    reference_bytes=reference,
                    min_length=min_length,
                    min_fields=min_fields,
                    min_structural_fields=min_structural_fields,
                )
                payload = _payload_bytes(row)
                if decision.accepted and payload in seen_payloads:
                    decision = AuditDecision(
                        **{
                            **asdict(decision),
                            "accepted": False,
                            "reasons": ("duplicate_within_candidate_pool",),
                        }
                    )
                if decision.accepted:
                    seen_payloads.add(payload)
                    accepted.append(row)
                decisions.append((row, decision, str(path), line_number))

    reason_counts = Counter(
        reason for _, decision, _, _ in decisions for reason in decision.reasons
    )
    accepted_decisions = [decision for _, decision, _, _ in decisions if decision.accepted]
    capture_ids = {
        str(row.get("capture_id")) for row in accepted if row.get("capture_id")
    }
    session_ids = {
        str(row.get("session_id")) for row in accepted if row.get("session_id")
    }
    report = {
        "criteria": {
            "min_length": min_length,
            "min_fields": min_fields,
            "min_structural_fields": min_structural_fields,
            "required_annotation_scope": "full_field_pdml",
            "forbidden_field_markers": list(FORBIDDEN_FIELD_MARKERS),
            "requires_exact_contiguous_byte_tiling": True,
            "rejects_exact_reference_bytes": True,
        },
        "candidate_records": len(decisions),
        "accepted_records": len(accepted),
        "rejected_records": len(decisions) - len(accepted),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "accepted_protocols": dict(
            Counter(str(row.get("protocol", "unknown")) for row in accepted)
        ),
        "accepted_captures": len(capture_ids),
        "accepted_sessions": len(session_ids),
        "accepted_min_length": min(
            (decision.length for decision in accepted_decisions), default=0
        ),
        "accepted_max_length": max(
            (decision.length for decision in accepted_decisions), default=0
        ),
        "accepted_max_largest_field_fraction": max(
            (decision.largest_field_fraction for decision in accepted_decisions),
            default=0.0,
        ),
        "reference_files": [str(path) for path in reference_paths],
        "reference_unique_messages": len(reference),
        "capture_and_session_split_ready": len(capture_ids) >= 2 and len(session_ids) >= 2,
        "accepted_examples": [
            {
                "id": row.get("id"),
                "protocol": row.get("protocol"),
                "capture_id": row.get("capture_id"),
                "session_id": row.get("session_id"),
                **asdict(decision),
            }
            for row, decision, _, _ in decisions
            if decision.accepted
        ][:20],
    }
    return accepted, report


def stable_group_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()

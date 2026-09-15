from mambapre.full_field import audit_record


def record(fields, length=520, scope="full_field_pdml"):
    return {
        "id": "sample",
        "protocol": "toy",
        "payload_hex": "00" * length,
        "annotation_scope": scope,
        "fields": fields,
    }


def test_accepts_contiguous_explicit_long_fields():
    fields = [
        {"off": 0, "len": 1, "name": "version", "role": "constant"},
        {"off": 1, "len": 2, "name": "length", "role": "length"},
        {"off": 3, "len": 1, "name": "opcode", "role": "opcode"},
        {"off": 4, "len": 4, "name": "address", "role": "address"},
        {"off": 8, "len": 512, "name": "response_data", "role": "payload"},
    ]
    decision = audit_record(record(fields))
    assert decision.accepted
    assert decision.length == 520


def test_rejects_pdml_gap_fill_even_when_bytes_are_tiled():
    fields = [
        {"off": 0, "len": 1, "name": "version", "role": "constant"},
        {"off": 1, "len": 2, "name": "length", "role": "length"},
        {"off": 3, "len": 1, "name": "opcode", "role": "opcode"},
        {"off": 4, "len": 4, "name": "address", "role": "address"},
        {"off": 8, "len": 512, "name": "undissected_bytes", "role": "payload"},
    ]
    decision = audit_record(record(fields))
    assert not decision.accepted
    assert "gap_or_opaque_field" in decision.reasons


def test_rejects_exact_training_overlap_and_envelope_scope():
    fields = [
        {"off": 0, "len": 1, "name": "tpkt_version", "role": "constant"},
        {"off": 1, "len": 1, "name": "tpkt_reserved", "role": "constant"},
        {"off": 2, "len": 2, "name": "tpkt_length", "role": "length"},
        {"off": 4, "len": 1, "name": "cotp_type", "role": "opcode"},
        {"off": 5, "len": 515, "name": "opaque_payload", "role": "payload"},
    ]
    row = record(fields, scope="rfc1006_cotp_envelope_only")
    decision = audit_record(row, reference_bytes={bytes(520)})
    assert not decision.accepted
    assert "not_full_field_pdml" in decision.reasons
    assert "gap_or_opaque_field" in decision.reasons
    assert "exact_reference_overlap" in decision.reasons

from scripts.generate_neupre_records import normalize_opcua_fields


def test_verified_opcua_length_prefix_is_reconstructed():
    payload = (4).to_bytes(4, "little") + b"test"
    fields = [
        {"off": 0, "len": 4, "name": "undissected_bytes", "role": "payload", "type": "bytes"},
        {"off": 4, "len": 4, "name": "EndpointUrl", "role": "payload", "type": "bytes"},
    ]
    refined = normalize_opcua_fields(payload.hex(), fields)
    assert refined[0]["name"] == "verified_length_prefix_EndpointUrl"
    assert refined[0]["role"] == "length"


def test_unverified_opcua_gap_remains_rejectable():
    payload = (99).to_bytes(4, "little") + b"test"
    fields = [
        {"off": 0, "len": 4, "name": "undissected_bytes", "role": "payload", "type": "bytes"},
        {"off": 4, "len": 4, "name": "EndpointUrl", "role": "payload", "type": "bytes"},
    ]
    refined = normalize_opcua_fields(payload.hex(), fields)
    assert refined[0]["name"] == "undissected_bytes"


def test_opcua_structural_roles_are_protocol_aware():
    fields = [
        {"off": 0, "len": 4, "name": "transport_size", "role": "payload", "type": "bytes"},
        {"off": 4, "len": 4, "name": "variant_ArraySize", "role": "payload", "type": "bytes"},
        {"off": 8, "len": 4, "name": "security_seq", "role": "payload", "type": "bytes"},
        {"off": 12, "len": 1, "name": "nodeid_encodingmask", "role": "payload", "type": "uint8"},
    ]
    refined = normalize_opcua_fields(bytes(13).hex(), fields)
    assert [field["role"] for field in refined] == [
        "length",
        "count",
        "sequence",
        "constant",
    ]

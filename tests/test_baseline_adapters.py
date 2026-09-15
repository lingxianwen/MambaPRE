import json

from baselines.common import load_rows, spans_to_cuts, tile_lengths
from baselines.run_binaryinferno import parse_spec


def test_load_normalized_rows(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "m1",
                "protocol": "demo",
                "direction": "c2s",
                "bytes": [1, 2, 3, 4],
                "boundaries": [0, 1, 3, 4],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_rows(path)
    assert rows[0]["raw"] == b"\x01\x02\x03\x04"
    assert rows[0]["gold_cuts"] == {1, 3}


def test_tile_lengths_and_cuts():
    spans = tile_lengths([1, 2, None], 7)
    assert spans == [(0, 1), (1, 2), (3, 4)]
    assert spans_to_cuts(spans, 7) == {1, 3}


def test_parse_binaryinferno_spec():
    output = """noise
SPECSTART
FieldFixed 1V (unknown)
Length 2V_BE (length)
FieldRep *V_BE (payload)
SPECEND
"""
    assert parse_spec(output) == [1, 2, None]

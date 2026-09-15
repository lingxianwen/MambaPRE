from __future__ import annotations

import argparse
import sys
import types
from collections import OrderedDict

from baselines.common import load_rows, score_predictions


def segment_unique(raws: list[bytes], sigma: float, refine: bool) -> dict[bytes, list[tuple[int, int]]]:
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage

    class MemoryLoader:
        """NEMESYS BaseLoader subset for in-memory RawMessage evaluation."""

        def __init__(self, messages):
            self.messagePool = OrderedDict((message, message) for message in messages)

    # NEMESYS' loader module imports PCAP-only dependencies (pcapy/scapy) at
    # module import time. The segmentation functions need only messagePool for
    # this in-memory benchmark, so expose that exact BaseLoader interface.
    loader_module = types.ModuleType("nemere.utils.loader")
    loader_module.BaseLoader = MemoryLoader
    sys.modules.setdefault("nemere.utils.loader", loader_module)
    ipython_module = types.ModuleType("IPython")

    def _noninteractive_embed() -> None:
        raise RuntimeError("NEMESYS requested an interactive debug shell")

    ipython_module.embed = _noninteractive_embed
    sys.modules.setdefault("IPython", ipython_module)
    from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements

    unique = list(dict.fromkeys(raws))
    segments = bcDeltaGaussMessageSegmentation(MemoryLoader([RawMessage(raw) for raw in unique]), sigma)
    if refine:
        segments = refinements(segments)
    output: dict[bytes, list[tuple[int, int]]] = {}
    for segment_list in segments:
        if segment_list:
            raw = bytes(segment_list[0].message.data)
            output[raw] = [(segment.offset, segment.length) for segment in segment_list if segment.length > 0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supplied NEMESYS source")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dir", required=True, help="directory containing the nemere package")
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="apply the later NEMESYS refinement pipeline; off for the WOOT'18 baseline",
    )
    args = parser.parse_args()
    sys.path.insert(0, args.source_dir)

    rows = load_rows(args.data)
    span_map = segment_unique([row["raw"] for row in rows], args.sigma, args.refine)
    predictions = [span_map.get(row["raw"], [(0, len(row["raw"]))]) for row in rows]
    score_predictions(
        rows,
        predictions,
        args.output,
        f"NEMESYS BCDG (sigma={args.sigma})",
        args.source_dir,
        {"sigma": args.sigma, "refine": args.refine},
    )


if __name__ == "__main__":
    main()

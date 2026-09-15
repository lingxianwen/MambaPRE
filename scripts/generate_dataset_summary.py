from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("Core train", "data/processed/core_corpus/train.jsonl", "original training files", "full byte-level fields", "training partition"),
    ("Core validation", "data/processed/core_corpus/validation.jsonl", "original training files", "full byte-level fields", "validation partition"),
    ("ID novel", "data/processed/core_corpus/test_novel.jsonl", "original test file", "full byte-level fields", "zero exact-byte overlap with train"),
    ("Source-held-out", "data/processed/core_corpus/ood_novel.jsonl", "separately provided evaluation file", "full byte-level fields", "file-held-out; zero exact-byte overlap"),
    ("External full", "data/processed/neupre_pdml/test_external_full_novel.jsonl", "NeuPRE public captures", "full-field PDML", "external captures; exact-byte novel"),
    ("Long calibration", "data/processed/neupre_pdml/calibration_long_envelope_novel.jsonl", "S7comm+ public capture sessions", "TPKT/COTP envelope only", "four calibration sessions"),
    ("Long test", "data/processed/neupre_pdml/test_long_envelope_session_disjoint_novel.jsonl", "S7comm+ public capture sessions", "TPKT/COTP envelope only", "six test sessions disjoint from calibration"),
    ("FINS calibration", "data/processed/long_full_strict/calibration_long_full_capture_disjoint_novel.jsonl", "controlled FINS groups", "63-field full-byte tiling", "three logical groups; threshold only"),
    ("FINS test", "data/processed/long_full_strict/test_long_full_capture_disjoint_novel.jsonl", "controlled variants + one public message", "63-field full-byte tiling", "seven groups + public message; zero exact-byte overlap"),
    ("Controlled LR train", "data/processed/controlled_long_range/train.jsonl", "synthetic diagnostic grammar", "dependency boundary only", "generated train split"),
    ("Controlled LR validation", "data/processed/controlled_long_range/validation.jsonl", "synthetic diagnostic grammar", "dependency boundary only", "generated validation split"),
    ("Controlled LR test", "data/processed/controlled_long_range/test.jsonl", "synthetic diagnostic grammar", "dependency boundary only", "zero exact-byte overlap"),
    ("Candidate LR train", "data/processed/controlled_candidate_selection/train.jsonl", "paired-candidate synthetic grammar", "one selector-controlled boundary", "generated train split"),
    ("Candidate LR validation", "data/processed/controlled_candidate_selection/validation.jsonl", "paired-candidate synthetic grammar", "one selector-controlled boundary", "generated validation split"),
    ("Candidate LR test", "data/processed/controlled_candidate_selection/test.jsonl", "paired-candidate synthetic grammar", "one selector-controlled boundary", "zero exact-byte overlap; primary local windows identical"),
)

PAPER_SPLITS = (
    "Core train",
    "Core validation",
    "ID novel",
    "Source-held-out",
    "External full",
    "Long test",
    "FINS test",
)

DISPLAY_NAMES = {
    "Core train": "Train",
    "Core validation": "Validation",
    "Long test": "Long envelope",
    "FINS test": "Strict FINS",
}

PROTOCOL_NAMES = {
    "cip_pccc": "CIP/PCCC",
    "dnp3": "DNP3",
    "iec104": "IEC104",
    "lon": "Lon",
    "modbus": "Modbus",
    "modbus_delta": "Modbus-Delta",
    "omron_fins": "FINS",
    "s7comm": "S7comm",
    "s7comm_plus": "S7comm+",
}

SCOPE_NAMES = {
    "full byte-level fields": "Full fields",
    "full-field PDML": "Full fields (PDML)",
    "TPKT/COTP envelope only": "TPKT/COTP envelope",
    "63-field full-byte tiling": "63 full fields",
}

ISOLATION_NAMES = {
    "Core train": "Deduplicated train partition",
    "Core validation": "Grouped validation partition",
    "ID novel": "Zero train byte overlap",
    "Source-held-out": "Held-out source file; zero byte overlap",
    "External full": "External captures; exact-byte novel",
    "Long test": "Six sessions disjoint from calibration",
    "FINS test": "Seven groups + one public message; zero byte overlap",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def esc(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def compact_protocols(value: str) -> str:
    names = [item.rsplit(":", 1)[0] for item in value.split("; ")]
    return ", ".join(PROTOCOL_NAMES.get(name, name) for name in names)


def length_range(value: str) -> str:
    parts = value.split("/")
    return parts[0] if parts[0] == parts[-1] else f"{parts[0]}--{parts[-1]}"


def main() -> None:
    output_rows = []
    for split, relative_path, source, label_scope, isolation in SPECS:
        rows = read_rows(ROOT / relative_path)
        lengths = [len(row["bytes"]) for row in rows]
        fields = [len(row["boundaries"]) + 1 for row in rows]
        total_bytes = sum(lengths)
        role_bytes = sum(sum(int(value) >= 0 for value in row["role_ids"]) for row in rows)
        type_bytes = sum(sum(int(value) >= 0 for value in row["type_ids"]) for row in rows)
        protocols = Counter(row["protocol"] for row in rows)
        captures = {str(row["capture_id"]) for row in rows if row.get("capture_id")}
        output_rows.append({
            "split": split,
            "protocols": "; ".join(f"{name}:{count}" for name, count in sorted(protocols.items())),
            "source_capture": f"{source}; captures={len(captures)}" if captures else source,
            "messages": len(rows),
            "length_min_median_max": f"{min(lengths)}/{statistics.median(lengths):g}/{max(lengths)}",
            "fields_min_median_max": f"{min(fields)}/{statistics.median(fields):g}/{max(fields)}",
            "role_coverage": role_bytes / total_bytes,
            "type_coverage": type_bytes / total_bytes,
            "label_scope": label_scope,
            "isolation": isolation,
        })
    csv_path = ROOT / "results/dataset_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    tex_dir = ROOT / "results/revision"
    tex_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Automatically generated by scripts/generate_dataset_summary.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dataset composition and isolation. Length and field columns report min/median/max; R/T is role/type byte coverage.}",
        "\\label{tab:dataset_summary}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{p{1.55cm}p{2.8cm}p{2.55cm}rcccp{2.25cm}p{2.65cm}}",
        "\\toprule",
        "Split & Protocol(s) (messages) & Source/capture & $N$ & Length & Fields & R/T & Label scope & Isolation \\\\",
        "\\midrule",
    ]
    for row in output_rows:
        coverage = f"{pct(row['role_coverage'])}/{pct(row['type_coverage'])}"
        lines.append(" & ".join((
            esc(row["split"]), esc(row["protocols"]), esc(row["source_capture"]), str(row["messages"]),
            row["length_min_median_max"], row["fields_min_median_max"], coverage,
            esc(row["label_scope"]), esc(row["isolation"]),
        )) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    tex_path = tex_dir / "dataset_summary.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    selected = {row["split"]: row for row in output_rows}
    compact_lines = [
        "% Automatically generated by scripts/generate_dataset_summary.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dataset and split summary from processed JSONL files. Strict FINS reports 29 test messages; its 12 calibration variants are excluded.}",
        "\\label{tab:dataset_summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{p{1.65cm}p{4.0cm}rcp{2.75cm}p{5.25cm}}",
        "\\toprule",
        "Split & Protocol(s) & $N$ & Length & Label scope & Isolation \\\\",
        "\\midrule",
    ]
    for split in PAPER_SPLITS:
        row = selected[split]
        compact_lines.append(" & ".join((
            esc(DISPLAY_NAMES.get(split, split)),
            esc(compact_protocols(row["protocols"])),
            str(row["messages"]),
            length_range(row["length_min_median_max"]),
            esc(SCOPE_NAMES.get(row["label_scope"], row["label_scope"])),
            esc(ISOLATION_NAMES[split]),
        )) + " \\\\")
    compact_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    compact_tex_path = tex_dir / "dataset_summary_compact.tex"
    compact_tex_path.write_text("\n".join(compact_lines), encoding="utf-8")
    print(csv_path)
    print(tex_path)
    print(compact_tex_path)


if __name__ == "__main__":
    main()

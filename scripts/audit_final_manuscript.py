from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT.parent / "ICASSP2026_Paper_Templates"
TEX = PAPER_DIR / "Template.tex"
FIG1 = PAPER_DIR / "fig1_overview.tex"
FIG1_STANDALONE = PAPER_DIR / "Fig1_standalone.tex"
PDF = PAPER_DIR / "Template.pdf"
LOG = PAPER_DIR / "Template.log"
FIG3 = PAPER_DIR / "Fig3_qualitative.pdf"
QUALITATIVE_SELECTION = ROOT / "results" / "reproduction" / "qualitative_case_selection.json"
OUT = ROOT / "results" / "revision" / "final_manuscript_audit.json"


def run_text(*command: str) -> str:
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return completed.stdout


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    fig1_text = FIG1.read_text(encoding="utf-8")
    fig1_standalone_text = FIG1_STANDALONE.read_text(encoding="utf-8")
    aggregate = json.loads(
        (ROOT / "results/batch_invariant_revision/final_aggregate.json").read_text()
    )
    scaling = json.loads((ROOT / "results/scaling/comparison.json").read_text())
    qualitative = json.loads(QUALITATIVE_SELECTION.read_text(encoding="utf-8"))
    long_counts = qualitative["long_envelope"]["counts"]
    fins_counts = qualitative["strict_fins"]["counts"]
    expected: dict[str, str] = {
        "final_title": "Mamba-PRE: Structure-Aware Dual-Stream State-Space Models for Protocol Field Segmentation",
        "strict_fins_exact_rounding": f"{aggregate['main']['strict_fins']['mambapre']['mean']:.4f}",
        "short_mamba": f"{aggregate['main']['id_novel']['mambapre']['mean']:.4f}",
        "short_transformer": f"{aggregate['main']['id_novel']['transformer']['mean']:.4f}",
        "long_mamba": f"{aggregate['main']['long_envelope']['mambapre']['mean']:.4f}",
        "long_transformer": f"{aggregate['main']['long_envelope']['transformer']['mean']:.4f}",
        "long_paired_delta": f"{aggregate['main']['long_envelope']['paired_mambapre_minus_transformer']['mean']:.4f}",
        "fins_transformer": f"{aggregate['main']['strict_fins']['transformer']['mean']:.4f}",
        "source_paired_delta": f"{aggregate['main']['source_held_out']['paired_mambapre_minus_transformer']['mean']:.4f}",
        "external_paired_delta": f"{aggregate['main']['external_full']['paired_mambapre_minus_transformer']['mean']:.4f}",
        "fins_paired_delta": f"{aggregate['main']['strict_fins']['paired_mambapre_minus_transformer']['mean']:.4f}",
        "factor_A_long": f"{aggregate['ablations']['bimamba_only']['long_envelope']['mean']:.3f}",
        "factor_B_long": f"{aggregate['ablations']['fixed_dual_stream']['long_envelope']['mean']:.3f}",
        "factor_CE_long": f"{aggregate['ablations']['unguided_learned_fusion']['long_envelope']['mean']:.3f}",
        "factor_F_long": f"{aggregate['ablations']['full_mambapre']['long_envelope']['mean']:.3f}",
        "full_minus_bimamba_long": f"{-aggregate['ablation_paired']['bimamba_only_minus_full']['long_envelope']['mean']:.4f}",
        "full_minus_unguided_long": f"{-aggregate['ablation_paired']['unguided_learned_fusion_minus_full']['long_envelope']['mean']:.4f}",
        "fixed_minus_full_long": f"{aggregate['ablation_paired']['fixed_dual_stream_minus_full']['long_envelope']['mean']:.4f}",
        "external_macro_mamba": f"{aggregate['external_protocol_macro']['mambapre']['mean']:.4f}",
        "external_macro_transformer": f"{aggregate['external_protocol_macro']['transformer']['mean']:.4f}",
        "external_role_accuracy": f"{aggregate['auxiliary_heads']['external_full']['byte_role_accuracy']['mean']:.4f}",
        "external_role_accuracy_std": f"{aggregate['auxiliary_heads']['external_full']['byte_role_accuracy']['sample_std']:.4f}",
        "external_type_accuracy": f"{aggregate['auxiliary_heads']['external_full']['byte_type_accuracy']['mean']:.4f}",
        "external_type_accuracy_std": f"{aggregate['auxiliary_heads']['external_full']['byte_type_accuracy']['sample_std']:.4f}",
        "qualitative_long_counts": (
            "Mamba-PRE/Transformer recover "
            f"{long_counts['mambapre_tp']}/{long_counts['gold']} and "
            f"{long_counts['transformer_tp']}/{long_counts['gold']} long-envelope cuts with "
            f"{long_counts['mambapre_fp']} and {long_counts['transformer_fp']} false positives"
        ),
        "qualitative_fins_counts": (
            f"They miss {fins_counts['mambapre_fn']}/{fins_counts['gold']} and "
            f"{fins_counts['transformer_fn']}/{fins_counts['gold']} strict-FINS cuts; "
            f"the Transformer adds {fins_counts['transformer_fp']} false positives"
        ),
    }
    for variant in ("cnn", "bimamba_only", "fixed_dual_stream", "unguided_learned_fusion", "full_mambapre"):
        for dataset in ("external_full", "strict_fins"):
            expected[f"{variant}_{dataset}_std"] = (
                f"{aggregate['ablations'][variant][dataset]['sample_std']:.3f}"
            )
    for block, summary in aggregate["gate"]["guided"].items():
        expected[f"gate_separation_{block}"] = f"{float(summary['mean']):.3f}"
    for block, summary in aggregate["gate"]["unguided"].items():
        expected[f"unguided_gate_separation_{block}"] = f"{float(summary['mean']):.3f}"
    fp32 = next(row for row in scaling["precisions"]["float32"]["rows"] if row["length"] == 4096)
    bf16 = next(row for row in scaling["precisions"]["bfloat16"]["rows"] if row["length"] == 4096)
    expected.update({
        "fp32_sdpa_speedup": f"{fp32['mamba_speedup_vs_sdpa']:.2f}",
        "fp32_sdpa_memory_penalty_percent": f"{-100.0 * fp32['mamba_memory_reduction_vs_sdpa']:.1f}",
        "bf16_flash_transformer_speedup": f"{1.0 / bf16['mamba_speedup_vs_sdpa']:.2f}",
        "bf16_flash_memory_advantage_percent": f"{-100.0 * bf16['mamba_memory_reduction_vs_sdpa']:.1f}",
    })

    forbidden = (
        "0.1282",
        "identical losses",
        "Source OOD",
        "capture-disjoint",
        "independent audit",
        "Fig3_candidate_selection",
        "tab:systems",
        "synthetic distant-evidence",
        "controlled candidate selection",
        "requires both local-pattern and long-range evidence",
        "Role-guided fusion",
        "Guided dual-stream block",
        "shared task supervision",
        "Strict full-field stress tests",
        "randomly cropped",
        "Icons by Freepik",
        "Flaticon",
        "Byte-wise gate",
        "limited transfer to unseen internal structure",
        "supporting the complementary local stream",
        "cannot be reproduced fairly",
        "padded batch length",
        "exact-field F1",
        "Message perfection",
        "Field-level ``perfection''",
        "0.7069",
        "role-derived guidance improves",
        "role supervision also helps",
    )
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    abstract = abstract_match.group(1) if abstract_match else ""
    abstract_forbidden = ("CNN", "synthetic", "fixed fusion", "Flash", "0.1281", "0.3262")
    main_table_match = re.search(
        r"\\begin\{table\*\}\[t\].*?\\label\{tab:five_seed\}(.*?)\\end\{table\*\}",
        text,
        re.DOTALL,
    )
    main_table = main_table_match.group(1) if main_table_match else ""
    leading_zero_findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if "includegraphics[width=." in line:
            continue
        if re.search(r"(?<![0-9])\.[0-9]", line):
            leading_zero_findings.append({"line": line_number, "text": line.strip()})

    pdf_info = run_text("pdfinfo", str(PDF))
    pages_match = re.search(r"^Pages:\s+(\d+)$", pdf_info, re.MULTILINE)
    pages = int(pages_match.group(1)) if pages_match else None
    page5 = run_text("pdftotext", "-f", "5", "-l", "5", "-layout", str(PDF), "-")
    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    report = {
        "expected_values": {key: {"value": value, "present": value in text} for key, value in expected.items()},
        "forbidden_phrases": {phrase: phrase in text for phrase in forbidden},
        "abstract_forbidden": {phrase: phrase in abstract for phrase in abstract_forbidden},
        "structure_checks": {
            "main_table_found": bool(main_table),
            "dataset_table_present": r"\input{dataset_summary_compact.tex}" in text,
            "cnn_absent_from_main_table": "CNN" not in main_table,
            "fins_visible": "Strict FINS" in text,
            "fig1_native_vectors_only": r"\includegraphics" not in fig1_text,
            "fig1_gate_name_matches_method": "Channel-wise byte gate" in fig1_text,
            "fig1_export_has_no_third_party_attribution": (
                "Flaticon" not in fig1_standalone_text
                and "Freepik" not in fig1_standalone_text
            ),
            "gate_diagnostic_framed_as_behavioral": (
                "behavioral validation rather than evidence that learned routing or guidance is necessary" in text
            ),
            "position_feature_defined": all(
                phrase in text
                for phrase in (
                    r"p_i^{r}=i/\max(L-1,1)",
                    r"p_i^{a}=\log(1+i)/\log(1+L_{\max})",
                    r"L_{\max}=4096",
                    "neither feature depends on the lengths of other messages in the batch",
                    "zero-based",
                )
            ),
            "fusion_contribution_is_investigative": (
                "We investigate structure-aware semantic supervision for learned fusion" in text
            ),
            "training_protocol_distribution_reported": all(
                phrase in text
                for phrase in (
                    "259 CIP/PCCC",
                    "304 DNP3",
                    "3,811 Modbus",
                    "616 FINS",
                    "1,007 S7comm",
                )
            ),
            "fins_abstract_scope_precise": (
                "limited transfer to densely dissected full-field layouts" in abstract
            ),
            "related_work_direct_comparability": (
                "not directly comparable under our byte-level labeling protocol" in text
            ),
            "classical_evidence_matches_citations": (
                "infer format from distributional, probabilistic, or semantic evidence" in text
            ),
            "dynamic_analysis_scope_bounded": (
                "ChatPRE combines program analysis, including dynamic taint tracking" in text
                and "These dynamic-analysis methods require a runnable implementation and coverage" in text
            ),
            "controlled_architecture_scope": "controlled architecture study" in text,
            "exact_byte_novel_defined": (
                "protocol label and complete extracted byte sequence" in text
                and "regardless of protocol label" in text
            ),
            "auxiliary_metrics_scoped": (
                "auxiliary diagnostics" in text
                and "not evidence of complete semantic recovery" in text
            ),
            "new_related_work_verified": all(
                key in text
                for key in (
                    "ren2026profield",
                    "huo2026chatpre",
                    "wei2026multiview",
                    "yang2025state",
                )
            ),
            "recent_work_preserved": all(
                key in text
                for key in ("zhao2026transre", "yang2026icpprag", "yang2026fieldweaver", "sheng2026pvparser")
            ),
            "qualitative_figure_present": (
                FIG3.exists()
                and r"\includegraphics[width=.94\columnwidth]{Fig3_qualitative.pdf}" in text
                and r"\label{fig:qualitative}" in text
            ),
            "qualitative_selection_is_deterministic": all(
                case["selection_rule"].startswith("Choose the seed minimizing L1 distance")
                for case in qualitative.values()
            ),
            "qualitative_sources_exist": all(
                Path(source).exists()
                for case in qualitative.values()
                for source in case["sources"].values()
            ),
        },
        "leading_zero_findings": leading_zero_findings,
        "pdf": {
            "pages": pages,
            "references_begin_on_page_5": "REFERENCES" in page5,
            "overfull_boxes": "Overfull \\hbox" in log_text or "Overfull \\vbox" in log_text,
            "undefined_references": bool(re.search(r"(?:Citation|Reference).+undefined", log_text)),
        },
    }
    report["all_pass"] = (
        all(item["present"] for item in report["expected_values"].values())
        and not any(report["forbidden_phrases"].values())
        and not any(report["abstract_forbidden"].values())
        and all(report["structure_checks"].values())
        and not leading_zero_findings
        and pages == 5
        and report["pdf"]["references_begin_on_page_5"]
        and not report["pdf"]["overfull_boxes"]
        and not report["pdf"]["undefined_references"]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

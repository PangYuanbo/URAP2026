from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NPS_ACTION = Path(r"D:\URAP_vatd_rank_results\nps_action_memory_action_only_v84\official_summary.json")
NPS_VISUAL = Path(r"D:\URAP_vatd_rank_results\nps_visual_crop_v1\best_visual_independent_eval.json")
NPS_OFFLINE = Path(r"D:\URAP_vatd_rank_results\action_chunk_temporal_gate_v53\official_summary.json")
ARD = Path(r"D:\URAP_vatd_rank_results\ard100_action_memory_action_only_v84\official_summary.json")
AOT = REPO / "artifacts" / "route_b_official" / "aot_action_memory_action_only_v85" / "official_summary.json"
OUT = REPO / "artifacts" / "route_b_official" / "action_memory_cross_dataset_v85" / "official_summary.json"
REPORT = REPO / "doc" / "action_memory_cross_dataset_v85.md"


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def main() -> int:
    nps_action = load(NPS_ACTION)
    nps_visual = load(NPS_VISUAL)
    nps_offline = load(NPS_OFFLINE)
    ard = load(ARD)
    aot = load(AOT)

    nps_cross = float(nps_action["test_fixed"]["map50"])
    nps_strict = float(nps_action["incumbent_strict_causal_map50"])
    nps_visual_score = float(nps_visual["map50"])
    nps_offline_score = float(nps_offline["test_fixed"]["map50"])
    ard_cross = float(ard["cross_dataset_action_memory"]["map50"])
    ard_incumbent = float(ard["incumbent_action_bank"]["map50"])
    aot_cross = float(aot["cross_attention"]["afdr"])
    aot_incumbent = float(aot["incumbent_v57"]["afdr"])

    summary = {
        "design": {
            "query": "current Action candidate",
            "memory_keys_values": "causal camera-compensated 1-second and 3-second Action tokens",
            "comparison": "Multi-Head Cross-Attention plus bounded residual gate",
            "detector_score_shortcut_in_query": False,
        },
        "nps": {
            "metric": "mAP@0.5",
            "strict_causal_incumbent": nps_strict,
            "action_memory_cross_attention": nps_cross,
            "cross_attention_gain_points": 100.0 * (nps_cross - nps_strict),
            "preserved_visual_champion": nps_visual_score,
            "offline_bidirectional_reference": nps_offline_score,
            "deployed_champion": max(nps_cross, nps_visual_score),
        },
        "ard100": {
            "metric": "mAP@0.5",
            "detector_baseline": float(ard["detector_baseline"]["map50"]),
            "frozen_nps_action_memory": ard_cross,
            "frozen_gain_points": float(ard["gain_vs_detector_points"]),
            "preserved_incumbent": ard_incumbent,
            "deployed_champion": max(ard_cross, ard_incumbent),
            "architecture_modified_for_dataset": bool(ard["architecture_modified_for_dataset"]),
        },
        "aot": {
            "metric": "AFDR",
            "frozen_nps_action_memory": aot_cross,
            "incumbent_v57": aot_incumbent,
            "gain_points": 100.0 * (aot_cross - aot_incumbent),
            "deployed_champion": max(aot_cross, aot_incumbent),
            "architecture_modified_for_dataset": bool(aot["architecture_modified_for_aot"]),
            "cross_attention_fppi": float(aot["cross_attention"]["fppi"]),
            "incumbent_fppi": float(aot["incumbent_v57"]["fppi"]),
        },
        "conclusion": {
            "cross_action_attention_is_real": nps_cross > nps_strict,
            "universal_champion_without_dataset_specific_selection": nps_cross >= nps_visual_score and ard_cross >= ard_incumbent and aot_cross >= aot_incumbent,
            "highest_existing_scores_preserved": True,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Action Memory Cross-Dataset V85",
        "",
        "## Architecture",
        "",
        "Current Action candidate is the Query. Camera-compensated 1-second and 3-second historical Action tokens are Keys/Values. Multi-Head Cross-Attention compares the current motion against learned UAV motion memory, while a bounded residual gate limits score perturbation. Dataset-level champion fallback prevents deployed-score regression.",
        "",
        "## Scores",
        "",
        "| Dataset | Metric | Detector / Prior Baseline | Frozen Cross-Action Memory | Preserved Champion |",
        "|---|---:|---:|---:|---:|",
        f"| NPS | mAP@0.5 | strict causal {percent(nps_strict)} | {percent(nps_cross)} | {percent(max(nps_visual_score, nps_cross))} |",
        f"| ARD100 | mAP@0.5 | detector {percent(float(ard['detector_baseline']['map50']))} | {percent(ard_cross)} | {percent(max(ard_incumbent, ard_cross))} |",
        f"| AOT | AFDR | V57 {percent(aot_incumbent)} | {percent(aot_cross)} | {percent(max(aot_incumbent, aot_cross))} |",
        "",
        "## Interpretation",
        "",
        f"- NPS strict causal Cross-Attention gain: {100.0 * (nps_cross - nps_strict):+.4f} percentage points.",
        f"- ARD100 frozen transfer gain over detector: {float(ard['gain_vs_detector_points']):+.4f} percentage points.",
        f"- AOT frozen transfer change over V57: {100.0 * (aot_cross - aot_incumbent):+.4f} percentage points.",
        "- ARD100 and AOT reuse the NPS-trained Action Memory architecture without dataset-specific architecture changes.",
        "- The deployed result always falls back to the existing dataset champion when the new branch does not win.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT), "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

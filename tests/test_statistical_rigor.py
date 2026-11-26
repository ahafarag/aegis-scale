"""Integration tests for Phase 3 audited multi-dataset traces, statistical rigor, and K8s 2D benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.statistical_rigor import (
    ABLATION_NAMES,
    DATA_DIR,
    DATASET_SPECS,
    STRATEGY_NAMES,
    load_empirical_trace,
    run_statistical_rigor_suite,
)


@pytest.mark.parametrize(
    "filename",
    [
        "real_aws_elb_cloudwatch.csv",
        "real_aws_ec2_network_in.csv",
        "real_aws_rds_cpu.csv",
        "real_flash_crowd_twitter.csv",
        "real_azure_inference_2024.csv",
    ],
)
def test_real_dataset_files_exist_and_exceed_1000_rows(filename: str) -> None:
    """Verify all five real production CSV traces exist on disk and contain > 1,000 rows."""
    csv_path = DATA_DIR / filename
    assert csv_path.is_file(), f"Missing empirical dataset CSV: {csv_path}"
    timestamps, scaled_rps, total_rows = load_empirical_trace(
        csv_path=csv_path,
        scale_to_mean_rps=3200.0,
        max_steps=600,
        start_offset=0,
        causal_warmup_steps=60,
    )
    assert total_rows > 1000, f"Expected > 1000 rows in {filename}, got {total_rows}"
    assert len(timestamps) == 600
    assert len(scaled_rps) == 600
    # Strictly causal prefix normalization: mean of first 60 steps == 3200.0
    assert abs(float(scaled_rps[:60].mean()) - 3200.0) < 1e-6


def test_statistical_rigor_fast_mode_end_to_end(tmp_path: Path) -> None:
    """Execute Phase 3 statistical_rigor in --fast mode and verify all 5 experiments."""
    out_json = tmp_path / "phase3_audited_results.json"
    fig_dir = tmp_path / "figures"

    results = run_statistical_rigor_suite(
        fast=True,
        max_steps=80,
        n_seeds=10,
        output_json=out_json,
        figures_dir=fig_dir,
    )

    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["suite_metadata"]["mode"] == "fast"
    assert loaded["suite_metadata"]["max_steps_per_dataset"] == 80
    assert loaded["suite_metadata"]["disjoint_rolling_windows"] == 10

    # Experiment A: all 5 datasets and 6 strategies present (including Savings Plan)
    exp_a = loaded["experiment_a_multi_dataset"]
    for spec in DATASET_SPECS:
        assert spec.name in exp_a
        assert exp_a[spec.name]["metadata"]["total_csv_rows"] > 1000
        for s_name in STRATEGY_NAMES:
            assert s_name in exp_a[spec.name]["strategies"]

    # Experiment B: Disjoint windows + Student-t CIs + Wilcoxon p-values
    exp_b = loaded["experiment_b_disjoint_wilcoxon"]
    assert exp_b["n_disjoint_windows"] == 10
    assert exp_b["student_t_critical_95"] > 2.0
    wilc = exp_b["wilcoxon_two_sided_tests"]
    assert wilc["p_value_cost_vs_isosla_od"] < 0.05
    assert wilc["p_value_cost_vs_savings_plan"] < 0.05

    # Experiment C: 6-way unconfounded ablations
    exp_c = loaded["experiment_c_ablations"]
    for ds_label in ("AWS-ELB-Web", "Viral-Flash-Traffic"):
        assert ds_label in exp_c
        for abl_name in ABLATION_NAMES:
            assert abl_name in exp_c[ds_label]

    # Experiment D: AZ correlation, true 60s cold-start, Spot multiplier sensitivity, and HiGHS scalability
    exp_d = loaded["experiment_d_sensitivity_and_scalability"]
    assert len(exp_d["az_correlation_sweep"]) == 4
    assert len(exp_d["cold_start_delay_sweep"]) == 4
    assert exp_d["cold_start_delay_sweep"][1]["step_resolution_seconds"] == 60
    assert len(exp_d["spot_price_multiplier_sensitivity"]["AWS-ELB-Web"]) == 4
    assert len(exp_d["spot_price_multiplier_sensitivity"]["Azure-LLM-Inference-2024"]) == 4
    assert len(exp_d["interruption_rate_sensitivity"]) == 4
    assert len(exp_d["capacity_mu_sensitivity"]) == 4
    assert [r["pool_count_k"] for r in exp_d["highs_scalability_sweep"]] == [21, 42, 84, 147, 210]

    # Experiment E: 2D Kubernetes/Karpenter across all 3 AWS regions
    exp_e = loaded["experiment_e_k8s_2d_multiregion"]
    for region in ("us-east-1", "eu-west-1", "ap-northeast-1"):
        assert region in exp_e
        assert exp_e[region]["karpenter_manifests_emitted"] >= 3
        aegis_2d = exp_e[region]["strategies"]["AegisScale_2D_Karpenter"]
        naive_1d = exp_e[region]["strategies"]["1D_CPU_Only_Mixed"]
        assert aegis_2d["mean_stranded_vcpu_pct"] < naive_1d["mean_stranded_vcpu_pct"]

    # Verify all 3 publication figures exist and are non-empty PNGs
    for fname in (
        "multi_dataset_generalization.png",
        "ablation_and_coldstart.png",
        "solver_scalability.png",
    ):
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing generated figure: {fpath}"
        raw = fpath.read_bytes()
        assert len(raw) > 8192
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert results["suite_metadata"]["generated_figures"]

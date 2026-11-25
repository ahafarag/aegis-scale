"""Integration tests for multi-dataset empirical traces and statistical rigor suite."""

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
    ],
)
def test_real_nab_dataset_files_exist_and_exceed_1000_rows(filename: str) -> None:
    """Verify all four real NAB CSV traces exist on disk and contain > 1,000 rows."""
    csv_path = DATA_DIR / filename
    assert csv_path.is_file(), f"Missing empirical dataset CSV: {csv_path}"
    timestamps, scaled_rps, total_rows = load_empirical_trace(
        csv_path=csv_path,
        scale_to_mean_rps=3200.0,
        max_steps=600,
        start_offset=0,
    )
    assert total_rows > 1000, f"Expected > 1000 rows in {filename}, got {total_rows}"
    assert len(timestamps) == 600
    assert len(scaled_rps) == 600
    assert abs(float(scaled_rps.mean()) - 3200.0) < 1e-6


def test_statistical_rigor_fast_mode_end_to_end(tmp_path: Path) -> None:
    """Execute statistical_rigor in --fast mode (max_steps=80, n_seeds=6) and verify outputs."""
    out_json = tmp_path / "phase2_statistical_results.json"
    fig_dir = tmp_path / "figures"

    results = run_statistical_rigor_suite(
        fast=True,
        max_steps=80,
        n_seeds=6,
        output_json=out_json,
        figures_dir=fig_dir,
    )

    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["suite_metadata"]["mode"] == "fast"
    assert loaded["suite_metadata"]["max_steps_per_dataset"] == 80
    assert loaded["suite_metadata"]["monte_carlo_seeds"] == 6

    # Experiment A: all 4 datasets and 5 strategies present
    exp_a = loaded["experiment_a_multi_dataset"]
    for spec in DATASET_SPECS:
        assert spec.name in exp_a
        assert exp_a[spec.name]["metadata"]["total_csv_rows"] > 1000
        for s_name in STRATEGY_NAMES:
            assert s_name in exp_a[spec.name]["strategies"]

    # Experiment B: Wilcoxon p-values strictly < 0.05
    exp_b = loaded["experiment_b_monte_carlo_wilcoxon"]
    assert exp_b["n_seeds"] == 6
    wilc = exp_b["wilcoxon_two_sided_tests"]
    assert wilc["p_value_cost_vs_isosla_od"] < 0.05
    assert wilc["p_value_continuity_vs_greedy"] < 0.05
    assert wilc["p_value_continuity_vs_reactive"] < 0.05
    assert wilc["p_value_storm_vs_static"] < 0.05
    assert wilc["all_significant_at_alpha_0_05"] is True

    # Experiment C: 5-way ablations across AWS-ELB-Web and Viral-Flash-Traffic
    exp_c = loaded["experiment_c_ablations"]
    for ds_label in ("AWS-ELB-Web", "Viral-Flash-Traffic"):
        assert ds_label in exp_c
        for abl_name in ABLATION_NAMES:
            assert abl_name in exp_c[ds_label]
        assert (
            exp_c[ds_label]["Full_AegisScale"]["total_sla_breaches"]
            <= exp_c[ds_label]["Ablation_No_ACI"]["total_sla_breaches"]
        )

    # Experiment D: AZ correlation, cold-start delay, and HiGHS scalability sweeps
    exp_d = loaded["experiment_d_sensitivity_and_scalability"]
    assert len(exp_d["az_correlation_sweep"]) == 4
    assert len(exp_d["cold_start_delay_sweep"]) == 4
    assert [r["pool_count_k"] for r in exp_d["highs_scalability_sweep"]] == [21, 42, 84, 147, 210]
    for item in exp_d["highs_scalability_sweep"]:
        assert item["median_ms"] < 1000.0
        if item["pool_count_k"] <= 147:
            assert item["p99_ms"] < 1000.0

    # Verify all 3 publication figures exist and are non-empty PNGs
    expected_figs = [
        "multi_dataset_generalization.png",
        "ablation_and_coldstart.png",
        "solver_scalability.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing generated figure: {fpath}"
        raw = fpath.read_bytes()
        assert len(raw) > 8192
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert results["suite_metadata"]["generated_figures"]

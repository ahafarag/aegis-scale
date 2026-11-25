# AegisScale: Distributionally-Robust Cloud & Kubernetes Karpenter Capacity Optimizer

[![CI](https://github.com/ahafarag/aegis-scale/actions/workflows/ci.yml/badge.svg)](https://github.com/ahafarag/aegis-scale/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Karpenter](https://img.shields.io/badge/EKS_Karpenter-v1_Ready-326CE5.svg)](src/aegis_scale/k8s/karpenter.py)
[![IaC](https://img.shields.io/badge/AWS_CDK-v2_Verified-FF9900.svg)](infra/stack.py)
[![Paper](https://img.shields.io/badge/IEEE_Transactions_Preprint-PDF-b31b1b.svg)](paper/main.pdf)

**AegisScale** is a mathematical control plane for **Amazon EC2** and **Amazon EKS / Kubernetes Karpenter (`karpenter.sh/v1`)** clusters that unifies **Adaptive Conformal Inference**, **heavy-traffic $G/G/c$ queueing theory**, **2D container bin-packing**, and **Rockafellar-Uryasev $\text{CVaR}_\beta$ Mixed-Integer Linear Programming** to cut cloud compute bills by **34%–39% vs. equal-SLA On-Demand fleets ($p = 1.86 \times 10^{-9}$)** while sustaining **>99.1% P99 tail-latency SLA continuity** across flash traffic surges and correlated Spot recalls.

---

## Core Architecture & Mathematical Guarantees

1. **Layer 1 — Scale-Invariant RLS + Adaptive Conformal Inference (Theorem 1):**
   Online Recursive Least Squares over normalized deltas and diurnal harmonics wrapped in Gibbs & Candes (2021) ACI guarantees distribution-free finite-sample demand coverage $|T^{-1}\sum_{t=1}^T \text{err}_t - \alpha_{\text{target}}| \le \frac{1 + 2\gamma}{\gamma T}$ without offline training or stationarity assumptions.
2. **Layer 2 — $M/M/1$-Exact $G/G/c$ Tail Sizing & 2D Pod Packing (Lemma 1):**
   Translates conformal arrival envelopes and node cold-start ramp buffers ($\hat{\lambda}_{\text{lead}} = \hat{U}_t + \dot{\lambda}_t\tau_{\text{boot}}$) into exact minimum safe vCPUs or two-dimensional `(vCPU, GiB)` Kubernetes pod replicas accounting for DaemonSet overhead across compute (`c7g`), balanced (`m7g`), and memory-optimized (`r7g`) families.
3. **Layer 3 — Auxiliary-Free Correlated Spot $\text{CVaR}_\beta$ MILP (Lemma 2):**
   A three-factor Gaussian copula with antithetic variates models correlated `(AZ, instance-family)` Spot recalls across `us-east-1`, `eu-west-1`, and `ap-northeast-1`. Constraining $\eta \ge 0$ eliminates $S$ auxiliary shortfall variables, allowing **HiGHS** to solve globally in **42–63 ms** (`328 ms` at $K = 210$ pools) and synthesize declarative `karpenter.sh/v1` `NodePool`, `EC2NodeClass`, and proactive `autoscaling/v1` `Scale` manifests.

---

## Multi-Dataset Empirical Generalization (Zero Hyperparameter Retuning)

Evaluated with a **single locked policy** ($\alpha_{\text{target}} = 0.015$, $\tau_{\text{SLA}} = 25\,\text{ms}$, $\beta = 0.95$, $\rho_{\text{floor}} = 0.25$) across **four distinct real-world production traces** from the Numenta Anomaly Benchmark (`benchmarks/statistical_rigor.py`):

| Real Production Dataset | Peak/Mean | Reactive HPA (55%) Continuity | Iso-SLA 100% OD Spend | Greedy Spot Continuity | **AegisScale Spend (Savings vs. Iso-OD)** | **AegisScale Continuity** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. AWS-ELB-Web** *(HTTP Load Balancer)* | `4.77x` *(10.6x 14d)* | 67.00% *(Fails)* | \$36,375 / yr | 90.67% | **\$23,015 / yr (36.73%)** | **99.17%** *(99.53% 14d)* |
| **2. AWS-EC2-Network** *(Ingress Bursts)* | `5.33x` | 82.17% *(Fails)* | \$51,599 / yr | 97.50% | **\$32,458 / yr (37.10%)** | **99.50%** *(0 storm br.)* |
| **3. AWS-RDS-Compute** *(DB Step-Shift)* | `2.37x` | 99.83% | \$13,487 / yr | 88.17% | **\$8,842 / yr (34.45%)** | **99.17%** |
| **4. Viral-Flash-Traffic** *(Twitter AAPL)* | **`22.65x`** | 94.33% *(Fails)* | \$51,612 / yr | 88.17% | **\$32,601 / yr (36.83%)** | **99.17%** |

![Multi-Dataset Generalization](docs/figures/multi_dataset_generalization.png)

---

## 30-Seed Statistical Rigor, Ablations & Cold-Start Resilience

### 1. 30-Seed Monte Carlo & Paired Wilcoxon Signed-Rank Tests (`AWS-ELB-Web`)

| Strategy ($N = 30$ Seeds) | Annualized Cost (`Mean ± 95% CI`) | Savings vs. Iso-OD | P99 Continuity (`Mean ± 95% CI`) | Spot-Storm Breaches |
| :--- | :---: | :---: | :---: | :---: |
| `Reactive HPA (55% CPU)` | \$15,373 ± \$5 | --- *(Fails SLA)* | 67.05% ± 0.26% | 16.50 ± 0.45 |
| `Iso-SLA 100% On-Demand` | \$37,088 ± \$501 | 0.00% ± 0.00% | 98.81% ± 0.09% | 2.83 ± 0.30 |
| `Greedy Cheapest Spot` | \$13,598 ± \$177 | 63.33% ± 0.03% | 91.31% ± 0.29% | 21.43 ± 0.86 |
| `Static 30/70 Mixed Spot` | \$19,523 ± \$261 | 47.36% ± 0.02% | 97.78% ± 0.12% | 7.03 ± 0.56 |
| **`AegisScale (Full)`** | **\$23,516 ± \$316** | **36.59% ± 0.03%** | **99.39% ± 0.07%** | **1.67 ± 0.34** |

* **Exact Two-Sided Paired Wilcoxon $p$-Values:** `p = 1.86e-9` across all four comparisons (strictly winning `30/30` independent seeds).

### 2. Architectural Ablation, Node Boot Delay (`0s–180s`), and 210-Pool Scalability

* **5-Way Ablation:** Removing ACI collapses ELB continuity to `75.50%` (`193` combined breaches vs. `10`); freezing $\gamma = 10^{-9}$ (`Static Conformal`) inflates Viral Flash cost by **+11.3%**; removing $G/G/c$ queueing pooling wastes **+15.0%–15.9%** spend; ignoring copula correlation increases SLA breaches by **+50%**.
* **Kubernetes Cold-Start Resilience:** Under a `120s` node boot delay, Reactive HPA drops to `39.50%` continuity (`2,829 ms` wait), while AegisScale sustains **97.33%** (`98.83%` at `60s`).

![Ablation and Cold-Start Resilience](docs/figures/ablation_and_coldstart.png)

![HiGHS MILP Scalability](docs/figures/solver_scalability.png)

---

## IEEE Transactions Manuscript & Documentation

* **Compiled IEEE Transactions Paper (PDF):** [paper/main.pdf](paper/main.pdf)
* **LaTeX Source & BibTeX:** [paper/main.tex](paper/main.tex) and [paper/references.bib](paper/references.bib)
* **Mathematical Whitepaper:** [docs/mathematical_whitepaper.md](docs/mathematical_whitepaper.md)
* **Serverless Custom Subdomain Deployment (`Hostinger + AWS CDK`):** [docs/deploy_hostinger_subdomain.md](docs/deploy_hostinger_subdomain.md)

---

## Repository Structure

```text
aegis-scale/
├── paper/                           # Full 2-column IEEE Transactions LaTeX manuscript & compiled PDF
├── docs/                            # Mathematical whitepaper, Hostinger deployment guide & PNG figures
├── src/aegis_scale/
│   ├── math_core/                   # Scale-invariant RLS+ACI, G/G/c bisection, 3-factor copula, CVaR MILP
│   ├── k8s/                         # 2D CPU+GiB pod bin-packing & Karpenter v1 NodePool/EC2NodeClass operator
│   ├── cloud/                       # Multi-region catalogs (us-east-1, eu-west-1, ap-northeast-1) & boto3 adapters
│   └── api/                         # Hardened FastAPI Control Plane + Executive & Karpenter Simulator UI
├── infra/
│   ├── stack.py                     # Production AWS CDK stack (VPC, LaunchTemplate, ASG, DDB, EventBridge)
│   └── demo_stack.py                # Scale-to-zero AWS Lambda Container + HTTP API custom domain stack
├── benchmarks/
│   ├── data/                        # 4 real empirical Numenta NAB AWS/Twitter production CSV traces
│   ├── run_benchmarks.py            # 7-day multi-storm synthetic stress benchmark
│   ├── real_trace_benchmark.py      # Full 14-day (4,032-step) real AWS CloudWatch ELB benchmark
│   └── statistical_rigor.py         # Multi-dataset, 30-seed Wilcoxon, 5-way ablation & scalability suite
└── tests/                           # 19 automated pytest tests (math theorems, K8s/Karpenter, AppSec, CDK, stats)
```

---

## Quickstart

```bash
# Option A: Docker
docker compose up --build

# Option B: Local Python & Full Verification Suite
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,infra]"

pytest -v
python benchmarks/statistical_rigor.py
uvicorn aegis_scale.api.app:app --port 8000
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

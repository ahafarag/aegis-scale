# Distributionally-Robust Cloud Capacity Scaling and Correlated Spot Portfolio Optimization via Adaptive Conformal Inference and CVaR MILP

**Author:** Abdelrahman H. Farag  
**System:** AegisScale (`v0.1.0`)

---

## Abstract

Cloud infrastructure operators face a fundamental tension between compute cost efficiency and service-level agreement (SLA) continuity. Provisioning exclusively On-Demand virtual machines guarantees availability at a $2.5\times$–$5\times$ price premium, whereas naive adoption of discounted Spot instances exposes workloads to correlated capacity recalls across Availability Zones (AZs) and hardware families, causing severe tail-latency ($\text{P99}$) violations during flash traffic surges. We present **AegisScale**, a three-layer mathematical control architecture that decouples distribution-free workload upper-bounding from risk-averse integer fleet allocation: (1) **Adaptive Conformal Inference (ACI)** wraps an online harmonic-autoregressive Recursive Least Squares (RLS) filter to guarantee finite-sample demand coverage under arbitrary non-stationary regime shifts without distributional assumptions; (2) a **Kingman–Sakasegawa $G/G/c$ heavy-traffic queueing law** maps conformal arrival-rate envelopes into exact minimum safe vCPU requirements for target $\text{P99}$ queueing delay bounds; and (3) a **multi-factor Gaussian copula** feeds a **Rockafellar–Uryasev Conditional Value-at-Risk ($\text{CVaR}_\beta$) Mixed-Integer Linear Program (MILP)** solved to global optimality in milliseconds via HiGHS.

---

## 1. Online Distribution-Free Demand Envelopes (Layer 1)

Let $\{D_t\}_{t=1}^T \subset \mathbb{R}_{\ge 0}$ denote incoming request arrival rates (requests/sec) observed over discrete control intervals. Cloud traffic violates classical i.i.d. and exchangeability conditions due to diurnal periodicity, deployment shifts, and heavy-tailed flash crowds.

### 1.1 Harmonic-AR Recursive Least Squares Base Forecaster
Define the feature vector $\phi_t \in \mathbb{R}^{p+5}$ combining $p$ autoregressive lags and two diurnal harmonics with period $P$:
$$\phi_t = \left[1, \; D_{t-p}, \dots, D_{t-1}, \; \sin\!\left(\frac{2\pi t}{P}\right), \cos\!\left(\frac{2\pi t}{P}\right), \sin\!\left(\frac{4\pi t}{P}\right), \cos\!\left(\frac{4\pi t}{P}\right)\right]^\top$$
With exponential forgetting factor $\lambda_{\text{rls}} \in (0, 1)$, parameter vector $\theta_t$ and inverse covariance matrix $P_t$ update in $O(d^2)$ time ($d = p + 5 = 9$):
$$k_t = \frac{P_{t-1}\phi_t}{\lambda_{\text{rls}} + \phi_t^\top P_{t-1}\phi_t}, \quad \theta_t = \theta_{t-1} + k_t\big(D_t - \theta_{t-1}^\top \phi_t\big), \quad P_t = \frac{1}{\lambda_{\text{rls}}}\big(P_{t-1} - k_t \phi_t^\top P_{t-1}\big)$$

### 1.2 Theorem 1 (Finite-Sample Non-Stationary Coverage via ACI)
Let $s_i = D_i - \hat{D}_{i|i-1}$ denote one-step signed residuals stored in a sliding calibration window $\mathcal{W}_t$ of length $W$. Following Gibbs & Candès (2021), define the empirical quantile $\hat{Q}_{1-\alpha}(\mathcal{W}_t)$ at nominal miscoverage $\alpha_t \in \mathbb{R}$, with boundary conventions $\hat{Q}_{1-\alpha} = +\infty$ for $\alpha \le 0$ and $\hat{Q}_{1-\alpha} = -\infty$ for $\alpha \ge 1$. Let $\hat{U}_t(\alpha_t) = \hat{D}_{t|t-1} + \hat{Q}_{1-\alpha_t}(\mathcal{W}_t)$, let $\text{err}_t = \mathbb{I}\{D_t > \hat{U}_t(\alpha_t)\}$, and update:
$$\alpha_{t+1} = \alpha_t + \gamma (\alpha_{\text{target}} - \text{err}_t), \qquad \gamma > 0, \; \alpha_1 = \alpha_{\text{target}} \in (0, 1)$$

**Theorem 1.** *For any deterministic or adversarial bounded sequence $\{D_t\}_{t=1}^T$, $\alpha_t \in [-\gamma, 1 + \gamma]$ for all $t \ge 1$, and the empirical under-provisioning rate satisfies:*
$$\left| \frac{1}{T}\sum_{t=1}^T \text{err}_t - \alpha_{\text{target}} \right| \le \frac{1 + 2\gamma}{\gamma T}$$

*Proof.* Whenever $\alpha_t \le 0$, $\hat{U}_t(\alpha_t) = +\infty \implies \text{err}_t = 0$, so $\alpha_{t+1} = \alpha_t + \gamma \alpha_{\text{target}} > \alpha_t$; since a single step decreases $\alpha$ by at most $\gamma(1 - \alpha_{\text{target}}) < \gamma$, induction yields $\alpha_t \ge -\gamma$. Symmetrically, whenever $\alpha_t \ge 1$, $\hat{U}_t(\alpha_t) = -\infty \implies \text{err}_t = 1$, so $\alpha_{t+1} = \alpha_t - \gamma(1 - \alpha_{\text{target}}) < \alpha_t$, yielding $\alpha_t \le 1 + \gamma$. Summing $\alpha_{t+1} - \alpha_t = \gamma(\alpha_{\text{target}} - \text{err}_t)$ from $t = 1$ to $T$ telescopes to $\alpha_{T+1} - \alpha_1 = \gamma \sum_{t=1}^T (\alpha_{\text{target}} - \text{err}_t)$. Dividing by $\gamma T$ and noting $|\alpha_{T+1} - \alpha_1| \le (1 + \gamma) - (-\gamma) = 1 + 2\gamma$ completes the proof. $\square$

---

## 2. Queueing-Theoretic Tail-Latency Mapping (Layer 2)

Given upper arrival rate $\hat{\lambda} = \hat{U}_t$, per-vCPU service rate $\mu > 0$, inter-arrival squared coefficient of variation $c_a^2$, and service-time squared coefficient of variation $c_s^2$, let $\rho(c) = \frac{\hat{\lambda}}{c\mu} < 1$ denote server utilization for $c > \hat{\lambda}/\mu$.

### 2.1 Heavy-Traffic Quantile Law and Exact $M/M/1$ Consistency
Combining Sakasegawa's delay probability $P_{\text{wait}}(c) = \rho(c)^{\sqrt{2(c+1)}-1}$ with Kingman's heavy-traffic mean wait $\mathbb{E}[W_q(c)] = \left(\frac{c_a^2 + c_s^2}{2}\right)\frac{\rho(c)^{\sqrt{2(c+1)}-1}}{c\mu(1-\rho(c))}$, the conditional mean wait given queueing is:
$$\mathbb{E}[W_q(c) \mid W_q > 0] = \frac{\mathbb{E}[W_q(c)]}{P_{\text{wait}}(c)} = \left(\frac{c_a^2 + c_s^2}{2}\right)\frac{1}{c\mu(1 - \rho(c))}$$
Under Kingman's exponential heavy-traffic tail theorem, for target quantile $q = 1 - \delta \in (0, 1)$ (e.g., $q = 0.99, \delta = 0.01$):
$$W_{q, 1-\delta}(c) = \begin{cases} 0, & \text{if } P_{\text{wait}}(c) \le \delta, \\[6pt] \left(\dfrac{c_a^2 + c_s^2}{2}\right)\dfrac{1}{c\mu(1 - \rho(c))} \ln\!\left(\dfrac{P_{\text{wait}}(c)}{\delta}\right), & \text{if } P_{\text{wait}}(c) > \delta. \end{cases}$$

**Lemma 1 (Exact $M/M/1$ Reduction & Global Uniqueness).**
1. *At $c = 1, c_a^2 = c_s^2 = 1$, $P_{\text{wait}}(1) = \rho$ and $W_{q, 1-\delta}(1) = \max\!\left(0, \frac{\ln(\rho/\delta)}{\mu(1-\rho)}\right)$, which is identical to the exact closed-form stationary waiting-time quantile of an $M/M/1$ queue.*
2. *For any $c_a^2 + c_s^2 > 0$, both $c\mu(1-\rho(c)) = c\mu - \hat{\lambda}$ and $1/P_{\text{wait}}(c)$ are strictly increasing on $(\hat{\lambda}/\mu, \infty)$. Hence $W_{q, 1-\delta}(c)$ is continuous and strictly decreasing wherever positive, guaranteeing a unique global root $C_{\text{req}}$ to $W_{q, 1-\delta}(c) \le \tau_{\text{SLA}}$ found to tolerance $\epsilon$ in $\lceil \log_2(\Delta c / \epsilon) \rceil$ bisection steps.*

---

## 3. Correlated Spot Copula & Exact CVaR MILP (Layer 3)

### 3.1 Three-Factor Gaussian Preemption Copula
For pool $k \in \mathcal{K} = \mathcal{I}_{\text{OD}} \cup \mathcal{I}_{\text{Spot}}$ in AZ $a(k)$ and hardware family $f(k)$ with marginal recall probability $p_k \in [0, 1)$, latent capacity stress is modeled as:
$$Z_k^{(s)} = \sqrt{\rho_{\text{az}}}\,G_{a(k)}^{(s)} + \sqrt{\rho_{\text{fam}}}\,H_{f(k)}^{(s)} + \sqrt{1 - \rho_{\text{az}} - \rho_{\text{fam}}}\,\epsilon_k^{(s)}, \qquad G_a^{(s)}, H_f^{(s)}, \epsilon_k^{(s)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$
Thresholding $\xi_k^{(s)} = \mathbb{I}\{Z_k^{(s)} < \Phi^{-1}(p_k)\}$ preserves exact marginals $\mathbb{P}(\xi_k = 1) = p_k$ ($\xi_k \equiv 0$ for On-Demand) while inducing latent correlation $\rho_{\text{az}} + \rho_{\text{fam}}$ within identical `(AZ, family)` pairs, $\rho_{\text{az}}$ across families in the same AZ, and $\rho_{\text{fam}}$ across AZs in the same family. Antithetic variates $(G, H, \epsilon) \mapsto (-G, -H, -\epsilon)$ reduce Monte Carlo variance across $S$ scenarios.

### 3.2 Exact Shortfall Linearization Without Auxiliary Surplus Variables
Let $x_k \in \mathbb{Z}_{\ge 0}$ denote instances allocated to pool $k$ with vCPU count $v_k$ and hourly cost $p_k^{\text{usd}}$. Surviving capacity in scenario $s$ is $C_{\text{surv}}^{(s)}(x) = \sum_{k \in \mathcal{K}} v_k (1 - \xi_k^{(s)}) x_k$, signed deficit is $d_s(x) = C_{\text{req}} - C_{\text{surv}}^{(s)}(x) \in \mathbb{R}$, and non-negative capacity shortfall is $L_s(x) = \max(0, d_s(x)) \ge 0$.

**Lemma 2 (Exact Non-Negative Shortfall Linearization).**
*Restricting the Rockafellar–Uryasev threshold variable to $\eta \ge 0$ satisfies:*
$$\min_{\eta \ge 0} \left\{ \eta + \frac{1}{(1-\beta)S}\sum_{s=1}^S \max\big(0, \; d_s(x) - \eta\big) \right\} = \text{CVaR}_\beta\big(L(x)\big)$$
*without introducing $S$ intermediate auxiliary variables for $L_s(x) = \max(0, d_s(x))$.*

*Proof.* Since $L_s(x) \ge 0$ for every scenario $s$, its $\beta$-quantile satisfies $\eta^* = \text{VaR}_\beta(L(x)) \ge 0$, so restricting $\eta \ge 0$ does not exclude the minimizer of Rockafellar & Uryasev (2000). Furthermore, for any $\eta \ge 0$ and any $d_s \in \mathbb{R}$: if $d_s \ge 0$, $\max(0, d_s - \eta) = \max(0, L_s - \eta)$; if $d_s < 0$, then $d_s - \eta < 0$ and $L_s - \eta = -\eta \le 0$, so both $\max(0, d_s - \eta)$ and $\max(0, L_s - \eta)$ equal $0$. $\square$

### 3.3 Computational Complexity
With $K \approx 21$ candidate pools and $S = 300\text{–}400$ scenarios, the resulting MILP contains only $K = 21$ integer variables and $K + 1 + S \approx 422$ continuous variables with sparse constraint matrix density $< 6\%$. Branch-and-cut in HiGHS solves each control cycle to global optimality in $< 15\text{ ms}$ median latency on commodity CPUs—four orders of magnitude faster than the 60-second cloud control interval.

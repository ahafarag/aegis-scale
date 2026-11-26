"""Adaptive Conformal Inference (ACI) for non-stationary cloud traffic streams.

Implements a scale-invariant online Recursive Least Squares (RLS) delta-harmonic
forecaster paired with Gibbs & Candes (2021) Adaptive Conformal Inference.
Updates dual quantile state alpha_t strictly against the actual provisioned
finite upper bound (zero 1e9 bookkeeping inflation and zero step-1 lookahead).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ConformalStepResult:
    """Result of one online predict-then-observe conformal step."""

    step: int
    point_forecast: float
    upper_bound: float
    observed: float | None
    alpha_t: float
    miscovered: bool | None
    empirical_miscoverage_rate: float


class OnlineHarmonicRLS:
    """Scale-invariant forgetting-factor RLS forecaster over normalized deltas and harmonics.

    Uses strictly lagged scale sigma_{t-1} when forming features phi_t and target_delta,
    updating sigma_t only after the RLS parameter update (matching paper Eqs. 1-2).
    """

    def __init__(
        self,
        n_lags: int = 4,
        period_steps: int = 288,
        forgetting_factor: float = 0.985,
        ridge_init: float = 1.0,
    ) -> None:
        self.n_lags = n_lags
        self.period_steps = max(2, period_steps)
        self.lam = forgetting_factor
        self.dim = 1 + n_lags + 4
        self.theta = np.zeros(self.dim, dtype=np.float64)
        self.cov = np.eye(self.dim, dtype=np.float64) * ridge_init
        self.history: deque[float] = deque(maxlen=n_lags + 1)
        self.running_scale = 1.0
        self.step_count = 0

    def _features(self, t: int, scale: float) -> np.ndarray:
        vals = list(self.history)
        denom_scale = max(1.0, scale)
        diffs = [(vals[i + 1] - vals[i]) / denom_scale for i in range(len(vals) - 1)]
        if len(diffs) < self.n_lags:
            diffs = [0.0] * (self.n_lags - len(diffs)) + diffs
        else:
            diffs = diffs[-self.n_lags :]
        omega = 2.0 * math.pi * float(t) / float(self.period_steps)
        feats = [
            1.0,
            *diffs,
            math.sin(omega),
            math.cos(omega),
            math.sin(2.0 * omega),
            math.cos(2.0 * omega),
        ]
        return np.asarray(feats, dtype=np.float64)

    def predict(self) -> float:
        if len(self.history) == 0:
            return 0.0
        last_y = float(self.history[-1])
        if len(self.history) < 2:
            return max(0.0, last_y)
        x = self._features(self.step_count, self.running_scale)
        norm_delta = float(np.clip(np.dot(self.theta, x), -0.75, 0.75))
        return max(0.0, last_y + norm_delta * self.running_scale)

    def update(self, y: float) -> None:
        y_val = max(0.0, float(y))
        prev_scale = max(1.0, self.running_scale)

        if len(self.history) >= 2:
            x = self._features(self.step_count, prev_scale)
            target_delta = (y_val - float(self.history[-1])) / prev_scale
            px = self.cov @ x
            denom = self.lam + float(np.dot(x, px))
            gain = px / denom
            err = target_delta - float(np.dot(self.theta, x))
            self.theta = self.theta + gain * err
            self.cov = (self.cov - np.outer(gain, px)) / self.lam

        # Update sigma_t strictly AFTER parameter update
        if len(self.history) == 0:
            self.running_scale = max(1.0, y_val)
        else:
            self.running_scale = 0.95 * self.running_scale + 0.05 * max(1.0, y_val)

        self.history.append(y_val)
        self.step_count += 1


class AdaptiveConformalPredictor:
    """Gibbs & Candes (2021) Adaptive Conformal Inference for upper demand bounds.

    Scores err_t strictly against the actual provisioned upper bound U_t(alpha_t).
    In infinite-ceiling theoretical mode (finite_cap=False), alpha_t <= 0 maps to
    +inf so Theorem 1 holds with exact bound |(1/T) sum err_t - alpha| <= (1+gamma)/(gamma*T).
    In practical cloud provisioning mode (default, finite_cap=True), alpha_t <= 0 maps
    to a finite conservative surge envelope and err_t is honestly evaluated against it.
    """

    def __init__(
        self,
        alpha_target: float = 0.02,
        gamma: float = 0.05,
        window_size: int = 120,
        period_steps: int = 288,
        finite_cap: bool = True,
        safety_envelope_ceiling: float = 1e9,
    ) -> None:
        if not (0.0 < alpha_target < 1.0):
            raise ValueError("alpha_target must lie strictly in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be strictly positive")
        if window_size < 5:
            raise ValueError("window_size must be >= 5")

        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.window_size = int(window_size)
        self.finite_cap = bool(finite_cap)
        self.safety_envelope_ceiling = float(safety_envelope_ceiling)

        self.alpha_t = float(alpha_target)
        self.forecaster = OnlineHarmonicRLS(period_steps=period_steps)
        self.residuals: deque[float] = deque(maxlen=self.window_size)

        self.total_steps = 0
        self.total_errors = 0
        self._pending_prediction: tuple[float, float] | None = None

    @staticmethod
    def theoretical_miscoverage_deviation_bound(num_steps: int, gamma: float) -> float:
        """Tight Theorem 1 bound |(1/T) sum err_t - alpha_target| <= (1 + gamma)/(gamma*T) for alpha_1 in (0,1)."""
        if num_steps <= 0:
            return 1.0
        return (1.0 + gamma) / (gamma * float(num_steps))

    def _conformal_quantile(self, alpha: float) -> float:
        if alpha <= 0.0:
            return self.safety_envelope_ceiling
        if alpha >= 1.0:
            return -self.safety_envelope_ceiling
        if len(self.residuals) == 0:
            return 0.15 * self.forecaster.running_scale if self.forecaster.step_count > 0 else 0.0

        arr = np.sort(np.asarray(self.residuals, dtype=np.float64))
        n = len(arr)
        q_level = math.ceil((n + 1) * (1.0 - alpha)) / float(n)
        if q_level >= 1.0:
            return float(arr[-1])
        q_level = max(0.0, min(1.0, q_level))
        return float(np.quantile(arr, q_level, method="higher"))

    def predict_upper_bound(self) -> tuple[float, float]:
        """Predict (point_forecast, provisioned_upper_bound) for the next interval."""
        point = self.forecaster.predict()
        q = self._conformal_quantile(self.alpha_t)
        if self.alpha_t <= 0.0:
            if not self.finite_cap:
                upper = self.safety_envelope_ceiling
            else:
                max_res = max(self.residuals) if self.residuals else 0.20 * point
                recent_max = max(self.forecaster.history) if self.forecaster.history else point
                upper = max(
                    point + max(0.0, max_res) * 1.35,
                    recent_max * 1.30,
                    point * 1.35 + 10.0,
                )
        elif self.alpha_t >= 1.0:
            upper = 0.0
        else:
            upper = max(0.0, point + q)

        self._pending_prediction = (point, upper)
        return point, upper

    def observe(self, observed_value: float) -> ConformalStepResult:
        """Observe true workload D_t, score err_t against actual upper_bound, and update alpha_{t+1}."""
        y = max(0.0, float(observed_value))
        if self._pending_prediction is None:
            self.predict_upper_bound()
        assert self._pending_prediction is not None
        point, upper = self._pending_prediction
        self._pending_prediction = None

        used_alpha = self.alpha_t
        is_warmup_first = self.forecaster.step_count == 0

        if is_warmup_first:
            # First observation initializes scale/history without polluting miscoverage statistics
            err_t = 0
        else:
            err_t = 1 if y > upper else 0
            self.total_steps += 1
            self.total_errors += err_t
            self.alpha_t = max(
                -self.gamma,
                min(1.0 + self.gamma, self.alpha_t + self.gamma * (self.alpha_target - float(err_t))),
            )
            self.residuals.append(y - point)

        self.forecaster.update(y)

        emp_rate = (
            float(self.total_errors) / float(self.total_steps)
            if self.total_steps > 0
            else 0.0
        )
        return ConformalStepResult(
            step=self.total_steps,
            point_forecast=point,
            upper_bound=upper,
            observed=y,
            alpha_t=used_alpha,
            miscovered=bool(err_t) if not is_warmup_first else None,
            empirical_miscoverage_rate=emp_rate,
        )

    def step(self, observed_value: float) -> ConformalStepResult:
        """Convenience method performing predict_upper_bound() followed by observe(observed_value)."""
        self.predict_upper_bound()
        return self.observe(observed_value)

    def state_dict(self) -> dict:
        """Serialize online predictor state for DynamoDB persistence."""
        return {
            "alpha_target": self.alpha_target,
            "gamma": self.gamma,
            "window_size": self.window_size,
            "finite_cap": self.finite_cap,
            "alpha_t": self.alpha_t,
            "total_steps": self.total_steps,
            "total_errors": self.total_errors,
            "residuals": list(self.residuals),
            "rls_step": self.forecaster.step_count,
            "rls_scale": self.forecaster.running_scale,
            "rls_history": list(self.forecaster.history),
            "rls_theta": self.forecaster.theta.tolist(),
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> "AdaptiveConformalPredictor":
        """Restore predictor state from serialized dictionary."""
        inst = cls(
            alpha_target=float(data["alpha_target"]),
            gamma=float(data["gamma"]),
            window_size=int(data["window_size"]),
            finite_cap=bool(data.get("finite_cap", True)),
        )
        inst.alpha_t = float(data["alpha_t"])
        inst.total_steps = int(data["total_steps"])
        inst.total_errors = int(data["total_errors"])
        for r in data.get("residuals", []):
            inst.residuals.append(float(r))
        inst.forecaster.step_count = int(data.get("rls_step", 0))
        inst.forecaster.running_scale = float(data.get("rls_scale", 1.0))
        for h in data.get("rls_history", []):
            inst.forecaster.history.append(float(h))
        if "rls_theta" in data and len(data["rls_theta"]) == inst.forecaster.dim:
            inst.forecaster.theta = np.asarray(data["rls_theta"], dtype=np.float64)
        return inst

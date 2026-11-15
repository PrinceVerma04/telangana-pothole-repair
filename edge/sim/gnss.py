"""GNSS error model for the simulated bus receiver.

``accuracy_m`` (the ``--gps-sigma`` option) has the meaning of Android ``Location.getAccuracy()``: the radius of
68 % horizontal confidence. For a circular Gaussian that radius is 1.515 sigma_axis, so the per-axis standard
deviation is ``accuracy_m / 1.515``. The error is mostly a slowly varying bias (multipath, atmosphere): a
first-order Gauss-Markov process per axis with correlation time ``tau_s`` carries ``corr_frac`` of the variance,
plus white noise for the rest. The fix also lags the true position by the receiver latency (the calibration
certificate's ``gnss_latency_s``); the server advances fixes by that latency (PLAN 5.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

R68_PER_SIGMA = 1.515  # 68 % radius of a 2-D circular Gaussian, in units of sigma per axis


@dataclass
class GnssNoise:
    accuracy_m: float = 7.0
    tau_s: float = 60.0
    corr_frac: float = 0.8  # share of the variance in the slow (correlated) part
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    bias: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.bias = self.rng.normal(0.0, self.sigma_corr, 2) if self.accuracy_m > 0 else np.zeros(2)

    @property
    def sigma_axis(self) -> float:
        return max(0.0, self.accuracy_m) / R68_PER_SIGMA

    @property
    def sigma_corr(self) -> float:
        return self.sigma_axis * math.sqrt(self.corr_frac)

    @property
    def sigma_white(self) -> float:
        return self.sigma_axis * math.sqrt(max(0.0, 1.0 - self.corr_frac))

    def step(self, dt: float) -> None:
        """Advance the correlated bias by ``dt`` seconds (exact AR(1) discretisation, stationary variance)."""
        if self.accuracy_m <= 0 or dt <= 0:
            return
        phi = math.exp(-dt / self.tau_s)
        self.bias = phi * self.bias + math.sqrt(1.0 - phi * phi) * self.sigma_corr * self.rng.normal(0.0, 1.0, 2)

    def sample(self) -> tuple[float, float]:
        """Current (east, north) error in metres: bias + fresh white noise."""
        if self.accuracy_m <= 0:
            return 0.0, 0.0
        w = self.rng.normal(0.0, self.sigma_white, 2)
        return float(self.bias[0] + w[0]), float(self.bias[1] + w[1])

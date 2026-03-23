"""
Stress tests: perturb transactions and measure prediction stability (resilience factor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger("aegis")


class ResilienceAnalyzer:
    """
    Resilience = 1 / (1 + Var(predictions across perturbations)).
    Lower resilience ⇒ unstable under stress.
    """

    def __init__(
        self,
        num_scenarios: int = 12,
        noise_std_amount: float = 0.15,
        spike_multiplier: float = 4.0,
    ) -> None:
        self.num_scenarios = num_scenarios
        self.noise_std_amount = noise_std_amount
        self.spike_multiplier = spike_multiplier

    def perturb(self, x: torch.Tensor, mask: torch.Tensor, scenario_id: int) -> torch.Tensor:
        """Perturb amount channel (index 0) — Gaussian noise, spikes, or shuffle padding-safe."""
        out = x.clone()
        valid = mask.bool()
        if scenario_id % 3 == 0:
            noise = torch.randn_like(out[:, :, 0]) * self.noise_std_amount
            out[:, :, 0] = out[:, :, 0] + noise * valid.float()
        elif scenario_id % 3 == 1:
            out[:, :, 0] = torch.where(valid, out[:, :, 0] * self.spike_multiplier, out[:, :, 0])
        else:
            out[:, :, 0] = out[:, :, 0] + torch.randn_like(out[:, :, 0]) * (2.0 * self.noise_std_amount) * valid.float()
        return out

    def run(
        self,
        predict_fn: Callable[[torch.Tensor, torch.Tensor], np.ndarray],
        x: torch.Tensor,
        mask: torch.Tensor,
        plot_path: Optional[str] = None,
    ) -> dict:
        preds_matrix: List[np.ndarray] = []
        for s in range(self.num_scenarios):
            xp = self.perturb(x, mask, s)
            p = predict_fn(xp, mask)
            preds_matrix.append(p)
        stack = np.stack(preds_matrix, axis=0)  # [S, B]
        var = np.var(stack, axis=0)
        resilience = 1.0 / (1.0 + var)
        mean_res = float(np.mean(resilience))
        scenario_means: List[float] = []
        for s in range(self.num_scenarios):
            scenario_means.append(float(np.mean(stack[s])))

        if plot_path:
            self._plot(resilience, scenario_means, plot_path)

        logger.info(
            "Resilience: mean=%.4f | per-sample variance mean=%.6f",
            mean_res,
            float(np.mean(var)),
        )
        return {
            "mean_resilience": mean_res,
            "per_sample_resilience": resilience,
            "prediction_variance": var,
            "scenario_mean_prediction": scenario_means,
        }

    def _plot(self, resilience: np.ndarray, scenario_means: List[float], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(np.arange(len(scenario_means)), scenario_means, marker="o", linewidth=1)
        axes[0].set_xlabel("Stress scenario index")
        axes[0].set_ylabel("Mean predicted risk")
        axes[0].set_title("Prediction drift across stress scenarios")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(np.arange(len(resilience)), resilience, marker="o", linewidth=1, color="darkgreen")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Resilience factor")
        axes[1].set_title("Per-sample resilience (higher = more stable)")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(p), dpi=120)
        plt.close()
        logger.info("Saved resilience plot to %s", path)

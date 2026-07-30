"""The three release figures: orbit continuation, section grid, extrapolation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .evaluate import rk4_step

TRUTH_COLOR = "#333333"
RF_COLOR = "#0a7d4f"
LAMBDA_CMAP = "coolwarm"


def _cloud(results_dir, family, tag, control):
    with np.load(Path(results_dir) / f"cloud_{family}_{tag}_{control:.2f}.npz") as data:
        return data["Y"], data["P"], data["L"]


def _scatter_section(ax, cloud, norm):
    y, p, lam = cloud
    order = np.argsort(lam)
    ax.scatter(y[order], p[order], c=lam[order], cmap=LAMBDA_CMAP, norm=norm,
               s=0.7, alpha=0.85, linewidths=0, rasterized=True)


def fig_trajectories(system, rf_field, out_path):
    """Configuration-space orbit continuation, truth vs RF-HNN (linear family)."""
    controls = (0.7, 0.95, 1.1)

    def initial(mu):
        y0, py0 = -0.27, -0.15
        kinetic = system.E0 - system.potential(0.0, y0, mu) - 0.5 * py0 * py0
        return np.array([[0.0, y0, np.sqrt(2.0 * max(kinetic, 1e-12)), py0]])

    def rollout(field, mu, n_steps=4000):
        state = initial(mu)
        xs, ys = np.full(n_steps, np.nan), np.full(n_steps, np.nan)
        for index in range(n_steps):
            state = rk4_step(field, state, mu, system.dt)
            if not np.all(np.isfinite(state)) or np.abs(state).max() > system.state_limit:
                break
            xs[index], ys[index] = state[0, 0], state[0, 1]
        return xs, ys

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
    for col, control in enumerate(controls):
        for row, (label, field, color) in enumerate(
            (("truth", system.vector_field, TRUTH_COLOR), ("RF-HNN", rf_field, RF_COLOR))
        ):
            xs, ys = rollout(field, control)
            ax = axes[row, col]
            ax.plot(xs, ys, color=color, lw=0.3, alpha=0.85)
            ax.set_xlim(-0.85, 0.85)
            ax.set_ylim(-0.85, 0.85)
            ax.set_aspect("equal")
            if row == 0:
                regime = "training" if control == 0.7 else "extrapolation"
                ax.set_title(f"$\\mu={control:g}$ ({regime})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\n$y$")
            if row == 1:
                ax.set_xlabel("$x$")
    fig.suptitle("Configuration-space orbits from one shared initial condition", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig_section_grid(results_dir, family, controls, vmax, title, out_path,
                     training_edge, symbol="\\mu"):
    """Poincare sections colored by the orbit-wise finite-time exponent."""
    norm = Normalize(0.0, vmax)
    fig, axes = plt.subplots(2, len(controls), figsize=(2.6 * len(controls) + 1.4, 5.8),
                             sharex=True, sharey=True)
    for col, control in enumerate(controls):
        for row, tag in enumerate(("truth", "rf")):
            ax = axes[row, col]
            ax.set_box_aspect(1)
            _scatter_section(ax, _cloud(results_dir, family, tag, control), norm)
            if row == 0:
                regime = "training" if control <= training_edge else "extrapolation"
                ax.set_title(f"${symbol}={control:g}$ ({regime})", fontsize=10)
            if col == 0:
                ax.set_ylabel(("truth" if tag == "truth" else "RF-HNN") + "\n$p_y$")
            if row == 1:
                ax.set_xlabel("$y$")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=LAMBDA_CMAP)
    fig.colorbar(mappable, ax=axes, shrink=0.85, label=r"$\lambda_T$")
    fig.suptitle(title, y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_extrapolation(results_dir, family, curve_controls, section_controls,
                      seeds, vmax, out_path, training_edge):
    """Occupancy and mean exponent across controls plus truth-vs-RF sections."""
    results_dir = Path(results_dir)

    def record(tag, control):
        return json.loads((results_dir / f"eval_{family.name}_{tag}_{control:.2f}.json").read_text())

    fig = plt.figure(figsize=(11.0, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 2.4], wspace=0.25)
    metric_grid = outer[0, 0].subgridspec(2, 1, hspace=0.12)
    section_grid = outer[0, 1].subgridspec(2, len(section_controls), wspace=0.08,
                                           hspace=0.10)

    for panel, name, label in ((0, "occupancy", "occupied cells"),
                               (1, "lambda_mean", r"mean $\lambda_T$")):
        ax = fig.add_subplot(metric_grid[panel, 0])
        truth_values = [record("truth", c)[name] for c in curve_controls]
        rf_values = np.array([[record(f"rf{seed}", c)[name] for c in curve_controls]
                              for seed in seeds])
        ax.axvspan(min(curve_controls), training_edge, color="#2166ac", alpha=0.08)
        ax.plot(curve_controls, truth_values, "-o", color=TRUTH_COLOR, ms=3,
                label="truth")
        ax.errorbar(curve_controls, rf_values.mean(axis=0), yerr=rf_values.std(axis=0),
                    marker="D", ms=3, capsize=2, color=RF_COLOR, label="RF-HNN")
        ax.set_ylabel(label)
        if panel == 0:
            ax.legend(fontsize=8)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("$\\mu$")

    norm = Normalize(0.0, vmax)
    for col, control in enumerate(section_controls):
        for row, tag in enumerate(("truth", "rf")):
            ax = fig.add_subplot(section_grid[row, col])
            ax.set_box_aspect(1)
            _scatter_section(ax, _cloud(results_dir, family.name, tag, control), norm)
            if row == 0:
                ax.set_title(f"$\\mu={control:g}$", fontsize=10)
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(("truth" if tag == "truth" else "RF-HNN") + "\n$p_y$")
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel("$y$")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=LAMBDA_CMAP)
    fig.colorbar(mappable, ax=fig.axes[2:], shrink=0.8, label=r"$\lambda_T$")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

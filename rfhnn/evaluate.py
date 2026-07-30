"""Frozen float64 evaluation protocol: RK4, Poincare section, Benettin exponent.

Truth and every learned field go through the same code path.  Section
crossings are collected on x = 0 with p_x > 0 (linear interpolation between
steps) while the Benettin pair advances inside the same RK4 calls; escaped
orbits are frozen and excluded from the Lyapunov average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

LYAPUNOV_T = 240.0  # paper protocol; lower together with max_steps for a quick demo
LYAPUNOV_D0 = 1e-5
DIRECTION_SEED = 3
SECTION_IC_COUNT = 28
SECTION_IC_SEED = 1
PROBE_STATES = 2500
PROBE_SEED_BASE = 8800
DENSE_ICS = 12
DENSE_CROSSINGS = 200
DENSE_MAX_STEPS = 50000


@dataclass(frozen=True)
class EvaluationSpec:
    dt: float
    crossings: int
    max_steps: int
    state_limit: float
    occupancy_range: tuple
    occupancy_min_points: int


def spec_for(system) -> EvaluationSpec:
    return EvaluationSpec(system.dt, system.crossings, system.max_steps,
                          system.state_limit, system.occupancy_range,
                          system.occupancy_min_points)


def dense_spec_for(system) -> EvaluationSpec:
    base = spec_for(system)
    return EvaluationSpec(base.dt, DENSE_CROSSINGS, DENSE_MAX_STEPS,
                          base.state_limit, base.occupancy_range,
                          base.occupancy_min_points)


def rk4_step(field, states, control, dt):
    states = np.asarray(states, dtype=np.float64)
    k1 = np.asarray(field(states, control), dtype=np.float64)
    k2 = np.asarray(field(states + 0.5 * dt * k1, control), dtype=np.float64)
    k3 = np.asarray(field(states + 0.5 * dt * k2, control), dtype=np.float64)
    k4 = np.asarray(field(states + dt * k3, control), dtype=np.float64)
    return states + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def shared_directions(n_ic, dimension=4, seed=DIRECTION_SEED):
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_ic, dimension))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def probe_states(system, control):
    """Held-out field-RMSE probe: fresh shell states, never used in any fit."""
    seed = PROBE_SEED_BASE + int(round(control * 1000.0))
    states = system.sample_shell(PROBE_STATES, seed, control)
    return states, np.asarray(system.vector_field(states, control), dtype=np.float64)


def chamfer_distance(model_y, model_p, truth_y, truth_p):
    if len(model_y) < 5 or len(truth_y) < 5:
        return float("nan")
    model = np.column_stack([model_y, model_p]).astype(np.float64, copy=False)
    truth = np.column_stack([truth_y, truth_p]).astype(np.float64, copy=False)
    if len(model) > 3000:
        model = model[np.linspace(0, len(model) - 1, 3000, dtype=int)]
    if len(truth) > 3000:
        truth = truth[np.linspace(0, len(truth) - 1, 3000, dtype=int)]
    return float(
        cKDTree(model).query(truth, k=1, workers=1)[0].mean()
        + cKDTree(truth).query(model, k=1, workers=1)[0].mean()
    )


def mean_occupied_cells(ys, ps, spec):
    ymin, ymax, pmin, pmax = spec.occupancy_range
    values = []
    for y_values, p_values in zip(ys, ps):
        if len(y_values) < spec.occupancy_min_points:
            continue
        histogram, _, _ = np.histogram2d(
            np.asarray(y_values), np.asarray(p_values),
            bins=20, range=[[ymin, ymax], [pmin, pmax]],
        )
        values.append(float(np.count_nonzero(histogram)))
    return float(np.mean(values)) if values else float("nan")


def evaluate_field(field, initial_states, control, spec, *,
                   lyapunov_t=LYAPUNOV_T, d0=LYAPUNOV_D0,
                   direction_seed=DIRECTION_SEED):
    states = np.asarray(initial_states, dtype=np.float64).copy()
    n_ic = len(states)
    directions = shared_directions(n_ic, states.shape[1], direction_seed)
    perturbed = states + d0 * directions
    accumulated = np.zeros(n_ic)
    lyapunov_valid = np.ones(n_ic, dtype=bool)
    escaped = np.zeros(n_ic, dtype=bool)

    ys = [[] for _ in range(n_ic)]
    ps = [[] for _ in range(n_ic)]
    previous = states.copy()
    crossing_counts = np.zeros(n_ic, dtype=np.int32)
    lyapunov_steps = int(round(lyapunov_t / spec.dt))

    for step_index in range(1, max(spec.max_steps, lyapunov_steps) + 1):
        need_section = step_index <= spec.max_steps and np.any(crossing_counts < spec.crossings)
        need_lyapunov = step_index <= lyapunov_steps
        if not need_section and not need_lyapunov:
            break

        previous[:] = states
        if need_lyapunov:
            both = rk4_step(field, np.concatenate([states, perturbed]), control, spec.dt)
            states, perturbed = both[:n_ic], both[n_ic:]
            delta = perturbed - states
            distance = np.linalg.norm(delta, axis=1)
            bad = ~np.isfinite(distance) | (distance <= 0.0)
            lyapunov_valid &= ~bad
            safe = np.where(bad, d0, distance)
            accumulated[lyapunov_valid] += np.log(safe[lyapunov_valid] / d0)
            perturbed = states + delta * (d0 / safe)[:, None]
        else:
            states = rk4_step(field, states, control, spec.dt)

        bad_state = ~np.isfinite(states).all(axis=1) | (
            np.abs(states) > spec.state_limit
        ).any(axis=1)
        newly_bad = bad_state & ~escaped
        if np.any(newly_bad):
            escaped |= newly_bad
            lyapunov_valid &= ~newly_bad
        if np.any(escaped):
            states[escaped] = 0.0
            perturbed[escaped] = d0 * directions[escaped]

        if need_section:
            x0, x1 = previous[:, 0], states[:, 0]
            candidates = np.where(
                (x0 < 0.0) & (x1 >= 0.0) & ~escaped & (crossing_counts < spec.crossings)
            )[0]
            for index in candidates:
                denominator = x1[index] - x0[index]
                alpha = -x0[index] / denominator if denominator != 0.0 else 1.0
                crossing = previous[index] + alpha * (states[index] - previous[index])
                if crossing[2] > 0.0:
                    ys[index].append(float(crossing[1]))
                    ps[index].append(float(crossing[3]))
                    crossing_counts[index] += 1

    lambdas = np.full(n_ic, np.nan)
    lambdas[lyapunov_valid] = accumulated[lyapunov_valid] / lyapunov_t
    section_y = np.concatenate([np.asarray(v) for v in ys if v]) if any(ys) else np.empty(0)
    section_p = np.concatenate([np.asarray(v) for v in ps if v]) if any(ps) else np.empty(0)
    point_ic = (np.concatenate([np.full(len(v), i, dtype=np.int32)
                                for i, v in enumerate(ys) if v])
                if any(ys) else np.empty(0, dtype=np.int32))
    finite = np.isfinite(lambdas)
    return {
        "occupancy": mean_occupied_cells(ys, ps, spec),
        "lambda": lambdas,
        "lambda_mean": float(np.mean(lambdas[finite])) if np.any(finite) else float("nan"),
        "Y": section_y,
        "P": section_p,
        "L": lambdas[point_ic] if len(point_ic) else np.empty(0),
        "I": point_ic,
        "complete_fraction": float(np.mean(crossing_counts >= spec.crossings)),
        "escaped_fraction": float(np.mean(escaped)),
        "lyapunov_valid_fraction": float(np.mean(lyapunov_valid)),
    }

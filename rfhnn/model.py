"""Depth-2 random-feature Hamiltonian neural network.

The hidden trunk is sampled once from a seed and stays fixed; the only fitted
object is the readout vector of a scalar Hamiltonian, obtained by one
closed-form ridge solve against vector-field targets.  The model field is
dz/dt = J grad_z H, Hamiltonian by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

J = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

DESIGN_CHUNK = 500


class RandomFeatureTrunk:
    """Two dense fixed-random hidden layers of equal width.

    One default_rng(seed) stream, in order: first-layer weights (width, 5)
    from N(0,1) with the state columns scaled by state_scale and the control
    column by control_scale; first bias U[-1,1]; second-layer weights times
    gain_layer2 / sqrt(width); second bias U[-1,1].  The trunk consumes the
    raw control value.
    """

    def __init__(self, *, width, state_scale, control_scale, gain_layer2,
                 seed, activation="tanh"):
        if activation not in {"tanh", "gelu", "softplus"}:
            raise ValueError(f"unsupported activation: {activation}")
        self.width = int(width)
        self.activation = str(activation)
        rng = np.random.default_rng(seed)
        first = rng.standard_normal((self.width, 5))
        first[:, :4] *= float(state_scale)
        first[:, 4] *= float(control_scale)
        bias1 = rng.uniform(-1.0, 1.0, self.width)
        second = rng.standard_normal((self.width, self.width)) * (
            float(gain_layer2) / np.sqrt(self.width)
        )
        bias2 = rng.uniform(-1.0, 1.0, self.width)
        self.weights = [first, second]
        self.biases = [bias1, bias2]

    def activate(self, values):
        if self.activation == "tanh":
            output = np.tanh(values)
            return output, 1.0 - output**2
        if self.activation == "gelu":
            cdf = ndtr(values)
            correction = np.zeros_like(values)
            central = np.abs(values) < 40.0
            v = values[central]
            correction[central] = v * np.exp(-0.5 * v**2) / np.sqrt(2.0 * np.pi)
            return values * cdf, cdf + correction
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
        return np.logaddexp(0.0, values), sigmoid

    def _inputs(self, states, control):
        states = np.asarray(states, dtype=np.float64)
        return np.column_stack([states, np.full(len(states), float(control))])

    def feature_jacobian(self, states, control):
        """Analytic d phi / d z, shape (batch, width, 4)."""
        inputs = self._inputs(states, control)
        features, slope1 = self.activate(inputs @ self.weights[0].T + self.biases[0])
        derivative = slope1[:, :, None] * self.weights[0][None, :, :4]
        _, slope2 = self.activate(features @ self.weights[1].T + self.biases[1])
        return slope2[:, :, None] * np.einsum(
            "om,bmj->boj", self.weights[1], derivative, optimize=True
        )

    def design(self, states, control):
        """Rows mapping the readout to the four field components."""
        derivative = self.feature_jacobian(states, control)
        basis = np.einsum("ij,bnj->bin", J, derivative, optimize=True)
        return basis.reshape(len(states) * 4, self.width)

    def field(self, states, control, readout):
        """J grad_z (readout . phi) via the adjoint pass."""
        inputs = self._inputs(states, control)
        features1, slope1 = self.activate(inputs @ self.weights[0].T + self.biases[0])
        _, slope2 = self.activate(features1 @ self.weights[1].T + self.biases[1])
        adjoint = slope2 * np.asarray(readout, dtype=np.float64)[None, :]
        adjoint = (adjoint @ self.weights[1]) * slope1
        gradient = adjoint @ self.weights[0][:, :4]
        return gradient @ J.T


def trunk_from_params(params: dict, seed: int) -> RandomFeatureTrunk:
    return RandomFeatureTrunk(
        width=params["width"],
        state_scale=params["state_scale"],
        control_scale=params["control_scale"],
        gain_layer2=params["gain_layer2"],
        activation=params.get("activation", "tanh"),
        seed=seed,
    )


def fit_readout(trunk, states_by_control, targets_by_control, ridge,
                chunk_size=DESIGN_CHUNK):
    """solve(A^T A + ridge I, A^T y), accumulated in float64 chunks."""
    ata = np.zeros((trunk.width, trunk.width))
    aty = np.zeros(trunk.width)
    for control, states in states_by_control.items():
        targets = np.asarray(targets_by_control[control], dtype=np.float64)
        for start in range(0, len(states), chunk_size):
            stop = min(start + chunk_size, len(states))
            design = trunk.design(states[start:stop], control)
            ata += design.T @ design
            aty += design.T @ targets[start:stop].reshape(-1)
    return np.linalg.solve(ata + float(ridge) * np.eye(trunk.width), aty)


def field_rmse(trunk, readout, states, targets, control):
    prediction = trunk.field(states, control, readout)
    return float(np.sqrt(np.mean((prediction - np.asarray(targets)) ** 2)))


class FittedRFHNN:
    """Callable vector field bundling a trunk with its fitted readout."""

    def __init__(self, trunk, readout):
        self.trunk = trunk
        self.readout = np.asarray(readout, dtype=np.float64)

    def __call__(self, states, control):
        return self.trunk.field(states, control, self.readout)

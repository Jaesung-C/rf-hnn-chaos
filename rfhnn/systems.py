"""Two-degree-of-freedom Hamiltonian families.

State order is (x, y, p_x, p_y); the control parameter enters the potential.
Training data are independent states rejection-sampled on the energy shell
H = E0 (no trajectories); all seed rules are fixed so every array is a pure
function of (family, role).
"""

from __future__ import annotations

import numpy as np


class _Family:
    def hamiltonian(self, state, control):
        state = np.asarray(state)
        x, y, px, py = np.moveaxis(state, -1, 0)
        return 0.5 * (px**2 + py**2) + self.potential(x, y, control)

    def sample_shell(self, count: int, seed: int, control: float) -> np.ndarray:
        """Independent shell states: uniform position proposals in the family box,
        rejection on negative kinetic energy, uniform momentum angle."""
        rng = np.random.default_rng(seed)
        accepted: list[list[float]] = []
        while len(accepted) < count:
            x = rng.uniform(*self.box_x, self.batch_factor * count)
            y = rng.uniform(*self.box_y, self.batch_factor * count)
            kinetic = self.E0 - self.potential(x, y, control)
            keep = kinetic > 0.0
            x, y, kinetic = x[keep], y[keep], kinetic[keep]
            speed = np.sqrt(2.0 * kinetic)
            angle = rng.uniform(0.0, 2.0 * np.pi, len(kinetic))
            block = np.stack([x, y, speed * np.cos(angle), speed * np.sin(angle)], axis=1)
            accepted.extend(block.tolist())
        return np.asarray(accepted[:count], dtype=np.float64)

    def training_set(self, control: float, count: int, seed: int):
        states = self.sample_shell(count, seed, control)
        return states, np.asarray(self.vector_field(states, control), dtype=np.float64)

    def section_ics(self, control: float, count: int = 28, seed: int = 1) -> np.ndarray:
        """Poincare-section initial conditions (x = 0, p_x > 0) of the frozen
        evaluation protocol; float32 storage is part of that protocol."""
        rng = np.random.default_rng(seed + int(control * 1000))
        accepted: list[list[float]] = []
        while len(accepted) < count:
            y = rng.uniform(*self.section_y)
            py = rng.uniform(*self.section_py)
            radicand = 2.0 * (self.E0 - self.potential(0.0, y, control)) - py * py
            if radicand > 0.0:
                accepted.append([0.0, y, np.sqrt(radicand), py])
        return np.asarray(accepted, dtype=np.float32)


class LinearHH(_Family):
    """Canonical Henon-Heiles with the cubic term scaled linearly by mu."""

    name = "linear"
    E0 = 0.13
    training_controls = (0.4, 0.5, 0.6, 0.7)
    search_fit_controls = (0.4, 0.5, 0.7)
    validation_control = 0.6
    box_x, box_y, batch_factor = (-0.7, 0.7), (-0.7, 0.9), 4
    section_y, section_py = (-0.6, 0.8), (-0.5, 0.5)
    # paper protocol; for a quick demo lower crossings/max_steps here and
    # LYAPUNOV_T in evaluate.py (numbers then deviate from the paper)
    dt, crossings, max_steps, state_limit = 0.05, 50, 9000, 8.0
    occupancy_range, occupancy_min_points = (-0.7, 0.9, -0.6, 0.6), 20

    @staticmethod
    def potential(x, y, mu):
        x = np.asarray(x)
        y = np.asarray(y)
        return 0.5 * (x**2 + y**2) + mu * (x**2 * y - y**3 / 3.0)

    @staticmethod
    def vector_field(state, mu):
        state = np.asarray(state)
        x, y, px, py = np.moveaxis(state, -1, 0)
        return np.stack((px, py, -x - 2.0 * mu * x * y, -y - mu * x * x + mu * y * y), axis=-1)


class BoundedHH(_Family):
    """Bounded variant: the cubic term scales with mu^2 and a quartic confining
    wall (kappa = 0.5) removes the escape channels."""

    name = "mu2"
    E0 = 0.16
    KAPPA = 0.5
    training_controls = (0.9, 1.0, 1.1, 1.2)
    search_fit_controls = (0.9, 1.0, 1.2)
    validation_control = 1.1
    box_x, box_y, batch_factor = (-0.9, 0.9), (-0.9, 0.9), 8
    section_y, section_py = (-0.9, 0.9), (-0.7, 0.7)
    dt, crossings, max_steps, state_limit = 0.05, 50, 9000, 6.0
    occupancy_range, occupancy_min_points = (-0.8, 0.8, -0.7, 0.7), 3

    @classmethod
    def potential(cls, x, y, mu):
        x = np.asarray(x)
        y = np.asarray(y)
        radius2 = x * x + y * y
        return 0.5 * radius2 + mu * mu * (x * x * y - y**3 / 3.0) + cls.KAPPA * radius2**2

    @classmethod
    def vector_field(cls, state, mu):
        state = np.asarray(state)
        x, y, px, py = np.moveaxis(state, -1, 0)
        radius2 = x * x + y * y
        fx = x + 2.0 * mu * mu * x * y + 4.0 * cls.KAPPA * x * radius2
        fy = y + mu * mu * (x * x - y * y) + 4.0 * cls.KAPPA * y * radius2
        return np.stack((px, py, -fx, -fy), axis=-1)


class Barbanis(_Family):
    """Barbanis-type coupling c x y^2 with a quartic confining wall
    (kappa = 0.3)."""

    name = "barbanis"
    E0 = 0.18
    KAPPA = 0.3
    training_controls = (0.8, 1.0, 1.2)
    search_fit_controls = (0.8, 1.2)
    validation_control = 1.0
    box_x, box_y, batch_factor = (-1.1, 1.1), (-1.1, 1.1), 8
    section_y = (-1.1, 1.1)
    section_py = (-np.sqrt(2.0 * E0 + 1.0), np.sqrt(2.0 * E0 + 1.0))
    dt, crossings, max_steps, state_limit = 0.04, 20, 15000, 5.2
    occupancy_range, occupancy_min_points = (-1.1, 1.1, -1.1, 1.1), 3

    @classmethod
    def potential(cls, x, y, c):
        x = np.asarray(x)
        y = np.asarray(y)
        radius2 = x * x + y * y
        return 0.5 * radius2 + c * x * y * y + cls.KAPPA * radius2**2

    @classmethod
    def vector_field(cls, state, c):
        state = np.asarray(state)
        x, y, px, py = np.moveaxis(state, -1, 0)
        radius2 = x * x + y * y
        fx = x + c * y * y + 4.0 * cls.KAPPA * x * radius2
        fy = y + 2.0 * c * x * y + 4.0 * cls.KAPPA * y * radius2
        return np.stack((px, py, -fx, -fy), axis=-1)


class MorsePair(_Family):
    """Two Morse oscillators (D = a = 1) with a bilinear coupling c x y, which
    is not globally confining; escaped orbits are recorded."""

    name = "morse"
    E0 = 0.18
    D = 1.0
    A = 1.0
    training_controls = (0.3, 0.4, 0.5)
    search_fit_controls = (0.3, 0.5)
    validation_control = 0.4
    box_x, box_y, batch_factor = (-0.8, 1.5), (-0.8, 1.5), 8
    section_y, section_py = (-0.8, 1.5), (-np.sqrt(2.0 * E0), np.sqrt(2.0 * E0))
    dt, crossings, max_steps, state_limit = 0.05, 50, 11000, 4.0
    occupancy_range, occupancy_min_points = (-0.8, 0.8, -0.7, 0.7), 3

    @classmethod
    def potential(cls, x, y, c):
        x = np.asarray(x)
        y = np.asarray(y)
        return (cls.D * (1.0 - np.exp(-cls.A * x)) ** 2
                + cls.D * (1.0 - np.exp(-cls.A * y)) ** 2 + c * x * y)

    @classmethod
    def vector_field(cls, state, c):
        state = np.asarray(state)
        x, y, px, py = np.moveaxis(state, -1, 0)
        exp_x = np.exp(-cls.A * x)
        exp_y = np.exp(-cls.A * y)
        dvx = 2.0 * cls.D * cls.A * exp_x * (1.0 - exp_x) + c * y
        dvy = 2.0 * cls.D * cls.A * exp_y * (1.0 - exp_y) + c * x
        return np.stack((px, py, -dvx, -dvy), axis=-1)


FAMILIES = {"linear": LinearHH(), "mu2": BoundedHH(), "barbanis": Barbanis(),
            "morse": MorsePair()}

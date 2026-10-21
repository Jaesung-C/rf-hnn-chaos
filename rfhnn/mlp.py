"""Backpropagated depth-2 MLP-HNN baseline used for comparison.

Each member is a 5-200-200-1 tanh network H(z, c) of the standardized state
(z - mz) / sz and the normalized control (c - offset) / scale; its field is
dz/dt = J grad_z H.  Members are trained by Adam on vector-field targets in
float64 (needs torch); the trained weights are stored in models/ and evaluated
here in numpy.  An ensemble is the field of the mean Hamiltonian of its members.
"""

from __future__ import annotations

import numpy as np

from .model import J

WIDTH = 200
MEMBERS = 20
SEED_BASE = 100
STATES_PER_CONTROL = 12000
BATCH_SIZE = 2048
LEARNING_RATE = 2e-3
TRAIN_THREADS = 3
STEPS = {"linear": 14000, "mu2": 12000, "barbanis": 14000, "morse": 14000}
CONTROL_NORMALIZATION = {"linear": (0.5, 0.6), "mu2": (0.8, 0.5),
                         "barbanis": (1.0, 0.6), "morse": (0.6, 0.4)}
ARRAYS = ("w1", "b1", "w2", "b2", "w3", "b3", "mz", "sz")


def normalized_control(system, control):
    offset, scale = CONTROL_NORMALIZATION[system.name]
    return (control - offset) / scale


def training_data(system, member):
    """Fresh shell states at every training control (seed round(1000 c) + 100
    + member), their vector fields, and the normalized controls."""
    states, targets, controls = [], [], []
    for control in system.training_controls:
        seed = int(round(1000.0 * control)) + SEED_BASE + member
        block, target = system.training_set(control, STATES_PER_CONTROL, seed)
        states.append(block)
        targets.append(target)
        controls.append(np.full(STATES_PER_CONTROL, control, dtype=np.float64))
    return (np.concatenate(states), np.concatenate(targets),
            normalized_control(system, np.concatenate(controls)))


def train_member(system, member):
    """Adam with a cosine schedule on the mean squared field error, batches of
    2048 drawn with replacement; returns the member's weights."""
    import torch
    from torch import nn

    torch.set_num_threads(TRAIN_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed = SEED_BASE + member
    torch.manual_seed(seed)
    states_np, targets_np, controls_np = training_data(system, member)
    states = torch.tensor(states_np, dtype=torch.float64)
    targets = torch.tensor(targets_np, dtype=torch.float64)
    controls = torch.tensor(controls_np, dtype=torch.float64)

    network = nn.Sequential(nn.Linear(5, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH),
                            nn.Tanh(), nn.Linear(WIDTH, 1)).double()
    mz = torch.tensor(states_np.mean(axis=0), dtype=torch.float64)
    sz = torch.tensor(states_np.std(axis=0) + 1e-6, dtype=torch.float64)

    def field(z, c):
        z = z.clone().requires_grad_(True)
        energy = network(torch.cat([(z - mz) / sz, c[:, None]], dim=1))
        gradient = torch.autograd.grad(energy.sum(), z, create_graph=True)[0]
        return torch.cat([gradient[:, 2:4], -gradient[:, 0:2]], dim=1)

    steps = STEPS[system.name]
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for _ in range(steps):
        batch = torch.randint(0, len(states), (BATCH_SIZE,))
        loss = ((field(states[batch], controls[batch]) - targets[batch]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"[mlp {system.name} member {member}] final minibatch RMSE = "
          f"{float(loss.detach()) ** 0.5:.3e}", flush=True)

    first, second, last = network[0], network[2], network[4]
    return {"w1": first.weight.detach().numpy().copy(), "b1": first.bias.detach().numpy().copy(),
            "w2": second.weight.detach().numpy().copy(), "b2": second.bias.detach().numpy().copy(),
            "w3": last.weight.detach().numpy()[0].copy(), "b3": float(last.bias.detach()[0]),
            "mz": mz.numpy().copy(), "sz": sz.numpy().copy()}


class MLPHNN:
    """Callable float64 field of one member, or of the mean Hamiltonian of
    several members (weights stacked along the first axis)."""

    def __init__(self, weights, system, members):
        index = np.atleast_1d(np.asarray(members, dtype=int))
        w1, b1, w2, b2, w3, _, mz, sz = (np.asarray(weights[key], dtype=np.float64)[index]
                                         for key in ARRAYS)
        self.state_weights = np.transpose(w1[:, :, :4], (0, 2, 1))
        self.control_weights = w1[:, None, :, 4]
        self.input_weights = w1[:, :, :4]
        self.bias1, self.bias2 = b1[:, None, :], b2[:, None, :]
        self.weights2, self.weights2_t = w2, np.transpose(w2, (0, 2, 1))
        self.weights3 = w3[:, None, :]
        self.mz, self.sz = mz[:, None, :], sz[:, None, :]
        self.offset, self.scale = CONTROL_NORMALIZATION[system.name]

    def __call__(self, states, control):
        """J grad_z of the mean member Hamiltonian via the adjoint pass."""
        states = np.asarray(states, dtype=np.float64)
        normalized = (float(control) - self.offset) / self.scale
        inputs = (states[None] - self.mz) / self.sz
        hidden1 = np.tanh(inputs @ self.state_weights + normalized * self.control_weights
                          + self.bias1)
        hidden2 = np.tanh(hidden1 @ self.weights2_t + self.bias2)
        adjoint = ((1.0 - hidden2**2) * self.weights3) @ self.weights2 * (1.0 - hidden1**2)
        gradient = (adjoint @ self.input_weights) / self.sz
        return gradient.mean(axis=0) @ J.T

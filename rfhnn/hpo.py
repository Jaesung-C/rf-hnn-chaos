"""Hyperparameter search for the RF-HNN readout fit.

Protocol: for every candidate, fit the readout on all training controls
EXCEPT the held-out validation control, and score the vector-field RMSE on
independent validation states at that control.  No extrapolation data enters
model selection.  The trunk activation is searched over tanh / gelu /
softplus (one seeded TPE study each); the best trial across activations wins.
"""

from __future__ import annotations

import numpy as np

from .model import fit_readout, field_rmse, trunk_from_params

TRIALS = 100
STARTUP_TRIALS = 24
FEATURE_SEED = 0
SAMPLER_SEEDS = {"linear": 1000, "mu2": 1001}
FIT_STATES_PER_CONTROL = 8000
VALIDATION_STATES = 3000
VALIDATION_SEED = 999

WIDTH_CANDIDATES = tuple(range(200, 2001, 200))
STATE_SCALE_RANGE = (0.05, 4.0)
CONTROL_SCALE_RANGE = (0.02, 6.0)
GAIN_LAYER2_RANGE = (0.03, 4.0)
LOG10_RIDGE_RANGE = (-16.0, -2.0)


def search_data(system):
    states, targets = {}, {}
    for control in system.search_fit_controls:
        states[control], targets[control] = system.training_set(
            control, FIT_STATES_PER_CONTROL, int(round(1000.0 * control))
        )
    validation = system.training_set(
        system.validation_control, VALIDATION_STATES, VALIDATION_SEED
    )
    return states, targets, validation


def search_activation(system, activation, trials=TRIALS):
    """One seeded TPE study; returns the best trial's params and RMSE."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_states, fit_targets, (val_states, val_targets) = search_data(system)

    def objective(trial):
        params = {
            "width": trial.suggest_categorical("width", WIDTH_CANDIDATES),
            "state_scale": trial.suggest_float("state_scale", *STATE_SCALE_RANGE, log=True),
            "control_scale": trial.suggest_float("control_scale", *CONTROL_SCALE_RANGE, log=True),
            "gain_layer2": trial.suggest_float("gain_layer2", *GAIN_LAYER2_RANGE, log=True),
            "log10_ridge": trial.suggest_float("log10_ridge", *LOG10_RIDGE_RANGE),
            "activation": activation,
        }
        trunk = trunk_from_params(params, FEATURE_SEED)
        readout = fit_readout(trunk, fit_states, fit_targets,
                              ridge=10.0 ** params["log10_ridge"])
        return field_rmse(trunk, readout, val_states, val_targets,
                          system.validation_control)

    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEEDS[system.name],
        n_startup_trials=STARTUP_TRIALS,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = dict(study.best_trial.params)
    params["activation"] = activation
    params["width"] = int(params["width"])
    return {"params": params, "validation_rmse": float(study.best_value),
            "trials": trials}


def search_family(system, activations=("tanh", "gelu", "softplus"), trials=TRIALS):
    results = {}
    for activation in activations:
        results[activation] = search_activation(system, activation, trials)
        print(f"[hpo {system.name}/{activation}] best validation RMSE = "
              f"{results[activation]['validation_rmse']:.3e}", flush=True)
    best = min(results, key=lambda a: results[a]["validation_rmse"])
    return {"selected": results[best]["params"],
            "validation_rmse": results[best]["validation_rmse"],
            "per_activation": {a: {"validation_rmse": r["validation_rmse"],
                                   "params": r["params"]}
                               for a, r in results.items()}}

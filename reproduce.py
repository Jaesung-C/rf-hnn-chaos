#!/usr/bin/env python3
"""Reproduce the RF-HNN chaos-extrapolation results and figures.

Pipeline (each stage runs only if its output is missing):
  1. models/selected_params.json   -- hyperparameter search (slow; needs optuna)
  2. readout ridge fits            -- always recomputed in memory (~minutes)
  3. models/mlp_<family>.npz       -- MLP-HNN baseline training (slow; needs torch)
  4. results/ evaluation caches    -- shipped; delete to recompute (~hours)
  5. figures/                      -- always rendered from the caches
The run ends with the RF-HNN vs MLP-HNN comparison of the four families.

Usage:
  python reproduce.py                 # regenerate whatever is missing + figures
  python reproduce.py --check         # additionally verify key numbers
  python reproduce.py --search        # force a fresh hyperparameter search
  python reproduce.py --trials N      # search budget per activation (default 100)
  python reproduce.py --train-mlp     # force retraining of the MLP-HNN baseline
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from rfhnn.evaluate import (DENSE_CROSSINGS, DENSE_ICS, SECTION_IC_COUNT,
                            SECTION_IC_SEED, dense_spec_for, evaluate_field,
                            probe_states, spec_for)
from rfhnn.mlp import ARRAYS, MEMBERS, MLPHNN
from rfhnn.model import FittedRFHNN, field_rmse, fit_readout, trunk_from_params
from rfhnn.systems import FAMILIES

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "selected_params.json"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
PROVENANCE = RESULTS / "provenance.json"

FEATURE_SEEDS = (12, 13, 14)
FIT_STATES_PER_CONTROL = 8000
MU2_CURVE_CONTROLS = (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5)
LINEAR_SECTION_CONTROLS = (0.6, 1.0, 1.1)
MU2_SECTION_CONTROLS = (1.3, 1.4, 1.5)
SINGLE_MLP_MEMBERS = (0, 1, 2)
# extrapolation controls of the RF-HNN vs MLP-HNN comparison; the last is c*
COMPARISON_CONTROLS = {"linear": (0.8, 0.95, 1.1), "mu2": (1.3, 1.4, 1.5),
                       "barbanis": (1.4, 1.6), "morse": (0.6, 0.7, 0.8, 1.1)}

# float64-protocol reference values (this machine); truth is deterministic,
# RF tolerances cover ridge-solve BLAS sensitivity and fresh-search variation.
REFERENCE = {
    "linear": {"control": 1.1, "truth": {"occupancy": 37.64, "lambda_mean": 0.0950}},
    "mu2": {"control": 1.5, "truth": {"occupancy": 43.86, "lambda_mean": 0.2631}},
    "barbanis": {"control": 1.6, "truth": {"occupancy": 14.82, "lambda_mean": 0.0626}},
    "morse": {"control": 1.1, "truth": {"occupancy": 42.14, "lambda_mean": 0.1718}},
}


def _params_hash():
    return hashlib.sha256(MODELS.read_bytes()).hexdigest()


def _mlp_path(name):
    return HERE / "models" / f"mlp_{name}.npz"


def _atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_search(trials):
    from rfhnn.hpo import search_family

    selected = {}
    for name, system in FAMILIES.items():
        print(f"=== hyperparameter search: {name} ===", flush=True)
        selected[name] = search_family(system, trials=trials)
    MODELS.parent.mkdir(exist_ok=True)
    _atomic_write(MODELS, json.dumps(selected, indent=2))
    print(f"wrote {MODELS}", flush=True)
    for stale in (RESULTS, FIGURES):
        if stale.exists():
            shutil.rmtree(stale)
    print("cleared results/ and figures/ (model parameters changed)", flush=True)


def fit_models(selected):
    """Ridge-fit every readout on the full training controls (in memory)."""
    fields = {}
    for name, system in FAMILIES.items():
        params = selected[name]["selected"]
        states, targets = {}, {}
        for control in system.training_controls:
            states[control], targets[control] = system.training_set(
                control, FIT_STATES_PER_CONTROL, int(round(1000.0 * control))
            )
        fields[name] = {}
        for seed in FEATURE_SEEDS:
            trunk = trunk_from_params(params, seed)
            readout = fit_readout(trunk, states, targets,
                                  ridge=10.0 ** params["log10_ridge"])
            fields[name][seed] = FittedRFHNN(trunk, readout)
        probe, probe_targets = probe_states(system, system.validation_control)
        rmse = field_rmse(fields[name][FEATURE_SEEDS[0]].trunk,
                          fields[name][FEATURE_SEEDS[0]].readout,
                          probe, probe_targets, system.validation_control)
        print(f"[fit {name}] {params['activation']} width {params['width']}; "
              f"in-band probe RMSE (seed {FEATURE_SEEDS[0]}) = {rmse:.3e}", flush=True)
    return fields


def _train_job(job):
    from rfhnn.mlp import train_member

    name, member = job
    return train_member(FAMILIES[name], member)


def ensure_mlp(force, workers):
    """Train the MLP-HNN members of every family without stored weights."""
    for name in FAMILIES:
        path = _mlp_path(name)
        if path.exists() and not force:
            continue
        print(f"=== MLP-HNN training: {name} ({MEMBERS} members) ===", flush=True)
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context("spawn")) as pool:
            members = list(pool.map(_train_job, [(name, member) for member in range(MEMBERS)]))
        _atomic_savez(path, **{key: np.stack([member[key] for member in members])
                               for key in ARRAYS})
        print(f"wrote {path}", flush=True)
        for pattern in (f"eval_{name}_mlp*.json", f"cloud_{name}_mlp*.npz"):
            for stale in RESULTS.glob(pattern):
                stale.unlink()


def _eval_job(job):
    name, tag, control, dense, model = job
    system = FAMILIES[name]
    if model is None:
        field = system.vector_field
    elif model[0] == "mlp":
        with np.load(_mlp_path(name)) as weights:
            field = MLPHNN(weights, system, model[1])
    else:
        params, seed, readout = model
        field = FittedRFHNN(trunk_from_params(params, seed), readout)

    spec = dense_spec_for(system) if dense else spec_for(system)
    ic_count = 12 if dense else SECTION_IC_COUNT
    initial = np.asarray(system.section_ics(control, ic_count, SECTION_IC_SEED),
                         dtype=np.float64)
    item = evaluate_field(field, initial, control, spec)
    if dense:
        _atomic_savez(RESULTS / f"cloud_{name}_{tag}_{control:.2f}.npz",
                      Y=item["Y"], P=item["P"], L=item["L"])
    else:
        record = {"family": name, "tag": tag, "control": control,
                  "occupancy": item["occupancy"], "lambda_mean": item["lambda_mean"],
                  "complete_fraction": item["complete_fraction"],
                  "escaped_fraction": item["escaped_fraction"],
                  "lyapunov_valid_fraction": item["lyapunov_valid_fraction"]}
        _atomic_write(RESULTS / f"eval_{name}_{tag}_{control:.2f}.json",
                      json.dumps(record, indent=2, allow_nan=False))
    print(f"[eval {name} {tag} c={control:g}{' dense' if dense else ''}] done", flush=True)
    return None


def ensure_results(selected, fields, workers):
    RESULTS.mkdir(exist_ok=True)
    current = _params_hash()
    existing = [p for p in RESULTS.iterdir() if p != PROVENANCE]
    if existing:
        recorded = (json.loads(PROVENANCE.read_text()).get("selected_params_sha256")
                    if PROVENANCE.exists() else None)
        if recorded != current:
            raise SystemExit(
                "results/ was produced with different model parameters "
                "(provenance mismatch); delete results/ to regenerate")

    wanted = {}

    def want(name, tag, control, dense, payload):
        key = (f"cloud_{name}_{tag}_{control:.2f}.npz" if dense
               else f"eval_{name}_{tag}_{control:.2f}.json")
        if not (RESULTS / key).exists() and key not in wanted:
            wanted[key] = (name, tag, control, dense, payload)

    def model(name, seed):
        return (selected[name]["selected"], seed, fields[name][seed].readout)

    for control in LINEAR_SECTION_CONTROLS:
        want("linear", "truth", control, True, None)
        want("linear", "rf", control, True, model("linear", FEATURE_SEEDS[0]))
        want("linear", "mlp0", control, True, ("mlp", SINGLE_MLP_MEMBERS[0]))
    for control in MU2_SECTION_CONTROLS:
        want("mu2", "truth", control, True, None)
        want("mu2", "rf", control, True, model("mu2", FEATURE_SEEDS[0]))
        want("mu2", "mlp0", control, True, ("mlp", SINGLE_MLP_MEMBERS[0]))
    for control in MU2_CURVE_CONTROLS:
        want("mu2", "truth", control, False, None)
        for seed in FEATURE_SEEDS:
            want("mu2", f"rf{seed}", control, False, model("mu2", seed))
        for member in SINGLE_MLP_MEMBERS:
            want("mu2", f"mlp{member}", control, False, ("mlp", member))
    for name, controls in COMPARISON_CONTROLS.items():
        for control in controls:
            want(name, "truth", control, False, None)
            for seed in FEATURE_SEEDS:
                want(name, f"rf{seed}", control, False, model(name, seed))
            for member in SINGLE_MLP_MEMBERS:
                want(name, f"mlp{member}", control, False, ("mlp", member))
            want(name, "mlpens", control, False, ("mlp", tuple(range(MEMBERS))))

    _atomic_write(PROVENANCE, json.dumps({"selected_params_sha256": current}, indent=2))
    if wanted:
        print(f"computing {len(wanted)} missing evaluation caches ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context("spawn")) as pool:
            list(pool.map(_eval_job, list(wanted.values())))


def make_figures(selected, fields):
    from rfhnn.figures import fig_extrapolation, fig_section_grid, fig_trajectories

    FIGURES.mkdir(exist_ok=True)
    with np.load(_mlp_path("linear")) as weights:
        mlp_field = MLPHNN(weights, FAMILIES["linear"], SINGLE_MLP_MEMBERS[0])
    fig_trajectories(FAMILIES["linear"], fields["linear"][FEATURE_SEEDS[0]],
                     mlp_field, FIGURES / "linear_orbits.png")
    fig_section_grid(RESULTS, "linear", LINEAR_SECTION_CONTROLS, vmax=0.16,
                     title="Linear Henon-Heiles Poincare sections",
                     out_path=FIGURES / "linear_sections.png", training_edge=0.7)
    fig_extrapolation(RESULTS, FAMILIES["mu2"], MU2_CURVE_CONTROLS,
                      MU2_SECTION_CONTROLS, FEATURE_SEEDS, vmax=0.30,
                      out_path=FIGURES / "mu2_extrapolation.png",
                      training_edge=1.2, mlp_members=SINGLE_MLP_MEMBERS)
    for figure in sorted(FIGURES.glob("*.png")):
        print(f"wrote {figure}", flush=True)


def print_comparison():
    """Mean absolute occupancy error over the extrapolation controls and the
    finite-time exponent at c*, for the RF-HNN (feature seeds 12-14), single
    MLP-HNNs (members 0-2), and the 20-member MLP-HNN ensemble."""

    def record(name, tag, control):
        return json.loads((RESULTS / f"eval_{name}_{tag}_{control:.2f}.json").read_text())

    models = (("RF-HNN", [f"rf{seed}" for seed in FEATURE_SEEDS]),
              ("single MLP-HNN", [f"mlp{member}" for member in SINGLE_MLP_MEMBERS]),
              ("MLP-HNN ensemble", ["mlpens"]))
    for name, controls in COMPARISON_CONTROLS.items():
        truth = [record(name, "truth", control) for control in controls]
        line = f"[compare {name} c*={controls[-1]:g}] truth lam {truth[-1]['lambda_mean']:.3f}"
        for label, tags in models:
            occupancy = [np.mean([record(name, tag, control)["occupancy"] for tag in tags])
                         for control in controls]
            error = np.mean([abs(value - item["occupancy"])
                             for value, item in zip(occupancy, truth)])
            lam = np.mean([record(name, tag, controls[-1])["lambda_mean"] for tag in tags])
            line += f" | {label} occ error {error:.2f} lam {lam:.3f}"
        print(line, flush=True)


def run_check():
    failures = []
    eval_files = sorted(RESULTS.glob("eval_*.json"))
    cloud_files = sorted(RESULTS.glob("cloud_*.npz"))
    for path in eval_files:
        record = json.loads(path.read_text())
        summary = (record["occupancy"], record["lambda_mean"],
                   record["complete_fraction"], record["escaped_fraction"])
        if not all(np.isfinite(v) for v in summary):
            failures.append(f"{path.name}: non-finite summary values")
        if record["complete_fraction"] < 1.0 or record["escaped_fraction"] > 0.0:
            failures.append(f"{path.name}: incomplete or escaped orbits")
        if record.get("lyapunov_valid_fraction", 1.0) < 1.0:
            failures.append(f"{path.name}: invalid Lyapunov pairs")
    for path in cloud_files:
        with np.load(path) as data:
            if any(key not in data.files for key in ("Y", "P", "L")):
                failures.append(f"{path.name}: missing arrays")
                continue
            if len(data["Y"]) != DENSE_ICS * DENSE_CROSSINGS:
                failures.append(f"{path.name}: {len(data['Y'])} points, "
                                f"expected {DENSE_ICS * DENSE_CROSSINGS}")
            if not all(np.isfinite(data[key]).all() for key in ("Y", "P", "L")):
                failures.append(f"{path.name}: non-finite entries")
    print(f"[check] validated {len(eval_files)} eval records and "
          f"{len(cloud_files)} section clouds", flush=True)
    for name, ref in REFERENCE.items():
        control = ref["control"]
        truth = json.loads((RESULTS / f"eval_{name}_truth_{control:.2f}.json").read_text())
        rf = json.loads(
            (RESULTS / f"eval_{name}_rf{FEATURE_SEEDS[0]}_{control:.2f}.json").read_text())
        for record, label in ((truth, "truth"), (rf, "RF")):
            values = (record["occupancy"], record["lambda_mean"])
            if not all(np.isfinite(v) for v in values):
                failures.append(f"{name} {label}: non-finite metrics {values}")
            if record["complete_fraction"] < 1.0 or record["escaped_fraction"] > 0.0:
                failures.append(f"{name} {label}: complete={record['complete_fraction']:.2f} "
                                f"escaped={record['escaped_fraction']:.2f}")
        if abs(truth["occupancy"] - ref["truth"]["occupancy"]) > 0.5:
            failures.append(f"{name}: truth occupancy {truth['occupancy']:.2f}")
        if abs(truth["lambda_mean"] - ref["truth"]["lambda_mean"]) > 0.003:
            failures.append(f"{name}: truth lambda {truth['lambda_mean']:.4f}")
        if abs(rf["lambda_mean"] - truth["lambda_mean"]) > 0.15 * truth["lambda_mean"]:
            failures.append(f"{name}: RF lambda {rf['lambda_mean']:.4f} vs "
                            f"truth {truth['lambda_mean']:.4f}")
        if abs(rf["occupancy"] - truth["occupancy"]) > 2.5:
            failures.append(f"{name}: RF occupancy {rf['occupancy']:.2f} vs "
                            f"truth {truth['occupancy']:.2f}")
        print(f"[check {name} c={control:g}] truth occ {truth['occupancy']:.2f} "
              f"lam {truth['lambda_mean']:.4f} | RF occ {rf['occupancy']:.2f} "
              f"lam {rf['lambda_mean']:.4f}", flush=True)
    if failures:
        raise SystemExit("CHECK FAILED:\n  " + "\n  ".join(failures))
    print("all checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true",
                        help="force a fresh hyperparameter search")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--train-mlp", action="store_true",
                        help="force retraining of the MLP-HNN baseline")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.search or not MODELS.exists():
        run_search(args.trials)
    selected = json.loads(MODELS.read_text())
    fields = fit_models(selected)
    ensure_mlp(args.train_mlp, args.workers)
    ensure_results(selected, fields, args.workers)
    make_figures(selected, fields)
    print_comparison()
    if args.check:
        run_check()


if __name__ == "__main__":
    main()

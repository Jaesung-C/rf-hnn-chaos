# rf-hnn-chaos

Random-feature Hamiltonian neural networks (RF-HNN) that extrapolate the
emergence of Hamiltonian chaos in the control parameter, for two
two-degree-of-freedom families: the canonical Henon-Heiles system (cubic term
scaled linearly by mu) and its bounded mu^2 variant with a quartic confining
wall.

The model is a depth-2 fixed random trunk taking (x, y, p_x, p_y, c); the only
fitted object is the readout of a scalar Hamiltonian, obtained by one
closed-form ridge solve against exact vector-field targets sampled on the
energy shell at a few training controls where regular motion dominates.
Everything runs in IEEE double precision.

## Install and run

```
pip install -r requirements.txt        # numpy, scipy, matplotlib (+ optuna for --search)
python reproduce.py --check
```

With the shipped `models/selected_params.json` and `results/` caches this
refits the readouts, renders the three figures into `figures/`, and verifies
key numbers — a few minutes on a laptop.

## Regeneration chain

Each stage recomputes only if its output is missing:

| Deleted | Recomputed by | Cost |
|---|---|---|
| `models/selected_params.json` | seeded hyperparameter search (100 TPE trials per activation tanh/gelu/softplus per family; `--trials` to change) | hours |
| readouts | ridge refit from `selected_params.json` (always rerun; nothing stored) | minutes |
| `results/*` | frozen evaluation protocol (RK4, Poincare section at x=0 with p_x>0, Benettin exponent with T=240, d0=1e-5) | hours (use `--workers`) |
| `figures/*` | rendered from `results/` | seconds |

No data files exist anywhere: every training, validation, and evaluation
array is generated deterministically from fixed seeds.

## Figures

- `linear_orbits.png` — configuration-space orbits from one shared
  initial condition, truth vs RF-HNN, at a training control and two
  extrapolation controls.
- `linear_sections.png` — Poincare sections colored by the orbit-wise
  finite-time Lyapunov exponent, truth vs RF-HNN.
- `mu2_extrapolation.png` — occupancy and mean exponent across the
  control range plus sections through the regular-to-chaotic transition of
  the bounded family.

## Protocol notes

- Hyperparameters are selected with a held-out training control (linear: fit
  on mu = 0.4, 0.5, 0.7, validate at 0.6; bounded: fit on 0.9, 1.0, 1.2,
  validate at 1.1); no extrapolation data enters model selection. The shipped
  `selected_params.json` carries the paper's selected hyperparameters,
  revalidated with this repository's pipeline. Running `--search` repeats the
  selection from scratch, overwrites `selected_params.json`, and clears
  `results/` and `figures/` so stale caches cannot mix with the new model;
  search winners vary slightly with the sampler seed but reproduce the same
  extrapolation behavior.
- The default trunk activation in the code is tanh; the shipped parameters
  use the activation the search selected.
- The weakly regularized ridge solve is sensitive to BLAS threading at the
  coefficient level; induced fields agree to ~1e-6 and the `--check`
  tolerances account for this.
- Error bars in the extrapolation figure show the spread over the three
  feature seeds, not initial-condition sampling uncertainty.
- One deliberate exception to end-to-end float64: section initial
  conditions are stored in float32 (part of the frozen evaluation
  protocol) and promoted to float64 before integration.

## License

MIT (see LICENSE).

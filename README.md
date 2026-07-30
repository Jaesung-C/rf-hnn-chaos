# rf-hnn-chaos

Code companion for random-feature Hamiltonian neural networks (RF-HNNs)
extrapolating from predominantly regular training regimes to unseen chaotic
regimes.

This repository covers two two-degree-of-freedom Hamiltonian families: the
canonical Henon-Heiles system and a bounded mu-squared variant. It is a
focused subset of the experiments in the associated manuscript.

## Run

```bash
pip install -r requirements.txt
python reproduce.py --check
```

The command uses the included hyperparameters and cached evaluations, refits
the model readouts, and writes the reproduced figures to `figures/`.

To repeat hyperparameter selection from scratch:

```bash
python reproduce.py --search --trials 100
```

This takes substantially longer. No external dataset is required; all
training, validation, and evaluation arrays are generated deterministically.
Extrapolation data is not used for fitting or model selection.

## License

MIT (see `LICENSE`).

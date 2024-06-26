# plnn: A Machine Learning approach to Parameterized Landscapes

## TODO
* Implement heteroclinic orbit algorithm minimizing orthogonal component of path connecting saddles.

## Ideas
* Precompute summary stat for $x_1$ data
* Adjoint method.
* batch normalization and dropout
* Normalize data beforehand?

# Installation
For the CPU:
```bash
mamba create -p ./env python=3.9 jax numpy matplotlib pytorch torchvision equinox optax ipykernel pytest
mamba activate env
pip install diffrax==0.4.1
```

For the GPU, specifying cuda toolkit 11.2:
```bash
mamba create -p <env-path> python=3.9 pytorch=1.11[build=cuda112*] numpy=1.25 matplotlib=3.7 pytest=7.4 tqdm ipykernel ipywidgets
mamba activate env
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax==0.1.7 diffrax==0.4.1 equinox==0.11.2
```

Then, to install the project,
```bash
pip install -e .
```

# Usage

### Main training script

### Notebooks

### Testing
Tests are located in the directory `tests/` and organized as follows.

* Model tests in `tests/test_models/`:
    * `DeepPhiPLNN`: `tests/test_models/test_deep_phi_plnn.py`
    * `NEDeepPhiPLNN`: `tests/test_models/test_ne_deep_phi_plnn.py`
* Temporary data should be stored in the directory `tests/tmp/`.
* Benchmarking tests in directory `tests/benchmarking/`.
* Miscellaneous data in `tests/data/`.

### Benchmarking

# References

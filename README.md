# plnn: A Machine Learning approach to Parameterized Landscapes

## TODO

* Remove conditionals in simulate_forward
* No bias for signal mapping (Completed)
* Ensure noise is not too great
* Constrained landscape. Repellor at infinity.

## Ideas

* How to handle noise.
* Infer noise parameter. (Completed-ish)
* Incorporate the number of desired fixed points
* Precompute summary stat for $x_1$ data
* Adjoint method.
* Symmetry in the order of the cells within the population: $\boldsymbol{x}_i$ (Completed?)
* Parallelize batches. (Completed).
* Parallelize individual simulations. (Completed)
* Customizable layer architecture
* batch normalization and dropout
* Autocorrelation time of individual cells to determine transitioning flag. 
* Softmax activation prevents super-linear growth in the potential function?
* Normalize data beforehand?

# Installation
For the CPU:
```bash
# mamba create -p <env-path> python=3.8 pytorch=1.11 numpy=1.24 matplotlib= 3.7 pytest=7.4 tqdm ipykernel ipywidgets
mamba create -p ./env python=3.9 jax numpy matplotlib pytorch torchvision equinox optax ipykernel pytest
mamba activate env
pip install diffrax==0.4.1
```

For the GPU, specifying cuda toolkit 11.2:
```bash
mamba create -p <env-path> python=3.8 pytorch=1.11[build=cuda112*] numpy=1.24 matplotlib= 3.7 pytest=7.4 tqdm ipykernel ipywidgets
```

For M1 Macs, where we want the MPS device:
```bash
mamba create -p <env-path> python=3.8 pytorch=2.1 numpy=1.24 matplotlib= 3.7 pytest=7.4 tqdm ipykernel ipywidgets
mamba activate <env-path>
mamba install -c pytorch pytorch=2.1 torchvision
```

Then, to install the project phiml,
```bash
pip install -e .
```

# Usage

### Main training script

### Notebooks

### Testing

### Benchmarking

# References

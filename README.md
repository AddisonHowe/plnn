# PLNN: A machine learning approach to parameterized landscapes

This repository implements the Parameterized Landscape Neural Network architecture described in "[Dynamical systems theory informed learning of cellular differentiation landscapes](https://journals.aps.org/prx/abstract/10.1103/8vpj-bj7d)."
An application using this architecture is contained in a separate repository, which can be found at [https://github.com/AddisonHowe/dynamical-landscape-inference](https://github.com/AddisonHowe/dynamical-landscape-inference).


## Setup

### Standard setup (CPU)

For a standard setup, without GPU acceleration, and an editable installation, clone the repository and create a conda environment as follows:

```bash
git clone https://github.com/AddisonHowe/plnn
cd plnn
conda env create -p ./env -f environment.yml
```

### GPU support (CUDA 12.4)

For a setup enabling GPU support (CUDA 12.4) run

```bash
git clone https://github.com/AddisonHowe/plnn
cd plnn
conda env create -p ./env -f environment-cuda.yml
```

If desired, check that things have been installed correctly and that all tests are passing.
Note that this may take some time.

```bash
export JAX_ENABLE_X64=1
pytest tests/  # run only non-GPU tests
# pytest --use_gpu tests/  # include GPU-related tests:
```

### Alternative installations

Alternatively, the following commands might be used to create the conda environment. 
(Please note that when it comes to GPU support, these installations may be sensitive to your system's configuration.)

#### Standard setup (CPU)

```bash
conda create -p ./env python=3.9 jax=0.4 numpy=1.26 matplotlib=3.8 scikit-learn=1.5 pytorch=2.0 torchvision equinox=0.11 optax=0.1 pyyaml=6.0 tqdm ipykernel ipywidgets pytest --yes
conda activate env
pip install diffrax==0.6.0
```

#### GPU support (CUDA 12.4)

```bash
conda create -p ./env python=3.9 numpy=1.25 matplotlib=3.8 scikit-learn=1.5 pytest=7.4 cuda-compat=12.4 pyyaml=6.0 tqdm ipykernel ipywidgets --yes
conda activate env
pip install jax[cuda12] optax==0.1.7 diffrax==0.6.0 equinox==0.11.5
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### GPU support (CUDA 12.0)

```bash
conda create -p ./env python=3.9 numpy=1.25 matplotlib=3.8 scikit-learn=1.5 pytest=7.4 cuda-compat=12.4 pyyaml=6.0 tqdm ipykernel ipywidgets --yes
conda activate env
pip install --upgrade pip
pip install jax[cuda12] optax==0.1.7 diffrax==0.6.0 equinox==0.11.5 torch==2.0.1 torchvision torchaudio
```

# Usage

## Tests

Tests are located in the `tests/` directory and can be run as followed:

```bash
export JAX_ENABLE_X64=1

# Not including GPU-related tests
pytest tests/

# To include GPU-related tests:
pytest --use_gpu tests/
```


# Acknowledgments
This work was inspired by the work of Sáez et al. in [Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/).


# References
[1] Sáez M, Blassberg R, Camacho-Aguilar E, Siggia ED, Rand DA, Briscoe J. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 2022 Jan 19;13(1):12-28.e3. doi: 10.1016/j.cels.2021.08.013. Epub 2021 Sep 17. PMID: 34536382; PMCID: PMC8785827.

# PLNN: A machine learning approach to parameterized landscapes

This repository implements the Parameterized Landscape Neural Network architecture described in "[Dynamical systems theory informed learning of cellular differentiation landscapes](https://www.biorxiv.org/content/10.1101/2024.09.21.614191v1)."
An application using this architecture is contained in a separate repository, which can be found at [https://github.com/AddisonHowe/dynamical-landscape-inference](https://github.com/AddisonHowe/dynamical-landscape-inference).


## Setup
Basic setup, without GPU acceleration:
```bash
conda create -p ./env python=3.9 jax=0.4 numpy=1.26 matplotlib=3.8 scikit-learn=1.5 pytorch=2.0 torchvision equinox=0.11 optax=0.1 tqdm ipykernel pytest
conda activate env
pip install diffrax==0.6.0
```

For GPU support:
```bash
conda create -p ./env python=3.9 numpy=1.25 matplotlib=3.7 scikit-learn=1.5 pytest=7.4 cuda-compat=12.4 tqdm ipykernel ipywidgets --yes
conda activate env
pip install --upgrade pip
pip install jax[cuda12] optax==0.1.7 diffrax==0.6.0 equinox==0.11.5 torch==2.0.1 torchvision torchaudio
```

Then, to install the project,
```bash
pip install -e .
```

Check that things have been installed correctly and that all tests are passing.
```bash
export JAX_ENABLE_X64=1
pytest tests/  # run only non-GPU tests
# pytest --use_gpu tests/  # include GPU-related tests:
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


# PyTorch Multiscale Parametric t-SNE (MSP-tSNE)

![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![W&B Sweeps](https://img.shields.io/badge/W%26B-Sweeps-orange)](https://wandb.ai/)

PyTorch implementation of [Multiscale Parametric t-SNE](https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE)

## Table of Contents

- [PyTorch Multiscale Parametric t-SNE (MSP-tSNE)](#pytorch-multiscale-parametric-t-sne-msp-tsne)
	- [Table of Contents](#table-of-contents)
	- [Getting Started](#getting-started)
		- [Prerequisites](#prerequisites)
		- [Installation](#installation)
	- [Sklearn Pipeline Usage](#sklearn-pipeline-usage)
	- [Training Script](#training-script)
	- [Hyperparameter Tuning \& W\&B Integration](#hyperparameter-tuning--wb-integration)
	- [Neighborhood Preservation Metric](#neighborhood-preservation-metric)
	- [Model Artifacts](#model-artifacts)
	- [Acknowledgements](#acknowledgements)

---

## Getting Started

### Prerequisites

Tested with Python 3.6+. All required packages are listed in `setup.py` and `requirements.txt`.

### Installation

```bash
pip install .
```

---

## Sklearn Pipeline Usage

Run MSP t-SNE using sklearn's Pipeline:

```bash
python scripts/train_sklearn_tsne.py --config configs/sklearn_config.yml
```

**Features:**

- Loads sklearn digits dataset
- Preprocessing (StandardScaler/MinMaxScaler)
- MSP t-SNE embedding
- Evaluates trustworthiness & silhouette score
- Saves pipeline, embeddings, metrics

**Example config:**

```yaml
data:
 source: sklearn_digits
preprocessing:
 scaler: StandardScaler
algorithm:
 n_components: 2
 n_iter: 1000
 batch_size: 500
 nl1: 1000
 nl2: 500
 nl3: 250
output:
 model_path: "./models/msp_tsne_pipeline.pkl"
 embeddings_path: "./models/msp_tsne_embeddings.npy"
```

---

## Training Script

Train MSP t-SNE with custom data/config:

```bash
python scripts/train_msp_tsne.py --config configs/train_config.yml [--seed SEED]
```

**Arguments:**

- `--config`: Path to YAML config (required)
- `--seed`: Random seed (default: 42)

**Example config:**

```yaml
data:
 features: null
 labels: null
 label_column: null
 format: auto
preprocessing:
 scaler: StandardScaler
model:
 n_components: 2
 n_iter: 1000
 batch_size: 500
 early_exaggeration_epochs: 50
 early_exaggeration_value: 4.0
 early_stopping_epochs: 1000
 early_stopping_min_improvement: 0.01
```

---

## Hyperparameter Tuning & W&B Integration

Automated hyperparameter sweeping with Weights & Biases (W&B):

**Script:** `scripts/hyperparameter_tuning.py`

**Config:** `configs/hp_config.yml`

**Usage:**

```bash
python scripts/hyperparameter_tuning.py --config configs/hp_config.yml --mode sweep --count 50
python scripts/hyperparameter_tuning.py --config configs/hp_config.yml --mode smoke_test
```

**Features:**

- W&B integration for experiment tracking
- Bayesian optimization sweeps
- Full parameter search space (neural architecture, training, optimization, preprocessing)
- Automatic logging of metrics & hyperparameters
- Parallel execution with `wandb.agent()`

**Metrics Tracked:**

- Trustworthiness score (primary)
- Silhouette score
- Neighborhood preservation ratio
- Training time
- Final KL divergence loss

---

## Neighborhood Preservation Metric

Measures how well k-nearest neighbor relationships are preserved between original and embedded spaces.

- Uses sklearn's NearestNeighbors
- Overlap percentage for each sample
- Logged as `neighborhood_preservation_ratio` in W&B

---

## Model Artifacts

Artifacts saved for each experiment:

- `msp_tsne_model.pth`: Trained neural network state dict
- `model_config.pt`: Model configuration parameters
- Uploaded to W&B via `wandb.save()`

---

## Acknowledgements

- This repository was originally forked from [Francesco Crecchi's implementation](https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE).
  - The migrated code from the [original forked repository](https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE) to TensorFlow 2.X can be found in the `tf_2x` branch.

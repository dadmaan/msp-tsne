"""
Utility functions for MSP t-SNE project.

This module contains common utility functions used across multiple scripts
for evaluation metrics, model saving, configuration handling, and other
shared functionality.
"""

import os
import time
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score


def calculate_neighborhood_preservation(X_original, X_embedded, k=12):
    """
    Calculate neighborhood preservation ratio between original and embedded spaces.

    This metric measures how well the k-nearest neighbors in the original
    high-dimensional space are preserved in the embedded low-dimensional space.

    Args:
        X_original (np.ndarray): Original high-dimensional data
        X_embedded (np.ndarray): Embedded low-dimensional data
        k (int): Number of neighbors to consider (default: 12)

    Returns:
        float: Neighborhood preservation ratio (0-1, higher is better)
    """
    n_samples = X_original.shape[0]

    # Find k-nearest neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_original)
    _, indices_original = nbrs_original.kneighbors(X_original)

    # Find k-nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_embedded)
    _, indices_embedded = nbrs_embedded.kneighbors(X_embedded)

    # Calculate preservation ratio
    preservation_scores = []
    for i in range(n_samples):
        # Get neighbors (exclude self which is always first)
        neighbors_original = set(indices_original[i][1:])
        neighbors_embedded = set(indices_embedded[i][1:])

        # Calculate overlap
        overlap = len(neighbors_original.intersection(neighbors_embedded))
        preservation_scores.append(overlap / k)

    return np.mean(preservation_scores)


def calculate_evaluation_metrics(X_original, X_embedded, y=None, model=None, k=12):
    """
    Calculate comprehensive evaluation metrics for dimensionality reduction embeddings.

    Args:
        X_original (np.ndarray): Original high-dimensional data
        X_embedded (np.ndarray): Embedded low-dimensional data
        y (np.ndarray, optional): Labels for supervised metrics
        model (object, optional): Fitted model to extract loss from
        k (int): Number of neighbors for neighborhood-based metrics

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {}

    # Calculate trustworthiness score
    trust_score = trustworthiness(X_original, X_embedded, n_neighbors=k)
    metrics['trustworthiness_score'] = trust_score

    # Calculate neighborhood preservation
    neighborhood_preservation = calculate_neighborhood_preservation(X_original, X_embedded, k=k)
    metrics['neighborhood_preservation_ratio'] = neighborhood_preservation

    # Calculate silhouette score if labels are available
    if y is not None:
        silhouette = silhouette_score(X_embedded, y)
        metrics['silhouette_score'] = silhouette

    # Extract final loss from model if available
    if model is not None:
        final_loss = getattr(model, '_final_loss', None)
        if final_loss is not None:
            metrics['final_kl_loss'] = final_loss

    return metrics


def create_model_config(model, training_time=None, config=None):
    """
    Create standardized model configuration dictionary for saving.

    Args:
        model: Trained MultiscaleParametricTSNE model
        training_time (float, optional): Training time in seconds
        config (dict, optional): Full configuration dictionary

    Returns:
        dict: Standardized model configuration
    """
    model_config = {
        'model_type': 'MultiscaleParametricTSNE',
        'n_components': model.n_components,
        'nl1': model.nl1,
        'nl2': model.nl2,
        'nl3': model.nl3,
        'alpha': model.alpha,
        'lr': model.lr,
        'n_iter': model.n_iter,
        'batch_size': model.batch_size,
        'early_exaggeration_epochs': model.early_exaggeration_epochs,
        'early_exaggeration_value': model.early_exaggeration_value,
        'early_stopping_epochs': model.early_stopping_epochs,
        'early_stopping_min_improvement': model.early_stopping_min_improvement,
        'device': str(model.device),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if training_time is not None:
        model_config['training_time_seconds'] = training_time

    if config is not None:
        model_config['config'] = config

    return model_config


def save_model_artifacts(model, config, X_transformed, fitted_scaler, label_encoder,
                        preprocessing_info, training_time, metrics, output_config):
    """
    Save trained model and all artifacts.

    Args:
        model: Trained MultiscaleParametricTSNE model
        config (dict): Full configuration dictionary
        X_transformed (np.ndarray): Transformed embeddings
        fitted_scaler: Fitted scaler object
        label_encoder: Fitted label encoder (or None)
        preprocessing_info (dict): Preprocessing information dictionary
        training_time (float): Training time in seconds
        metrics (dict): Dictionary of evaluation metrics
        output_config (dict): Output configuration section
    """
    model_dir = Path(output_config['model_dir'])
    model_name = output_config['model_name']

    # Create output directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    model_path = model_dir / f"{model_name}.pth"
    if model_path.exists() and not output_config.get('overwrite_existing', False):
        raise FileExistsError(f"Model already exists at {model_path}. Set overwrite_existing=true to overwrite.")

    print(f"Saving model artifacts to {model_dir}")

    # Save PyTorch model state dict
    torch.save(model._model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")

    # Save model configuration using utility function
    model_config = create_model_config(model, training_time, config)
    config_path = model_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    print(f"✓ Model config saved to {config_path}")

    # Save preprocessing artifacts
    if output_config.get('save_preprocessing', True):
        preprocessing_path = model_dir / f"{model_name}_preprocessing.pth"
        torch.save({
            'scaler': fitted_scaler,
            'label_encoder': label_encoder,
            'preprocessing_info': preprocessing_info
        }, preprocessing_path)
        print(f"✓ Preprocessing artifacts saved to {preprocessing_path}")

    # Save embeddings
    if output_config.get('save_embeddings', True):
        embeddings_path = model_dir / f"{model_name}_embeddings.npy"
        np.save(embeddings_path, X_transformed)
        print(f"✓ Embeddings saved to {embeddings_path}")

    # Save metrics
    if output_config.get('save_metrics', True):
        metrics_path = model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✓ Metrics saved to {metrics_path}")

    print(f"\nModel artifacts saved successfully to {model_dir}")


def load_and_validate_config(config_path, required_sections=None):
    """
    Load and validate configuration from YAML file.

    Args:
        config_path (str): Path to YAML configuration file
        required_sections (list, optional): List of required top-level sections

    Returns:
        dict: Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections are missing
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections if specified
    if required_sections:
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config file")

    return config


def create_temp_model_dir(base_name="temp_model_artifacts"):
    """
    Create temporary directory for model artifacts.

    Args:
        base_name (str): Base name for temporary directory

    Returns:
        str: Path to created temporary directory
    """
    model_dir = base_name
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def cleanup_temp_files(*file_paths):
    """
    Clean up temporary files and directories.

    Args:
        *file_paths: Variable number of file/directory paths to clean up
    """
    for path in file_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)
        except OSError:
            pass  # Directory might not be empty or files might be locked
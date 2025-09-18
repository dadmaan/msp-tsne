#!/usr/bin/env python3
"""
Training Multiscale Parametric t-SNE using sklearn pipeline
"""
import sys
import time
import json
import yaml
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.datasets import load_digits

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from msp_tsne import MultiscaleParametricTSNE
from msp_tsne.data_loader import load_data
from msp_tsne.utils import calculate_evaluation_metrics


class MSPTSNEWrapper(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper for MultiscaleParametricTSNE."""

    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Fit the MSP t-SNE model."""
        self.model = MultiscaleParametricTSNE(**self.kwargs)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        """Transform data using fitted model."""
        if self.model is None:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


def _load_data(data_config):
    """Load data - using the improved data_loader."""
    source = data_config.get('source', 'sklearn_digits')

    if source == 'sklearn_digits':
        X, y = load_digits(return_X_y=True)
        return X, y
    elif source == 'file':
        # Use the improved load_data function
        return load_data(data_config)
    else:
        raise ValueError(f"Unsupported data source: {source}. Use 'sklearn_digits' or 'file'")


def create_pipeline(config):
    """Create sklearn pipeline with MSP t-SNE."""
    # Preprocessing
    scaler_type = config['preprocessing'].get('scaler', 'StandardScaler')
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # MSP t-SNE with algorithm config
    msp_tsne = MSPTSNEWrapper(**config['algorithm'])

    # Build pipeline
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('msp_tsne', msp_tsne))

    return Pipeline(steps)


def calculate_metrics(X_original, X_embedded, y=None, pipeline=None):
    """Calculate evaluation metrics using utility function."""
    # Extract model from pipeline if available
    model = None
    if pipeline is not None:
        try:
            msp_tsne_step = pipeline.named_steps.get('msp_tsne')
            if msp_tsne_step and hasattr(msp_tsne_step, 'model'):
                model = msp_tsne_step.model
        except:
            pass

    # Use utility function for comprehensive metrics
    metrics = calculate_evaluation_metrics(X_original, X_embedded, y, model, k=12)

    # Rename trustworthiness_score to trustworthiness for backward compatibility
    if 'trustworthiness_score' in metrics:
        metrics['trustworthiness'] = metrics.pop('trustworthiness_score')

    # Rename final_kl_loss to final_loss for backward compatibility
    if 'final_kl_loss' in metrics:
        metrics['final_loss'] = metrics.pop('final_kl_loss')

    return metrics


def save_results(pipeline, X_embedded, metrics, config):
    """Save results in format."""
    output_config = config['output']

    # Create output directory
    model_dir = Path(output_config['model_path']).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    joblib.dump(pipeline, output_config['model_path'])
    print(f"✓ Pipeline saved to {output_config['model_path']}")

    # Save embeddings
    np.save(output_config['embeddings_path'], X_embedded)
    print(f"✓ Embeddings saved to {output_config['embeddings_path']}")

    # Save metrics
    results = {
        'metrics': metrics,
        'embedding_shape': X_embedded.shape,
        'config': config,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_config['metrics_path'], 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Metrics saved to {output_config['metrics_path']}")


def main():
    parser = argparse.ArgumentParser(description='MSP t-SNE with sklearn pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')

    args = parser.parse_args()

    print("=== MSP t-SNE Pipeline ===")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    print("Loading data...")
    X, y = _load_data(config['data'])
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline(config)
    print("Pipeline steps:", [name for name, _ in pipeline.steps])

    # Train
    print("Training MSP t-SNE...")
    start_time = time.time()
    X_embedded = pipeline.fit_transform(X)
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds")

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(X, X_embedded, y, pipeline)
    print("✓ Metrics calculated:")
    print(f"  - Trustworthiness: {metrics['trustworthiness']:.4f}")
    if 'silhouette_score' in metrics:
        print(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}")
    if 'final_loss' in metrics:
        print(f"  - Final Loss (KL-Divergence): {metrics['final_loss']:.6f}")

    # Save results
    print("Saving results...")
    save_results(pipeline, X_embedded, metrics, config)

    print("\n=== Done ===")
    print(f"Pipeline: {config['output']['model_path']}")
    print(f"Embeddings: {config['output']['embeddings_path']}")


if __name__ == "__main__":
    main()
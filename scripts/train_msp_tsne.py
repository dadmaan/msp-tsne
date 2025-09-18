#!/usr/bin/env python3
"""
Training script for Multiscale Parametric t-SNE
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path
import torch

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from msp_tsne import MultiscaleParametricTSNE
from msp_tsne.data_loader import load_data
from msp_tsne.preprocess import preprocess, get_preprocessing_info
from msp_tsne.utils import (
    calculate_evaluation_metrics,
    save_model_artifacts,
    load_and_validate_config
)


def main():
    parser = argparse.ArgumentParser(description='Train MSP t-SNE model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the YAML configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=== MSP t-SNE Training Script ===")
    print(f"Configuration: {args.config}")
    print(f"Random seed: {args.seed}")
    print()

    # Load configuration
    try:
        config = load_and_validate_config(args.config, ['data', 'preprocessing', 'model', 'output'])
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)

    # Load data
    print("\n--- Loading Data ---")
    try:
        data_config = config['data']
        X, y = load_data(data_config)
        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        if y is not None:
            print(f"✓ Labels loaded: {len(np.unique(y))} classes")
        else:
            print("ℹ No labels provided (unsupervised mode)")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)

    # Preprocess data
    print("\n--- Preprocessing Data ---")
    try:
        preprocessing_config = config['preprocessing']
        X_scaled, y_encoded, fitted_scaler, label_encoder = preprocess(X, y, preprocessing_config)
        preprocessing_info = get_preprocessing_info(fitted_scaler, label_encoder)
        print("✓ Data preprocessing completed")
        print(f"  - Scaler: {preprocessing_info['scaler_type']}")
        print(f"  - Features scaled to shape: {X_scaled.shape}")
        if y_encoded is not None:
            print(f"  - Labels encoded: {preprocessing_info['label_encoder']['n_classes']} classes")
    except Exception as e:
        print(f"✗ Failed to preprocess data: {e}")
        sys.exit(1)

    # Create and train model
    print("\n--- Training MSP t-SNE Model ---")
    try:
        model_config = config['model']
        model = MultiscaleParametricTSNE(**model_config)
        print("✓ Model created with configuration:")
        print(f"  - Architecture: {model.nl1} -> {model.nl2} -> {model.nl3} -> {model.n_components}")
        print(f"  - Training: {model.n_iter} iterations, batch_size={model.batch_size}")
        print(f"  - Device: {model.device}")

        # Train the model
        start_time = time.time()
        X_transformed = model.fit_transform(X_scaled)
        training_time = time.time() - start_time

        print(f"✓ Training completed in {training_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Failed to train model: {e}")
        sys.exit(1)

    # Calculate evaluation metrics
    print("\n--- Calculating Evaluation Metrics ---")
    try:
        # Calculate comprehensive metrics using utility function
        metrics = calculate_evaluation_metrics(X_scaled, X_transformed, y_encoded, model, k=12)

        # Add additional metadata
        metrics.update({
            'training_time_seconds': training_time,
            'n_samples': X_scaled.shape[0],
            'n_features': X_scaled.shape[1],
            'n_components': X_transformed.shape[1]
        })

        # Print metrics
        print(f"✓ Trustworthiness: {metrics['trustworthiness_score']:.4f}")
        print(f"✓ Neighborhood Preservation: {metrics['neighborhood_preservation_ratio']:.4f}")
        if 'silhouette_score' in metrics:
            print(f"✓ Silhouette Score: {metrics['silhouette_score']:.4f}")
        if 'final_kl_loss' in metrics:
            print(f"✓ Final KL Loss: {metrics['final_kl_loss']:.6f}")

        print("✓ Evaluation metrics calculated")

    except Exception as e:
        print(f"✗ Failed to calculate metrics: {e}")
        metrics = {'error': str(e)}

    # Save model and artifacts
    print("\n--- Saving Model Artifacts ---")
    try:
        output_config = config['output']
        save_model_artifacts(
            model=model,
            config=config,
            X_transformed=X_transformed,
            fitted_scaler=fitted_scaler,
            label_encoder=label_encoder,
            preprocessing_info=preprocessing_info,
            training_time=training_time,
            metrics=metrics,
            output_config=output_config
        )
    except Exception as e:
        print(f"✗ Failed to save model artifacts: {e}")
        sys.exit(1)

    print("\n=== Training Completed Successfully ===")
    print(f"Model saved to: {Path(config['output']['model_dir']) / config['output']['model_name']}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Trustworthiness: {metrics.get('trustworthiness_score', 'N/A')}")
    print(f"Neighborhood Preservation: {metrics.get('neighborhood_preservation_ratio', 'N/A')}")
    if 'silhouette_score' in metrics:
        print(f"Silhouette Score: {metrics['silhouette_score']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Hyperparameter tuning with wandb sweeps
"""
import wandb
from msp_tsne import MultiscaleParametricTSNE
from msp_tsne.data_loader import load_data
from msp_tsne.preprocess import preprocess
from msp_tsne.utils import (
    calculate_neighborhood_preservation,
    calculate_evaluation_metrics,
    create_model_config,
    create_temp_model_dir,
    cleanup_temp_files
)
import time
import yaml
import argparse
import sys
import torch
import os


def run_experiment():
    # Use a 'with' statement for the run to ensure wandb.finish() is called
    with wandb.init() as run:
        config = run.config

        # Load data using new data loader (fallback to load_digits if no path specified)
        data_config = {'features': None, 'labels': None, 'label_column': None, 'format': 'auto'}
        X, y = load_data(data_config)

        # Preprocess data using new preprocessing pipeline
        X_scaled, y_encoded, fitted_scaler, label_encoder = preprocess(X, y, config)

        # Create the model, casting hyperparameters from the sweep to their correct types
        model = MultiscaleParametricTSNE(
            n_components=2,
            n_iter=int(config.n_iter),
            batch_size=int(config.batch_size),
            early_exaggeration_epochs=int(config.early_exaggeration_epochs),
            early_exaggeration_value=config.early_exaggeration_value,
            early_stopping_epochs=int(config.early_stopping_epochs),
            early_stopping_min_improvement=config.early_stopping_min_improvement,
            alpha=config.alpha,
            lr=config.lr,
            nl1=int(config.nl1),
            nl2=int(config.nl2),
            nl3=int(config.nl3),
            logdir=None,
            verbose=1
        )

        # Track training time
        start_time = time.time()

        # Fit the model and transform the data (no pipeline needed, data already preprocessed)
        X_transformed = model.fit_transform(X_scaled)

        training_time = time.time() - start_time

        # Calculate evaluation metrics using utility function
        metrics = calculate_evaluation_metrics(X_scaled, X_transformed, y, model, k=12)
        metrics['training_time'] = training_time

        # Log metrics to W&B. Hyperparameters are logged automatically via config.
        wandb.log(metrics)

        # Save model artifacts for W&B
        model_dir = create_temp_model_dir()

        save_model = config.get('saving', {}).get('save_model', False)

        if save_model:
            # Save the trained neural network model
            model_path = os.path.join(model_dir, "msp_tsne_model.pth")
            torch.save(model._model.state_dict(), model_path)
            wandb.save(model_path)
        else:
            model_path = None

        # Save the model configuration using utility function
        config_path = os.path.join(model_dir, "model_config.json")
        model_config = create_model_config(model, training_time, config)
        with open(config_path, 'w') as f:
            import json
            json.dump(model_config, f, indent=2, default=str)
        wandb.save(config_path)

        # Clean up temporary files
        if save_model:
            cleanup_temp_files(model_path, config_path, model_dir)
        else:
            cleanup_temp_files(config_path, model_dir)

def smoke_test(config):
    """
    Runs a minimal version of the experiment to verify the pipeline.
    """
    print("--- Running Smoke Test ---")

    smoke_config = config['modes']['smoke_test']

    # Use a very small subset of the data
    data_config = {'features': None, 'labels': None, 'label_column': None, 'format': 'auto'}
    X_mnist, y_mnist = load_data(data_config)
    sample_size = smoke_config['sample_size']
    X_smoke, y_smoke = X_mnist[:sample_size], y_mnist[:sample_size]

    # Preprocess data using new preprocessing pipeline with StandardScaler for smoke test
    smoke_preprocess_config = {'scaler': 'StandardScaler'}
    X_scaled, y_encoded, fitted_scaler, label_encoder = preprocess(X_smoke, y_smoke, smoke_preprocess_config)

    # Use a fixed, minimal set of hyperparameters from config
    model = MultiscaleParametricTSNE(
        n_components=2,
        n_iter=smoke_config['iterations'],
        batch_size=smoke_config['batch_size'],
        early_exaggeration_epochs=smoke_config['early_exaggeration_epochs'],
        verbose=smoke_config['verbose']
    )

    try:
        # Check if the model can fit and transform without errors (no pipeline needed)
        start_time = time.time()
        X_transformed = model.fit_transform(X_scaled)
        end_time = time.time()

        # Simple assertions to check output
        assert X_transformed.shape == (sample_size, 2), f"Output shape is incorrect, expected ({sample_size}, 2)"

        print(f"Smoke test completed in {end_time - start_time:.2f} seconds.")
        print("Smoke test PASSED!")

    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        # Exit with a non-zero code to indicate failure in a CI/CD environment
        sys.exit(1)



# Main execution block to initialize sweep and run agent
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSP t-SNE Hyperparameter Tuning')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the YAML config file.')
    parser.add_argument('--mode', type=str, choices=['sweep', 'smoke_test'],
                       default='sweep', help='Mode to run: sweep or smoke_test')
    parser.add_argument('--count', type=int,
                       help='Number of sweep runs (overrides config file)')

    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == 'smoke_test':
        # Run smoke test
        if not config['modes']['smoke_test']['enabled']:
            print("Smoke test is disabled in configuration.")
            sys.exit(1)
        smoke_test(config)

    elif args.mode == 'sweep':
        # Run wandb sweep
        if not config['modes']['sweep']['enabled']:
            print("Sweep mode is disabled in configuration.")
            sys.exit(1)

        # Use sweep count from args or config
        sweep_count = args.count if args.count is not None else config['sweep_count']

        # Initialize sweep with config from YAML
        sweep_id = wandb.sweep(config['sweep_config'], project=config['project'])
        wandb.agent(sweep_id, function=run_experiment, count=sweep_count)

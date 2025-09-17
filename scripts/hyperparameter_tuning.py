import wandb
from msp_tsne import MultiscaleParametricTSNE
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
import time
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import yaml
import argparse
import sys
import torch
import os

def calculate_neighborhood_preservation(X_original, X_embedded, k=12):
    """
    Calculate neighborhood preservation ratio between original and embedded spaces.

    Args:
        X_original: Original high-dimensional data
        X_embedded: Embedded low-dimensional data
        k: Number of neighbors to consider

    Returns:
        Float: Neighborhood preservation ratio (0-1, higher is better)
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

def run_experiment():
    # Use a 'with' statement for the run to ensure wandb.finish() is called
    with wandb.init() as run:
        config = run.config

        # Load data
        X_mnist, y_mnist = load_digits(return_X_y=True)

        # Choose scaler based on config
        if config.scaler == "StandardScaler":
            scaler = StandardScaler()
        else:  # MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))

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

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('msp-tsne', model)
        ])

        # Track training time
        start_time = time.time()

        # Fit the pipeline and transform the data
        X_transformed = pipeline.fit_transform(X_mnist)

        training_time = time.time() - start_time

        # Retrieve the scaled data from the fitted pipeline for metric calculation
        X_scaled = pipeline.named_steps['scaler'].transform(X_mnist)

        # Calculate evaluation metrics
        trust_score = trustworthiness(X_scaled, X_transformed, n_neighbors=12)
        silhouette = silhouette_score(X_transformed, y_mnist)
        neighborhood_preservation = calculate_neighborhood_preservation(X_scaled, X_transformed, k=12)

        # Get final loss from model (accessing a private attribute can be fragile)
        final_loss = getattr(model, '_final_loss', None)

        # Log metrics to W&B. Hyperparameters are logged automatically via config.
        wandb.log({
            "trustworthiness_score": trust_score,
            "silhouette_score": silhouette,
            "neighborhood_preservation_ratio": neighborhood_preservation,
            "training_time": training_time,
            "final_kl_loss": final_loss
        })

        # Save model artifacts if this is the best performing run based on trustworthiness score
        # Create a temporary directory for model saving
        model_dir = "temp_model_artifacts"
        os.makedirs(model_dir, exist_ok=True)

        # Save the trained neural network model
        model_path = os.path.join(model_dir, "msp_tsne_model.pth")
        torch.save(model._model.state_dict(), model_path)

        # Save model artifacts to W&B
        wandb.save(model_path)

        # Also save the model configuration for reproducibility
        config_path = os.path.join(model_dir, "model_config.pt")
        model_config = {
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
            'device': str(model.device)
        }
        torch.save(model_config, config_path)
        wandb.save(config_path)

        # Clean up temporary files
        try:
            os.remove(model_path)
            os.remove(config_path)
            os.rmdir(model_dir)
        except OSError:
            pass  # Directory might not be empty or files might be locked

def smoke_test(config):
    """
    Runs a minimal version of the experiment to verify the pipeline.
    """
    print("--- Running Smoke Test ---")

    smoke_config = config['modes']['smoke_test']

    # Use a very small subset of the data
    X_mnist, y_mnist = load_digits(return_X_y=True)
    sample_size = smoke_config['sample_size']
    X_smoke, y_smoke = X_mnist[:sample_size], y_mnist[:sample_size]

    scaler = StandardScaler()

    # Use a fixed, minimal set of hyperparameters from config
    model = MultiscaleParametricTSNE(
        n_components=2,
        n_iter=smoke_config['iterations'],
        batch_size=smoke_config['batch_size'],
        early_exaggeration_epochs=smoke_config['early_exaggeration_epochs'],
        verbose=smoke_config['verbose']
    )

    pipeline = Pipeline([
        ('scaler', scaler),
        ('msp-tsne', model)
    ])

    try:
        # Check if the pipeline can fit and transform without errors
        start_time = time.time()
        X_transformed = pipeline.fit_transform(X_smoke)
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

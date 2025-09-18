import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Tuple, Optional, Union
from sklearn.datasets import load_digits


def _detect_format(file_path: Union[str, Path]) -> str:
    """Auto-detect file format from extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        return 'csv'
    elif suffix == '.npy':
        return 'npy'
    elif suffix == '.npz':
        return 'npz'
    elif suffix in ['.pkl', '.pickle']:
        return 'pkl'
    elif suffix == '.json':
        return 'json'
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .npy, .npz, .pkl, .json")


def _load_file(file_path: Union[str, Path], format_type: str) -> Union[np.ndarray, pd.DataFrame]:
    """Load file based on format type."""
    if format_type == 'csv':
        return pd.read_csv(file_path)
    elif format_type == 'npy':
        return np.load(file_path)
    elif format_type == 'npz':
        data = np.load(file_path)
        # Return the first array if multiple arrays in npz
        key = list(data.keys())[0]
        return data[key]
    elif format_type == 'pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            # Array of objects format: [{"col1": val1, "col2": val2}, ...]
            return pd.DataFrame.from_records(data)
        elif isinstance(data, dict):
            # Dictionary format - check for common keys
            if 'data' in data:
                # Matrix format: {"data": [[...], [...]], ...}
                return np.array(data['data'])
            elif 'features' in data:
                # Features format: {"features": [[...], [...]]}
                return np.array(data['features'])
            else:
                # Try to convert dict to DataFrame
                return pd.DataFrame(data)
        else:
            # Direct array format
            return np.array(data)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def _validate_config(data_config: dict) -> None:
    """Validate data configuration for conflicts."""
    labels_path = data_config.get('labels')
    label_column = data_config.get('label_column')

    if labels_path is not None and label_column is not None:
        raise ValueError("Cannot specify both 'labels' path and 'label_column'. Choose one or set both to null.")


def _extract_labels_from_features(features_data: Union[np.ndarray, pd.DataFrame],
                                label_column: str,
                                allow_missing: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[np.ndarray]]:
    """Extract labels from features data and remove label column."""
    if isinstance(features_data, pd.DataFrame):
        if label_column not in features_data.columns:
            if allow_missing:
                print(f"Warning: Column '{label_column}' not found in features. Available columns: {list(features_data.columns)}")
                print("Proceeding without labels...")
                return features_data, None
            else:
                raise ValueError(f"Column '{label_column}' not found in features. Available columns: {list(features_data.columns)}")

        labels = features_data[label_column].values
        features = features_data.drop(columns=[label_column])
        return features, labels

    elif isinstance(features_data, np.ndarray):
        if features_data.ndim != 2:
            raise ValueError(f"Cannot extract column from {features_data.ndim}D array. Expected 2D array.")

        try:
            # Try to interpret label_column as integer index
            col_idx = int(label_column)
            if col_idx >= features_data.shape[1] or col_idx < -features_data.shape[1]:
                if allow_missing:
                    print(f"Warning: Column index {col_idx} out of range for array with {features_data.shape[1]} columns")
                    print("Proceeding without labels...")
                    return features_data, None
                else:
                    raise ValueError(f"Column index {col_idx} out of range for array with {features_data.shape[1]} columns")

            labels = features_data[:, col_idx]
            features = np.delete(features_data, col_idx, axis=1)
            return features, labels

        except ValueError as e:
            if "invalid literal" in str(e):
                if allow_missing:
                    print(f"Warning: For numpy arrays, label_column must be an integer index, got '{label_column}'")
                    print("Proceeding without labels...")
                    return features_data, None
                else:
                    raise ValueError(f"For numpy arrays, label_column must be an integer index, got '{label_column}'")
            raise

    else:
        raise ValueError(f"Unsupported data type for label extraction: {type(features_data)}")


def load_data(data_config: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load data based on configuration.

    Args:
        data_config: Dictionary containing:
            - features: Path to features file or null
            - labels: Path to separate labels file or null
            - label_column: Column name/index to extract from features or null
            - format: File format ('auto', 'csv', 'npy', 'npz', 'pkl', 'json')
            - allow_missing_labels: Whether to allow missing label columns (default: true)

    Returns:
        Tuple of (X, y) where:
            - X: Feature matrix as numpy array
            - y: Labels as numpy array or None if no labels specified
    """
    _validate_config(data_config)

    features_path = data_config.get('features')
    labels_path = data_config.get('labels')
    label_column = data_config.get('label_column')
    format_type = data_config.get('format', 'auto')
    allow_missing_labels = data_config.get('allow_missing_labels', True)

    # Fallback to sklearn digits dataset
    if features_path is None:
        X, y = load_digits(return_X_y=True)
        return X, y

    # Check if features file exists
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # Auto-detect format if needed
    if format_type == 'auto':
        format_type = _detect_format(features_path)

    # Load features
    features_data = _load_file(features_path, format_type)

    # Handle labels
    labels = None

    if label_column is not None:
        # Extract labels from features data
        features_data, labels = _extract_labels_from_features(features_data, label_column, allow_missing=allow_missing_labels)

    elif labels_path is not None:
        # Load separate labels file
        if not Path(labels_path).exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        labels_format = _detect_format(labels_path) if data_config.get('format') == 'auto' else format_type
        labels_data = _load_file(labels_path, labels_format)

        # Convert to numpy array if needed
        if isinstance(labels_data, pd.DataFrame):
            if labels_data.shape[1] == 1:
                labels = labels_data.iloc[:, 0].values
            else:
                raise ValueError(f"Labels file has {labels_data.shape[1]} columns, expected 1")
        else:
            labels = np.asarray(labels_data)
            if labels.ndim > 1:
                labels = labels.flatten()

    # Convert features to numpy array
    if isinstance(features_data, pd.DataFrame):
        X = features_data.values
    else:
        X = np.asarray(features_data)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError(f"Features must be 1D or 2D, got {X.ndim}D array")

    return X, labels
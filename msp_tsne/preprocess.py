import numpy as np
from typing import Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def preprocess(X: np.ndarray,
               y: Optional[np.ndarray],
               config: Union[dict, object],
               feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Optional[np.ndarray], object, Optional[LabelEncoder]]:
    """
    Preprocess features and labels for MSP t-SNE.

    Args:
        X: Feature matrix
        y: Labels array or None for unsupervised mode
        config: Configuration dictionary or WandB config object containing scaler type
        feature_range: Range for MinMaxScaler (default: (0, 1))

    Returns:
        Tuple of (X_scaled, y_encoded_or_none, fitted_scaler, label_encoder_or_none)
            - X_scaled: Scaled feature matrix
            - y_encoded_or_none: Encoded labels or None if y was None
            - fitted_scaler: Fitted scaler object (StandardScaler or MinMaxScaler)
            - label_encoder_or_none: Fitted LabelEncoder or None if y was None
    """
    # Handle both dict and object config formats (WandB vs YAML)
    if hasattr(config, 'scaler'):
        # WandB config object (dot notation)
        scaler_type = config.scaler
    elif isinstance(config, dict):
        # YAML config dictionary
        scaler_type = config.get('scaler', 'StandardScaler')
    else:
        raise ValueError(f"Unsupported config type: {type(config)}. Expected dict or object with 'scaler' attribute.")

    # Handle feature scaling
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Supported: 'StandardScaler', 'MinMaxScaler'")

    # Fit and transform features
    X_scaled = scaler.fit_transform(X)

    # Handle labels
    if y is None:
        return X_scaled, None, scaler, None

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled, y_encoded, scaler, label_encoder


def inverse_transform_features(X_scaled: np.ndarray, scaler: object) -> np.ndarray:
    """
    Inverse transform scaled features back to original scale.

    Args:
        X_scaled: Scaled feature matrix
        scaler: Fitted scaler object

    Returns:
        Original scale feature matrix
    """
    return scaler.inverse_transform(X_scaled)


def inverse_transform_labels(y_encoded: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    """
    Inverse transform encoded labels back to original labels.

    Args:
        y_encoded: Encoded labels
        label_encoder: Fitted LabelEncoder

    Returns:
        Original labels
    """
    return label_encoder.inverse_transform(y_encoded)


def get_preprocessing_info(scaler: object, label_encoder: Optional[LabelEncoder]) -> dict:
    """
    Get information about preprocessing transformations.

    Args:
        scaler: Fitted scaler object
        label_encoder: Fitted LabelEncoder or None

    Returns:
        Dictionary with preprocessing information
    """
    info = {
        'scaler_type': type(scaler).__name__,
        'n_features': getattr(scaler, 'n_features_in_', None),
    }

    if hasattr(scaler, 'mean_'):
        info['feature_mean'] = scaler.mean_
    if hasattr(scaler, 'scale_'):
        info['feature_scale'] = scaler.scale_
    if hasattr(scaler, 'data_min_'):
        info['feature_min'] = scaler.data_min_
    if hasattr(scaler, 'data_max_'):
        info['feature_max'] = scaler.data_max_

    if label_encoder is not None:
        info['label_encoder'] = {
            'classes': label_encoder.classes_,
            'n_classes': len(label_encoder.classes_)
        }
    else:
        info['label_encoder'] = None

    return info
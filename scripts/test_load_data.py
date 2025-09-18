#!/usr/bin/env python3
"""
Test script for the improved load_data function
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from msp_tsne.data_loader import load_data

def test_load_data():
    """Test the improved load_data function with various scenarios."""

    print("=== Testing Improved load_data Function ===\n")

    # Test 1: sklearn_digits (default)
    print("Test 1: sklearn_digits (default)")
    config1 = {'features': None}
    try:
        X, y = load_data(config1)
        print(f"✓ Success: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Labels: {y.shape if y is not None else None}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    print()

    # Test 2: File with missing label column (should work now)
    print("Test 2: File with missing label column")
    config2 = {
        'features': '/home/jovyan/work/data/features.json',
        'label_column': 'label',
        'format': 'auto',
        'allow_missing_labels': True
    }
    try:
        X, y = load_data(config2)
        print(f"✓ Success: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Labels: {y.shape if y is not None else None}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    print()

    # Test 3: File with allow_missing_labels=False (should fail)
    print("Test 3: File with allow_missing_labels=False")
    config3 = {
        'features': '/home/jovyan/work/data/features.json',
        'label_column': 'label',
        'format': 'auto',
        'allow_missing_labels': False
    }
    try:
        X, y = load_data(config3)
        print(f"✓ Success: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Labels: {y.shape if y is not None else None}")
    except Exception as e:
        print(f"✗ Failed (expected): {e}")
    print()

    # Test 4: File without label_column (should work)
    print("Test 4: File without label_column")
    config4 = {
        'features': '/home/jovyan/work/data/features.json',
        'format': 'auto'
    }
    try:
        X, y = load_data(config4)
        print(f"✓ Success: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Labels: {y.shape if y is not None else None}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    print()

    print("=== Test Complete ===")

if __name__ == "__main__":
    test_load_data()
# Hyperparameter Tuning Script Refactor

## Overview
Refactored `scripts/hyperparameter_tuning.py` to use external YAML configuration and enhanced argparser support.

## Changes Made

### 1. Configuration Externalization
- **Created**: `configs/hp_config.yml`
- **Moved**: All wandb sweep configuration from hardcoded dict to YAML
- **Added**: Project settings, mode configurations, and smoke test parameters

### 2. Enhanced Argparser
- **Added**: `--mode` argument (choices: `sweep`, `smoke_test`)
- **Added**: `--count` argument (optional override for sweep count)
- **Enhanced**: `--config` argument integration with full YAML loading

### 3. Smoke Test Integration
- **Updated**: `smoke_test()` function to accept config parameter
- **Integrated**: Smoke test parameters from YAML configuration
- **Added**: Mode-based execution logic in main block

### 4. Script Structure
- **Removed**: Hardcoded `sweep_config` dictionary (~25 lines)
- **Added**: Dynamic config loading and mode switching
- **Added**: Error handling for disabled modes

## Usage Examples

### Run Sweep Mode (default)
```bash
python scripts/hyperparameter_tuning.py --config configs/hp_config.yml
python scripts/hyperparameter_tuning.py --config configs/hp_config.yml --mode sweep --count 25
```

### Run Smoke Test
```bash
python scripts/hyperparameter_tuning.py --config configs/hp_config.yml --mode smoke_test
```

## Benefits
- ✅ External configuration management
- ✅ No code changes needed for parameter tuning
- ✅ Clean separation of concerns
- ✅ Support for multiple execution modes
- ✅ Configurable smoke test parameters
- ✅ Better maintainability and flexibility
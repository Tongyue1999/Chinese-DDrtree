#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script to run VAE-based feature selection analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('code')
from vae_feature_selection import VAEFeatureSelector


def main():
    """
    Run VAE feature selection analysis
    """
    print("VAE Feature Selection Analysis")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load real data
    data_file = Path('data/ahsl_continuous.tsv')  # Adjust path if needed
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure your data file is located at 'data/ahsl_continuous.tsv'")
        return

    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file, sep='\t', index_col=0)
    X_data = data.values
    feature_names = data.columns.tolist()
    print(f"Loaded data: {X_data.shape[0]} samples, {X_data.shape[1]} features")

    print(f"Data shape: {X_data.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Initialize VAE feature selector
    selector = VAEFeatureSelector()

    # Compute feature importance
    print("\nComputing feature importance...")
    results = selector.compute_feature_importance(X_data, feature_names)

    # Print feature rankings
    selector.print_feature_rankings(results, feature_names)

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
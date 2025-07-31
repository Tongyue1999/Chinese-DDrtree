#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE-based Feature Selection for Clinical Data Analysis

This script implements a feature selection pipeline using Variational
Autoencoders (VAE) for clinical data analysis.

Key Features:
- Single VAE model training with optimized architecture
- Weight-based feature importance using encoder weights
- Perturbation-based feature importance using latent space analysis
- Feature importance ranking output

Architecture (optimized through systematic analysis):
- Single hidden layer with 512 units
- Latent dimension: 32
- Dropout: 0.1
- Beta parameter: 1e-4
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')
import time
import json
from sklearn.preprocessing import StandardScaler


class VAEModel(nn.Module):
    """
    Variational Autoencoder for feature selection.

    Architecture based on hyperparameter optimization results:
    - 1 hidden layer with 512 units
    - Latent dimension: 32
    - Dropout: 0.1
    - Beta parameter: 1e-4
    """

    def __init__(self, input_dim, hidden_dim=512, latent_dim=32, dropout=0.1):
        super().__init__()
        # Encoder: input -> hidden(512) -> latent(32)
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent(32) -> hidden(512) -> output
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.encoder_hidden(x))
        h = self.encoder_dropout(h)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.decoder_hidden(z))
        h = self.decoder_dropout(h)
        return self.decoder_output(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class VAEFeatureSelector:
    """
    VAE-based feature selection with perturbation analysis and weight-based importance.
    """

    def __init__(self):
        self.vae_model = None
        self.scaler = None
        self.device = None

    def train_vae(self, X_data, epochs=150, lr=0.0001, beta=1e-4, batch_size=1024):
        """
        Train VAE model for feature importance analysis.

        Parameters:
        -----------
        X_data : np.ndarray
            Input data matrix (samples x features)
        epochs : int
            Training epochs (default: 150)
        lr : float
            Learning rate (default: 0.0001)
        beta : float
            Beta parameter for beta-VAE (default: 1e-4)
        batch_size : int
            Batch size for training (default: 1024)

        Returns:
        --------
        VAEModel : Trained VAE model
        """
        print("Training VAE model for feature importance analysis:")
        print("   - Architecture: 1 layer × 512 hidden units")
        print(f"   - Latent dim: 32, Dropout: 0.1, Beta: {beta}")
        print(f"   - Learning rate: {lr}, Batch size: {batch_size}")
        print(f"   - Training {epochs} epochs")

        # Check CUDA availability and set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - Using device: {device}")
        if torch.cuda.is_available():
            print(f"   - GPU: {torch.cuda.get_device_name()}")
            print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Data standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        self.scaler = scaler
        self.device = device

        input_dim = X_scaled.shape[1]

        # Create and train single VAE model
        torch.manual_seed(42)
        np.random.seed(42)

        self.vae_model = VAEModel(input_dim, hidden_dim=512, latent_dim=32, dropout=0.1).to(device)
        optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=lr)

        # Convert data and move to GPU
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # VAE loss function
        def vae_loss(recon_x, x, mu, logvar, beta=beta):
            recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
            kl_loss = -0.5 * torch.sum(
                1 + torch.clamp(logvar, -10, 10) - mu.pow(2) - torch.clamp(logvar, -10, 10).exp())

            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                recon_loss = torch.tensor(0.0, device=device)
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.0, device=device)

            return recon_loss + beta * kl_loss, recon_loss, kl_loss

        # Training loop
        self.vae_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                optimizer.zero_grad()
                recon_batch, mu, logvar, z = self.vae_model(data)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            # Print loss at key epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                avg_loss = total_loss / len(X_scaled)
                print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f}")

        self.vae_model.eval()
        print("VAE training completed!")
        return self.vae_model

    def compute_weight_based_importance(self, feature_names):
        """
        Compute weight-based feature importance using encoder weights.

        Parameters:
        -----------
        feature_names : list
            List of feature names

        Returns:
        --------
        np.ndarray : Feature importance scores based on encoder weights
        """
        if self.vae_model is None:
            raise ValueError("No trained model found. Run train_vae first.")

        print("Computing weight-based feature importance...")

        # Extract encoder weights (input -> hidden layer)
        encoder_weights = self.vae_model.encoder_hidden.weight.data.cpu().numpy()

        # Calculate feature importance as L1 norm (sum of absolute weights)
        feature_importance = np.sum(np.abs(encoder_weights), axis=0)

        print("Weight-based importance computed!")
        return feature_importance

    def compute_perturbation_importance(self, X_data, feature_names):
        """
        Compute perturbation-based feature importance using latent space analysis.

        Parameters:
        -----------
        X_data : np.ndarray
            Input data matrix
        feature_names : list
            List of feature names

        Returns:
        --------
        np.ndarray : Feature importance scores based on latent space perturbation
        """
        if self.vae_model is None:
            raise ValueError("No trained model found. Run train_vae first.")

        print("Computing perturbation-based feature importance...")

        # Use standardized data
        X_scaled = self.scaler.transform(X_data)

        def compute_latent_representation(x):
            """Compute latent representation (mu)"""
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                mu, _ = self.vae_model.encode(x_tensor)
            return mu.cpu().numpy()

        # Compute original latent representation
        original_latent = compute_latent_representation(X_scaled)  # shape: (samples, 32)

        # Compute perturbation importance for each feature
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        feature_importance = np.zeros(n_features)

        print(f"   - Analyzing {n_features} features on ALL {n_samples} samples...")
        print(f"   - Using full dataset for highly reliable statistics")

        for feature_idx in range(n_features):
            # Step 1: Replace feature values with missing (mean=0 for standardized data)
            X_perturbed = X_scaled.copy()
            X_perturbed[:, feature_idx] = 0.0  # Missing value = mean = 0 in standardized space

            # Step 2: Re-encode the modified data through the trained encoder
            perturbed_latent = compute_latent_representation(X_perturbed)

            # Step 3: Compute change in latent representation
            latent_change = perturbed_latent - original_latent  # shape: (samples, 32)

            # Step 4: Calculate change for each latent dimension
            # First take mean across samples for each latent dimension
            dim_changes = np.mean(np.abs(latent_change), axis=0)  # shape: (32,) - mean change per dimension

            # Then sum across all latent dimensions to get feature importance
            feature_importance[feature_idx] = np.sum(dim_changes)  # sum over latent dimensions

        print("Perturbation-based importance computed!")
        return feature_importance

    def compute_feature_importance(self, X_data, feature_names):
        """
        Compute feature importance using weight and perturbation analysis.

        Parameters:
        -----------
        X_data : np.ndarray
            Input data matrix
        feature_names : list
            List of feature names

        Returns:
        --------
        dict : Dictionary containing importance results
        """
        print("Starting feature importance computation...")

        # Train single VAE model
        self.train_vae(X_data)

        # Compute weight-based importance
        weight_importance = self.compute_weight_based_importance(feature_names)

        # Compute perturbation-based importance
        perturbation_importance = self.compute_perturbation_importance(X_data, feature_names)

        # Get all features for visualization (no selection needed)
        all_indices = np.arange(len(feature_names))

        results = {
            'all_features': feature_names,
            'all_indices': all_indices,
            'weight_importance': weight_importance,
            'perturbation_importance': perturbation_importance,
            'X_data': X_data  # Store for analysis
        }

        print("Feature importance computation completed!")
        return results

    def print_feature_rankings(self, results, feature_names):
        """
        Print feature importance rankings.

        Parameters:
        -----------
        results : dict
            Results from feature_selection
        feature_names : list
            List of all feature names
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE RANKINGS")
        print("=" * 60)

        # Weight-based ranking
        weight_importance = results['weight_importance']
        weight_top_indices = np.argsort(weight_importance)[::-1][:15]

        print("\n1. WEIGHT-BASED FEATURE IMPORTANCE (Top 15):")
        print("-" * 50)
        for i, idx in enumerate(weight_top_indices):
            feature_name = feature_names[idx]
            importance = weight_importance[idx]
            print(f"{i + 1:2d}. {feature_name:<15} : {importance:.6f}")

        # Perturbation-based ranking
        perturbation_importance = results['perturbation_importance']
        perturbation_top_indices = np.argsort(perturbation_importance)[::-1][:15]

        print("\n2. PERTURBATION-BASED FEATURE IMPORTANCE (Top 15):")
        print("-" * 50)
        for i, idx in enumerate(perturbation_top_indices):
            feature_name = feature_names[idx]
            importance = perturbation_importance[idx]
            print(f"{i + 1:2d}. {feature_name:<15} : {importance:.6f}")

        print("\n" + "=" * 60)


def main():
    """
    Main execution function for VAE feature selection pipeline.
    """
    print("VAE Feature Selection Pipeline")
    print("=" * 50)

    # Load real clinical data
    data_path = '../data/ahsl_continuous.tsv'
    print(f"Loading data from: {data_path}")

    # Read TSV file
    df = pd.read_csv(data_path, sep='\t')
    print(f"Data shape: {df.shape}")

    # Extract features (excluding sample_id column)
    feature_names = df.columns[1:].tolist()  # Skip first column (sample_id)
    X_data = df.iloc[:, 1:].values  # All rows, all columns except first

    print(f"Number of samples: {X_data.shape[0]}")
    print(f"Number of features: {X_data.shape[1]}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10 features

    # Initialize feature selector
    selector = VAEFeatureSelector()

    # Run feature selection
    results = selector.compute_feature_importance(X_data, feature_names)

    # Print feature rankings
    selector.print_feature_rankings(results, feature_names)

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
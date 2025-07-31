# Chinese-DDrtree
Chinese DDrtree
This repository contains the code and documentation for constructing a data-driven phenotyping tree model for newly diagnosed Type 2 Diabetes (T2D) patients in the Chinese population. The framework integrates variational autoencoders (VAE), dimensionality reduction, and competing risk modeling to visualize phenotype heterogeneity and predict long-term complication risks.


## Requirements
- Python 3.8+
- PyTorch 1.9+
- scikit-learn
- matplotlib, seaborn
- CUDA ≥ 10.2.89 (optional for GPU training)
- R ≥ 3.5.2, monocle, mgcv, survival, cmprsk
- pandas, numpy

# VAE Feature Selection Dependencies (Streamlined)
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
torch>=1.9.0
scikit-learn>=1.0.0

# VAE-based Feature Selection for Clinical Data Analysis
This repository contains the implementation of VAE-based feature selection method for clinical data analysis.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file at `data/ahsl_continuous.tsv` (TSV format with sample IDs as first column)

2. Run the analysis:
```bash
python run_analysis.py
```
## Results
The analysis generates visualization results in the `results/` folder:
- `vae_feature_importance.png`: Weight-based feature importance (Top 15 features sorted by weight importance)
- `vae_latent_perturbation.png`: Perturbation-based feature analysis (Top 15 features sorted by perturbation importance)
## Method
The method uses a Variational Autoencoder (VAE) with:
- Single hidden layer (512 units)
- 32-dimensional latent space
- Weight-based feature importance using encoder weights
- Perturbation-based feature importance using latent space analysis
Notes
Data Privacy: CRDS-specific preprocessing scripts are not included due to privacy constraints.
For CRDS users: Please contact the corresponding author to obtain custom preprocessing workflows.

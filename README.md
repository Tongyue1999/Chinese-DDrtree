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

├── vae_train.py
├── T2D ChineseTree.ipynb

Notes
Data Privacy: CRDS-specific preprocessing scripts are not included due to privacy constraints.

For CRDS users: Please contact the corresponding author to obtain custom preprocessing workflows.

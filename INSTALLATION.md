# DCG Project Installation Guide

## Quick Start (CPU)
```bash
pip install -r requirements.txt
```

## GPU Installation (CUDA 11.8)
For GPU support with CUDA 11.8, install torch before running requirements.txt:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-base.txt
```

## GPU Installation (CUDA 12.1)
For GPU support with CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-base.txt
```

## Using Conda (Recommended)
If you prefer conda, use the existing environment file:
```bash
conda env create -f DCG/environment.cpu.yml
```

For GPU support, modify `environment.cpu.yml` and replace the torch line in pip dependencies:
```yaml
- pip:
    - --index-url https://download.pytorch.org/whl/cu118
    - torch
    - torchvision
    - torchaudio
    - munkres
```

## Project Dependencies

**Core Scientific Computing:**
- numpy: Numerical computing
- scipy: Scientific computing utilities
- scikit-learn: Machine learning preprocessing and clustering metrics

**Deep Learning:**
- torch: PyTorch tensor computation and neural networks
- torchvision: Computer vision utilities
- torchaudio: Audio processing utilities

**Clustering:**
- munkres: Hungarian algorithm for optimal assignment

**Optional (for Jupyter notebooks):**
- jupyter, ipython, matplotlib (if using dcg_colab_t4.ipynb)

## Verification

To verify installation:
```bash
cd DCG
python smoke_test.py
```

This should complete without errors and display clustering metrics.

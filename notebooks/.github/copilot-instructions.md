# Copilot Instructions

## Context
I am working on a master's thesis applying deep learning to classify Parkinson’s Disease (PD) vs. Multiple System Atrophy (MSA) using fluorescence microscopy images of dermal sweat gland biopsies. Images are 3D with three RGB biomarker channels, projected to 2D via Maximum Intensity Projection (MIP). The dataset is small (~50 images per class). 
Suggest the simplest/fastest to implement solution possible but not compromising on performance expectation.

## Used Libraries
- **Deep Learning**: MONAI, PyTorch, Scikit-learn, ecc ..
- **Experiment Tracking**: mlflow

## Programming Standards
- Follow **PEP 8** for clean code.
- Use **modular, OOP design** with SOLID principles when using OOP.
- Provide **docstrings** and some **inline comments** for clarity.

## Expected Output Format
- **Clear markdown explanations** in Notebooks.
- Reproducible experiments with logged parameters and models.
# Machine Learning Pipelines Architecture Guide — Scikit-Learn

A practical reference guide and architectural template for creating reproducible, clean, and leak-free machine learning pipelines with Scikit-Learn.

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lazer999/ml-pipelines-simplified-for-everyone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Field](https://img.shields.io/badge/Field-ML%20Engineering%20/%20Educational-brightgreen)](#)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Highlights & Results](#key-highlights--results)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [Dataset Details](#dataset-details)
- [Author & Acknowledgments](#author--acknowledgments)

---

## Project Overview

This repository provides the complete, production-structured implementation of the **[Machine Learning Pipelines Architecture Guide — Scikit-Learn](https://www.kaggle.com/code/lazer999/ml-pipelines-simplified-for-everyone)** project originally published on Kaggle. 

The primary focus of this work is translating complex data into actionable machine learning solutions using disciplined data engineering, rigorous validation strategies, and clean, leak-free preprocessing pipelines.

---

## Key Highlights & Results

- Step-by-step tutorial on structuring machine learning code into modular, production-grade components.
- Full implementation of `ColumnTransformer` with distinct numeric and categorical transformers.
- Eliminated data leakage across train/validation splits by nesting imputation inside pipeline stages.
- Clean classification modeling with Logistic Regression, complete with evaluation metrics and confusion matrices.

---

## System Architecture & Workflow

The pipeline follows a structured, modular execution path:

```mermaid
flowchart TD
    A[Raw Input Data] --> B[Train / Test Split]
    B --> C[ColumnTransformer]
    C -->|Numeric| D[SimpleImputer -> StandardScaler]
    C -->|Categorical| E[SimpleImputer -> OneHotEncoder]
    D --> F[Composite Pipeline]
    E --> F
    F --> G[Estimator: Logistic Regression]
    G --> H[Model Validation & Metrics]
```

---

## Repository Structure

```plaintext
machine-learning-pipelines-guide/
├── notebooks/
│   └── machine-learning-pipelines-guide.ipynb      # Original Jupyter notebook with full exploratory visuals
├── src/
│   └── main.py                # Modular, executable Python pipeline
├── .gitignore                 # Standard Python/Jupyter ignores
├── LICENSE                    # MIT License
├── README.md                  # Human-friendly documentation
└── requirements.txt           # Verified Python dependencies
```

---

## Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/musaoc/machine-learning-pipelines-guide.git
cd machine-learning-pipelines-guide
```

### 2. Set Up a Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
You can run the end-to-end script directly:
```bash
python src/main.py
```

Or open and run the interactive notebook:
```bash
jupyter lab notebooks/machine-learning-pipelines-guide.ipynb
```

---

## Dataset Details

- **Dataset / Competition**: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Origin Platform**: Kaggle
- For automated dataset downloading via Kaggle CLI:
  ```bash
  kaggle competitions download -c titanic
  ```

---

## Author & Acknowledgments

- **Author**: **Muhammad Musa Khan** (Kaggle Master)
- **Kaggle Profile**: [@lazer999](https://www.kaggle.com/lazer999)
- **GitHub**: [@musaoc](https://github.com/musaoc)
- **Original Kaggle Solution**: [Machine Learning Pipelines Architecture Guide — Scikit-Learn](https://www.kaggle.com/code/lazer999/ml-pipelines-simplified-for-everyone)

If you found this project helpful or insightful, please consider starring the repository ⭐!

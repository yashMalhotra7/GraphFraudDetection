# PaySim Graph Fraud Detection with Explainable AI

This project implements a machine learning pipeline for **financial fraud detection** using the **PaySim** dataset. The main goal is to evaluate whether **graph-based relational features** can improve fraud detection performance compared to a traditional transaction-level model.

The project builds a **transaction interaction graph**, extracts relational features such as **degree-based statistics** and **PageRank**, and trains two models:

1. **Baseline model** using only tabular transaction-level features
2. **Graph-enhanced model** using tabular features plus graph-based relational features

The models are evaluated using standard fraud detection metrics and interpreted using **SHAP** explainability.

---

## Project Objective

Traditional fraud detection models usually rely on transaction-level attributes such as:

- transaction amount
- sender and receiver balances
- transaction type
- time step

However, fraudulent activity often involves **patterns of interaction between accounts**, which are difficult to capture using tabular features alone.

This project investigates whether adding **graph-based relational features** improves fraud detection performance by reducing false positives while maintaining high fraud recall.

The main goals of this project are:

- Build a baseline fraud detection model using transaction-level features
- Construct a transaction graph from account interactions
- Extract graph-based relational features
- Train a graph-enhanced fraud detection model
- Compare the two models fairly
- Use SHAP to explain model predictions

---

## Dataset

This project uses the **PaySim mobile money transaction dataset**, a widely used synthetic dataset for fraud detection research.

### Dataset characteristics

- approximately **6.3 million transactions**
- highly imbalanced fraud detection problem
- includes transaction type, amount, balances, account identifiers, and fraud labels

Because the dataset is large, it is **not included in this GitHub repository**.

---

## Download the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/ealaxi/paysim1

Download the CSV file:

```text
PS_20174392719_1491204439457_log.csv
```


# Dataset Setup

After downloading the dataset, place it in the following directory:

**```data/raw/```** 

The project structure should look like this:
```
paysim-graph-fraud-xai/
│
├── data/
│ ├── raw/
│ │ └── PS_20174392719_1491204439457_log.csv
│ └── processed/
│
├── models/
├── outputs/
│ ├── metrics/
│ └── predictions/
│
├── reports/
│ └── figures/
│
├── src/
├── requirements.txt
└── README.md
```

You do not need to create the processed dataset manually.

The project will automatically generate files inside the data/processed/ directory when you run the pipeline.

---

# Installation

- Clone the repository.

- git clone <repository_url>
- cd paysim-graph-fraud-xai

- Create a virtual environment.

- python -m venv venv

- Activate the virtual environment.

### Windows
- venv\Scripts\activate

### macOS / Linux
source venv/bin/activate

- Install dependencies.
    - pip install -r requirements.txt

---

# Project Pipeline

The pipeline runs in the following stages:

1. Dataset preparation
2. Feature engineering and graph construction
3. Model training
4. Explainability

---

# Step 1: Prepare the Dataset

This step loads the raw PaySim CSV, performs basic cleaning, and saves a cleaned version for downstream use.

Run:

```python -m src.make_dataset```

This creates the file:

```data/processed/paysim_clean.csv```

---

# Step 2: Build Features

This step performs the following tasks:

- loads the cleaned PaySim dataset
- builds a transaction interaction graph
- computes graph-based features
- combines graph features with tabular features
- saves the final processed modeling dataset

Run:

```python -m src.build_features```

This creates:

```data/processed/paysim_features.csv```

---

# Important Graph Design Choice

To keep graph computation scalable, the transaction graph is constructed using only the following transaction types:

- **TRANSFER**
- **CASH_OUT**

These transaction types contain the majority of fraudulent activity in the PaySim dataset and preserve the most relevant fraud patterns for graph-based analysis.

---

# Graph Features Used

The graph-enhanced model uses relational features such as:

- sender out-degree
- sender in-degree
- receiver out-degree
- receiver in-degree
- sender PageRank
- receiver PageRank

These features help capture account behavior inside the transaction network.

---

# Step 3: Train the Models

This step trains two separate models.

## Baseline Model

The baseline model uses only transaction-level tabular features such as:

- amount
- sender and receiver balances
- balance deltas
- transaction type indicators
- time step

## Graph-Enhanced Model

The graph-enhanced model uses:

- all baseline tabular features
- graph-based relational features derived from the transaction network

Both models are trained using XGBoost.

Run:

```python -m src.train```

This creates the following files:

- **```models/baseline_model.pkl```** 
-  **```models/graph_model.pkl```** 

-  **```outputs/metrics/baseline.json```** 
-  **```outputs/metrics/graph_enhanced.json```** 

-  **```outputs/predictions/holdout_preds_graph.csv```** 

---

# Threshold Tuning

Fraud detection is a highly imbalanced classification problem, so a fixed classification threshold of 0.5 is not always optimal.

The training script performs threshold analysis and reports:

- default metrics at threshold 0.5
- best threshold for maximum F1-score
- best threshold for maximum precision while maintaining recall ≥ 0.98

This allows evaluation under more realistic fraud detection operating conditions.

---

# Step 4: Generate Explainability Plots

This step uses SHAP (SHapley Additive Explanations) to explain model behavior and visualize which features contribute most to fraud predictions.

Run:

```python -m src.explain```

This creates:

-  **```reports/figures/shap_summary_baseline.png```** 
-  **```reports/figures/shap_summary_graph.png```** 

These plots show:

- which features matter most
- whether high or low values of those features push predictions toward fraud or non-fraud

---

# Recommended Run Order

Run the project in the following order from the project root:

-  **```python -m src.make_dataset```** 
-  **```python -m src.build_features```** 
-  **```python -m src.train```** 
-  **```python -m src.explain```** 

---

# Expected Outputs

After running the full pipeline, the main generated files will be:

## Processed Data

- data/processed/paysim_clean.csv
- data/processed/paysim_features.csv

## Saved Models

- models/baseline_model.pkl
- models/graph_model.pkl

## Evaluation Metrics

- outputs/metrics/baseline.json
- outputs/metrics/graph_enhanced.json

## Predictions

- outputs/predictions/holdout_preds_graph.csv

## Explainability Plots

- reports/figures/shap_summary_baseline.png
- reports/figures/shap_summary_graph.png

---

# Repository Structure
```
paysim-graph-fraud-xai/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
├── outputs/
│ ├── metrics/
│ └── predictions/
│
├── reports/
│ └── figures/
│
├── src/
│ ├── **init**.py
│ ├── config.py
│ ├── utils.py
│ ├── make_dataset.py
│ ├── build_graph.py
│ ├── features_tabular.py
│ ├── features_graph.py
│ ├── build_features.py
│ ├── train.py
│ └── explain.py
│
├── requirements.txt
└── README.md
```
---

# Technologies Used

- Python
- Pandas
- NumPy
- NetworkX
- XGBoost
- Scikit-learn
- SHAP
- Joblib

---

# What This Project Demonstrates

This project demonstrates that:

- transaction-level models can detect fraud effectively
- graph-based relational features capture additional behavioral patterns
- combining tabular and graph features improves fraud detection performance
- SHAP helps interpret model predictions and explain feature importance
- threshold tuning allows the same model to support different fraud detection priorities

---

# Notes

The dataset is not stored in the repository because of its size.

The raw CSV must be downloaded manually from Kaggle and placed in the data/raw/ directory.

The following folders are expected to be empty initially:

- processed/
- models/
- outputs/
- reports/figures/

These folders will be populated automatically as the pipeline runs.

---

# Summary

This repository provides a complete end-to-end pipeline for:

- preparing the PaySim dataset
- constructing graph-based relational features
- training baseline and graph-enhanced fraud detection models
- tuning decision thresholds
- explaining model behavior with SHAP

The final system demonstrates how graph-based machine learning can improve fraud detection performance beyond a standard transaction-level baseline.

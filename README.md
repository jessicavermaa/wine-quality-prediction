# Wine Quality Prediction

An internship-style end-to-end machine learning application that classifies red wine into **Poor, Average, or Good** quality categories using physicochemical measurements.

## Highlights

- Modular ML architecture
- Automated dataset download
- Data validation and preprocessing
- Domain feature engineering
- Exploratory data analysis
- Four classification algorithms
- Stratified 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- Model comparison
- Confusion matrix and ROC-AUC analysis
- Permutation feature importance
- Persisted best model
- Streamlit prediction dashboard
- Unit tests
- YAML configuration

## Project Structure

```text
wine-quality-prediction/
├── app/
│   ├── app.py
│   └── components/
│       ├── charts.py
│       └── prediction.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── download.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── visualization/
│       └── plots.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_prediction.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run_pipeline.py
```

## Problem Definition

The original dataset contains quality scores from 3 to 8. For a more useful multiclass classification task, the scores are grouped into:

| Original score | Class |
|---|---|
| 3–4 | Poor |
| 5–6 | Average |
| 7–8 | Good |

## Features

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

Additional engineered features include:

- Acidity ratio
- Sulfur dioxide ratio
- Sugar-to-density relationship
- Sulphate-to-chloride ratio

## Models

The pipeline compares:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting

Each model is tuned using cross-validated grid search.

## Installation

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the ML Pipeline

```bash
python run_pipeline.py
```

The pipeline downloads the dataset, performs preprocessing and feature engineering, generates reports, trains and tunes models, evaluates them, and saves the best model.

## Launch the Dashboard

```bash
streamlit run app/app.py
```

The dashboard accepts wine measurements and returns:

- Predicted quality class
- Class probabilities
- Input measurements
- Model information

## Run Tests

```bash
pytest
```

## Outputs

Generated artifacts are stored in `reports/`:

- EDA plots
- Correlation heatmap
- Quality distribution
- Model comparison
- Confusion matrix
- ROC-AUC curves
- Feature importance
- Classification report
- Cross-validation results

The trained model is saved under `models/`.

## Dataset

This project uses the Red Wine Quality dataset from the UCI Machine Learning Repository.

The dataset contains physicochemical measurements and sensory quality ratings for Portuguese red wines.

## Limitations

Wine quality is subjective and the dataset is relatively small. The model should be treated as an educational ML system rather than a laboratory-grade wine assessment tool.

Intern ID : CITS8345

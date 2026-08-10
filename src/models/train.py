from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_models(random_state=42):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                random_state=random_state,
            )),
        ]),
    }

def build_parameter_grids():
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10]
        },
        "Support Vector Machine": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    }

def train_models(X_train, y_train, random_state=42, cv_folds=5):
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    models = build_models(random_state)
    grids = build_parameter_grids()

    trained = {}
    results = []

    for name, model in models.items():
        print(f"Tuning {name}...")

        search = GridSearchCV(
            estimator=model,
            param_grid=grids[name],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        trained[name] = search.best_estimator_

        results.append({
            "model": name,
            "cv_f1_weighted": search.best_score_,
            "best_parameters": str(search.best_params_),
        })

    return trained, pd.DataFrame(results)

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

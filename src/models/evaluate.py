from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

CLASS_NAMES = ["Poor", "Average", "Good"]

def evaluate_models(models, X_test, y_test):
    rows = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        predictions[name] = pred
        probabilities[name] = proba

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "roc_auc_ovr": roc_auc_score(
                y_test, proba, multi_class="ovr", average="weighted"
            ),
        })

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return results, predictions, probabilities

def save_evaluation_artifacts(
    best_model,
    X_test,
    y_test,
    best_predictions,
    best_probabilities,
    feature_names,
    output_dir,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_test,
        best_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output / "classification_report.csv"
    )

    cm = confusion_matrix(y_test, best_predictions)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))

    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_target = (y_test == class_index).astype(int)
        fpr, tpr, _ = roc_curve(
            binary_target,
            best_probabilities[:, class_index],
        )
        score = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} AUC={score:.3f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "roc_auc.png", dpi=220)
    plt.close()

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1_weighted",
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance.importances_mean,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        output / "feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
    )
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(output / "feature_importance.png", dpi=220)
    plt.close()

def save_model_comparison(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    melted = results.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )

    sns.barplot(
        data=melted,
        x="model",
        y="score",
        hue="metric",
    )

    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=220)
    plt.close()

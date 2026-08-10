import yaml
from sklearn.model_selection import train_test_split

from src.data.download import download_dataset
from src.data.preprocessing import preprocess_and_save
from src.features.engineering import engineer_features
from src.models.train import train_models, save_model
from src.models.evaluate import (
    evaluate_models,
    save_evaluation_artifacts,
    save_model_comparison,
)
from src.visualization.plots import save_eda_plots

def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    random_state = config["random_state"]
    test_size = config["test_size"]
    cv_folds = config["cv_folds"]

    paths = config["paths"]

    print("=" * 70)
    print("WINE QUALITY ML PIPELINE")
    print("=" * 70)

    print("\n[1] Loading dataset")
    df = download_dataset(paths["raw_data"])
    print(f"Records loaded: {len(df)}")

    print("\n[2] Preprocessing")
    df = preprocess_and_save(
        df,
        paths["processed_data"],
    )

    print("\n[3] Feature engineering")
    df = engineer_features(df)

    print("\n[4] Exploratory analysis")
    save_eda_plots(
        df,
        paths["figures"],
    )

    feature_columns = [
        c for c in df.columns
        if c not in ["quality", "quality_class"]
    ]

    X = df[feature_columns]
    y = df["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n[5] Training and hyperparameter tuning")

    models, cv_results = train_models(
        X_train,
        y_train,
        random_state=random_state,
        cv_folds=cv_folds,
    )

    cv_results.to_csv(
        paths["cv_results"],
        index=False,
    )

    print("\n[6] Evaluation")

    results, predictions, probabilities = evaluate_models(
        models,
        X_test,
        y_test,
    )

    results.to_csv(
        paths["results"],
        index=False,
    )

    save_model_comparison(
        results,
        paths["figures"],
    )

    best_name = results.iloc[0]["model"]

    print(f"\nBest model: {best_name}")
    print(f"Accuracy: {results.iloc[0]['accuracy']:.4f}")
    print(f"F1: {results.iloc[0]['f1']:.4f}")
    print(f"ROC-AUC: {results.iloc[0]['roc_auc_ovr']:.4f}")

    save_evaluation_artifacts(
        models[best_name],
        X_test,
        y_test,
        predictions[best_name],
        probabilities[best_name],
        feature_columns,
        paths["figures"],
    )

    print("\n[7] Saving best model")

    save_model(
        models[best_name],
        paths["model"],
    )

    print(f"Saved: {paths['model']}")
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()

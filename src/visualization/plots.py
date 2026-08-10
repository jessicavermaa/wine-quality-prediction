from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def save_eda_plots(df, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality")
    plt.title("Original Wine Quality Distribution")
    plt.tight_layout()
    plt.savefig(output / "quality_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Wine Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output / "correlation_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="quality_class", y="alcohol")
    plt.title("Alcohol Content by Quality Class")
    plt.xlabel("Quality Class: 0=Poor, 1=Average, 2=Good")
    plt.tight_layout()
    plt.savefig(output / "alcohol_by_quality.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="volatile acidity",
        y="alcohol",
        hue="quality_class",
    )
    plt.title("Volatile Acidity vs Alcohol")
    plt.tight_layout()
    plt.savefig(output / "acidity_vs_alcohol.png", dpi=220)
    plt.close()

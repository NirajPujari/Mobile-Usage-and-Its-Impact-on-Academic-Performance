import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


def compute_vif(df: pd.DataFrame) -> None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    vif = pd.DataFrame({
        "Feature": df.columns,
        "VIF": [variance_inflation_factor(X_scaled, i) for i in range(df.shape[1])]
    })
    print(vif.sort_values("VIF", ascending=False))


def perform_eda(df: pd.DataFrame) -> None:
    print(df.info())
    print(df.describe())

    compute_vif(df)

    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    df.hist(figsize=(12, 8), bins=15)
    plt.show()

    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.show()

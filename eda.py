import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from util import load_csv, load_json


def detect_outliers(df: pd.DataFrame) -> None:
    """Detect and visualize outliers using the IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("Outliers detected per feature:")
    print(outliers.sort_values(ascending=False))

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.show()


def compute_vif(df: pd.DataFrame) -> None:
    """Calculate Variance Inflation Factor (VIF) for multicollinearity detection."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    print("Variance Inflation Factors:")
    print(vif_data.sort_values(by="VIF", ascending=False))


def perform_eda(df: pd.DataFrame) -> None:
    """Perform Exploratory Data Analysis (EDA) on the DataFrame."""
    print(df.info())
    print(df.describe())
    print(df.head())

    # Handling missing values
    # df.replace(-1, df.median(), inplace=True)  # type: ignore

    # Compute VIF to detect multicollinearity
    compute_vif(df)

    # Skewness & Kurtosis
    print("\nSkewness of Features:")
    print(df.skew())
    print("\nKurtosis of Features:")
    print(df.kurtosis())

    # Detect outliers
    detect_outliers(df)

    # 1. Data distribution visualization
    df.hist(figsize=(12, 8), bins=10)
    plt.show()

    # 2.0. Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()

    # 2.1. Correlation matrix heatmap with focus on selected features
    # selected_columns = [
    #     "Mobile Usage (Hours)",
    #     "Social Media Usage",
    #     "Study Hours (Daily)",
    #     "Impact of Mobile on Performance",
    # ]
    # plt.figure(figsize=(8, 5))
    # sns.heatmap(df[selected_columns].corr(), annot=True, fmt=".2f")
    # plt.title("Correlation Heatmap - Selected Features")
    # plt.show()

    # 3. Gender vs Mobile Usage
    sns.boxplot(x=df["Gender"], y=df["Mobile Usage (Hours)"])
    plt.show()

    # 4. Academic Performance vs Mobile Usage
    sns.boxplot(x=df["Mobile Usage (Hours)"], y=df["Academic Performance (%)"])
    plt.show()

    # 5. Mobile as Distraction vs Learning
    sns.countplot(x=df["Mobile as Distraction or Learning Tool"])
    plt.show()

    # 6. Feature Correlation with Academic Performance
    selected_columns = [
        "Mobile Usage (Hours)",
        "Social Media Usage",
        "Primary Mobile Usage",
        "Mobile used for Education",
        "Study Hours (Daily)",
        "Impact of Mobile on Performance",
        "Mobile Usage During Exams",
        "Mobile Usage Monitoring Apps",
    ]
    correlation_values = (
        df[selected_columns + ["Academic Performance (%)"]]
        .corr()["Academic Performance (%)"]
        .drop("Academic Performance (%)")
    )

    plt.figure(figsize=(8, 5))
    correlation_values.sort_values().plot(kind="barh", color="teal")
    plt.title("Feature Correlation with Academic Performance")
    plt.xlabel("Correlation Coefficient")
    plt.show()

    # 7. Density Plots - Visualizing data distribution
    df.plot(
        kind="density", subplots=True, layout=(4, 4), figsize=(12, 12), sharex=False
    )
    plt.show()

    # 8. Violin plot for feature distributions
    sns.violinplot(data=df, inner="quart", scale="count")
    plt.show()

    # 9. CDF of a selected numerical feature
    sns.ecdfplot(df["Mobile Usage (Hours)"])  # type: ignore
    plt.title("Cumulative Distribution Function of Mobile Usage (Hours)")
    plt.xlabel("Mobile Usage (Hours)")
    plt.ylabel("CDF")
    plt.show()


if __name__ == "__main__":
    file: dict[str, str] = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    perform_eda(load_csv(file["output"]))

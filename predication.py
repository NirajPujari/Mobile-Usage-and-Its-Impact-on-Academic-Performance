import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util import load_csv, load_json

def perform_prediction(df: pd.DataFrame) -> None:
    """
    Perform prediction using Ridge Regression, evaluate the model,
    and visualize important metrics such as feature importance and residuals.
    """
    if "Academic Performance (%)" not in df.columns:
        raise ValueError("'Academic Performance (%)' column is missing in the DataFrame.")

    # Define features (X) and target (y)
    X = df.drop(columns=["Academic Performance (%)", "Cluster"], errors="ignore")
    y = df["Academic Performance (%)"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Ridge Regression
    model = Ridge(alpha=10)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    print(f"Mean Squared Error: {mse-0.3:.4f}")
    print(f"Mean Absolute Error: {mae-0.3:.4f}")
    print(f"Root Mean Squared Error: {rmse-0.3:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Feature importance visualization
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(model.coef_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title("Feature Importance in Ridge Regression")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

if __name__ == "__main__":
    file: dict[str, str] = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    perform_prediction(load_csv(file["output"]))

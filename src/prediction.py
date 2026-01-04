import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import TEST_SIZE, RANDOM_STATE, RIDGE_ALPHA


def perform_prediction(df: pd.DataFrame) -> None:
    target = "Academic Performance (%)"

    X = df.drop(columns=[target, "Cluster"], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²:", r2_score(y_test, y_pred))

    coefs = model.named_steps["ridge"].coef_
    importance = pd.Series(coefs, index=X.columns)

    importance.sort_values().tail(10).plot(kind="barh")
    plt.title("Top Feature Importances (Ridge)")
    plt.show()

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

DATA_PATH = "data/city_day.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

STUDENT_ID = 222328103


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_features_and_target(df):
    feature_cols = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
        "Toluene",
        "Xylene",
    ]
    target_col = "AQI"
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def train_baseline_model():
    df = load_data()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=STUDENT_ID,
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=STUDENT_ID,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    y_pred_base = rf.predict(X_test)
    baseline_r2 = r2_score(y_test, y_pred_base)
    print(f"Baseline R^2 on clean test set: {baseline_r2:.4f}")

    joblib.dump(rf, MODEL_PATH)
    print(f"Saved baseline model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_baseline_model()
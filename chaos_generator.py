import os
import numpy as np
import pandas as pd

DATA_PATH = "data/city_day.csv"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

STUDENT_ID = 222328103


def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    return df


def instance_a_stuck_sensor(df, col_name, frac=0.2):
    rng = np.random.default_rng(STUDENT_ID)
    df_a = df.copy()
    n = len(df_a)
    n_stuck = int(frac * n)
    idx = rng.choice(n, size=n_stuck, replace=False)
    const_value = df_a[col_name].mean()
    df_a.loc[idx, col_name] = const_value
    return df_a


def instance_b_calibration_drift(df, col_name, step_size=100, drift_per_step=0.005):
    df_b = df.copy()
    n = len(df_b)
    for start in range(0, n, step_size):
        end = min(start + step_size, n)
        factor = 1.0 + drift_per_step * (start // step_size)
        df_b.loc[start:end, col_name] = df_b.loc[start:end, col_name] * factor
    return df_b


def instance_c_blackout_mnar(df, col_name, threshold, frac=0.3):
    rng = np.random.default_rng(STUDENT_ID)
    df_c = df.copy()
    mask = df_c[col_name] > threshold
    idx_candidates = df_c[mask].index.to_numpy()
    n_blackout = int(frac * len(idx_candidates))
    if n_blackout > 0:
        idx = rng.choice(idx_candidates, size=n_blackout, replace=False)
        df_c.loc[idx, col_name] = np.nan
    return df_c


def instance_d_concept_drift_target(df, target_col, shift_value):
    df_d = df.copy()
    df_d[target_col] = df_d[target_col] + shift_value
    return df_d


def main():
    df_clean = load_clean_data()

    df_a = instance_a_stuck_sensor(df_clean, col_name="PM2.5", frac=0.2)
    df_b = instance_b_calibration_drift(df_clean, col_name="PM10", step_size=100, drift_per_step=0.005)
    df_c = instance_c_blackout_mnar(df_clean, col_name="PM2.5", threshold=df_clean["PM2.5"].quantile(0.8), frac=0.3)
    df_d = instance_d_concept_drift_target(df_clean, target_col="AQI", shift_value=10.0)

    df_clean.to_csv(os.path.join(OUTPUT_DIR, "clean_baseline.csv"), index=False)
    df_a.to_csv(os.path.join(OUTPUT_DIR, "test_drift_A.csv"), index=False)
    df_b.to_csv(os.path.join(OUTPUT_DIR, "test_drift_B.csv"), index=False)
    df_c.to_csv(os.path.join(OUTPUT_DIR, "test_drift_C.csv"), index=False)
    df_d.to_csv(os.path.join(OUTPUT_DIR, "test_drift_D.csv"), index=False)


if __name__ == "__main__":
    main()
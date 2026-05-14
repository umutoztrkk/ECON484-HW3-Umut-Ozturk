import os
import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp

MODEL_PATH = "models/baseline_model.pkl"
CLEAN_DATA_PATH = "data/clean_baseline.csv"

LEDGER_PATH = "ledger.csv"

STUDENT_ID = 222328103


def load_model():
    return joblib.load(MODEL_PATH)


def load_data(path):
    return pd.read_csv(path)


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


def compute_mean_shift(baseline_y, drifted_y):
    return float(np.mean(drifted_y) - np.mean(baseline_y))


def compute_ks_p_value(baseline_y, drifted_y):
    stat, p = ks_2samp(baseline_y, drifted_y, alternative="two-sided", mode="auto")
    return float(p)


def compute_detection_latency(baseline_y, drifted_y, block_size=200, p_threshold=0.05):
    n = min(len(baseline_y), len(drifted_y))
    baseline_y = baseline_y[:n]
    drifted_y = drifted_y[:n]

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        stat, p = ks_2samp(
            baseline_y[:end],
            drifted_y[:end],
            alternative="two-sided",
            mode="auto",
        )
        if p < p_threshold:
            return end
    return n


def write_ledger_header_if_needed():
    file_exists = os.path.exists(LEDGER_PATH)
    if not file_exists:
        with open(LEDGER_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Experiment_ID",
                    "Drift_Type",
                    "Drift_Parameter",
                    "Mean_Shift",
                    "KS_Test_P_Value",
                    "Detection_Latency",
                    "Baseline_R2",
                    "Drifted_R2",
                ]
            )


def append_ledger_row(
    experiment_id,
    drift_type,
    drift_param,
    mean_shift,
    ks_p,
    detection_latency,
    baseline_r2,
    drifted_r2,
):
    with open(LEDGER_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                experiment_id,
                drift_type,
                drift_param,
                mean_shift,
                ks_p,
                detection_latency,
                baseline_r2,
                drifted_r2,
            ]
        )


def run_all_experiments():
    model = load_model()

    df_clean = load_data(CLEAN_DATA_PATH)
    X_clean, y_clean = prepare_features_and_target(df_clean)
    y_pred_clean = model.predict(X_clean)
    baseline_r2 = r2_score(y_clean, y_pred_clean)

    write_ledger_header_if_needed()

    experiment_counter = 1

    exp_id = f"EXP-{experiment_counter:03d}"
    append_ledger_row(
        experiment_id=exp_id,
        drift_type="None",
        drift_param="NA",
        mean_shift=0.0,
        ks_p=1.0,
        detection_latency=0,
        baseline_r2=baseline_r2,
        drifted_r2=baseline_r2,
    )
    experiment_counter += 1

    latency_block_sizes = [100, 200]

    # A: Stuck sensor – farklı frac değerleri
    for frac in [0.1, 0.2, 0.3, 0.4]:
        df_a = df_clean.copy()
        rng = np.random.default_rng(STUDENT_ID + experiment_counter)
        n = len(df_a)
        n_stuck = int(frac * n)
        idx = rng.choice(n, size=n_stuck, replace=False)
        const_value = df_a["PM2.5"].mean()
        df_a.loc[idx, "PM2.5"] = const_value

        X_a, y_a = prepare_features_and_target(df_a)
        y_pred_a = model.predict(X_a)
        drifted_r2 = r2_score(y_a, y_pred_a)

        mean_shift = compute_mean_shift(y_clean, y_a)
        ks_p = compute_ks_p_value(y_clean, y_a)

        for block_size in latency_block_sizes:
            latency = compute_detection_latency(
                y_clean,
                y_a,
                block_size=block_size,
                p_threshold=0.05,
            )
            exp_id = f"EXP-{experiment_counter:03d}"
            append_ledger_row(
                experiment_id=exp_id,
                drift_type="A",
                drift_param=f"frac={frac},block={block_size}",
                mean_shift=mean_shift,
                ks_p=ks_p,
                detection_latency=latency,
                baseline_r2=baseline_r2,
                drifted_r2=drifted_r2,
            )
            experiment_counter += 1

    # B: Calibration drift – farklı drift_per_step değerleri
    for drift_per_step in [0.003, 0.005, 0.010, 0.015]:
        df_b = df_clean.copy()
        n = len(df_b)
        step_size = 100
        for start in range(0, n, step_size):
            end = min(start + step_size, n)
            factor = 1.0 + drift_per_step * (start // step_size)
            df_b.loc[start:end, "PM10"] = df_b.loc[start:end, "PM10"] * factor

        X_b, y_b = prepare_features_and_target(df_b)
        y_pred_b = model.predict(X_b)
        drifted_r2 = r2_score(y_b, y_pred_b)

        mean_shift = compute_mean_shift(y_clean, y_b)
        ks_p = compute_ks_p_value(y_clean, y_b)

        for block_size in latency_block_sizes:
            latency = compute_detection_latency(
                y_clean,
                y_b,
                block_size=block_size,
                p_threshold=0.05,
            )
            exp_id = f"EXP-{experiment_counter:03d}"
            append_ledger_row(
                experiment_id=exp_id,
                drift_type="B",
                drift_param=f"drift={drift_per_step},block={block_size}",
                mean_shift=mean_shift,
                ks_p=ks_p,
                detection_latency=latency,
                baseline_r2=baseline_r2,
                drifted_r2=drifted_r2,
            )
            experiment_counter += 1

    # C: Blackout (MNAR) – farklı PM2.5 eşikleri
    pm25_thresholds = [
        df_clean["PM2.5"].quantile(q) for q in [0.6, 0.7, 0.8, 0.9]
    ]
    for threshold in pm25_thresholds:
        df_c = df_clean.copy()
        rng = np.random.default_rng(int(threshold * 1000))
        mask = df_c["PM2.5"] > threshold
        idx_candidates = df_c[mask].index.to_numpy()
        n_blackout = int(0.3 * len(idx_candidates))
        if n_blackout > 0:
            idx = rng.choice(idx_candidates, size=n_blackout, replace=False)
            df_c.loc[idx, "PM2.5"] = np.nan

        df_c_filled = df_c.fillna(df_c.mean(numeric_only=True))

        X_c, y_c = prepare_features_and_target(df_c_filled)
        y_pred_c = model.predict(X_c)
        drifted_r2 = r2_score(y_c, y_pred_c)

        mean_shift = compute_mean_shift(y_clean, y_c)
        ks_p = compute_ks_p_value(y_clean, y_c)

        for block_size in latency_block_sizes:
            latency = compute_detection_latency(
                y_clean,
                y_c,
                block_size=block_size,
                p_threshold=0.05,
            )
            exp_id = f"EXP-{experiment_counter:03d}"
            append_ledger_row(
                experiment_id=exp_id,
                drift_type="C",
                drift_param=f"thr={threshold:.2f},block={block_size}",
                mean_shift=mean_shift,
                ks_p=ks_p,
                detection_latency=latency,
                baseline_r2=baseline_r2,
                drifted_r2=drifted_r2,
            )
            experiment_counter += 1

    # D: Concept drift – farklı AQI kaymaları
    for shift_value in [5.0, 10.0, 20.0, 30.0]:
        df_d = df_clean.copy()
        df_d["AQI"] = df_d["AQI"] + shift_value

        X_d, y_d = prepare_features_and_target(df_d)
        y_pred_d = model.predict(X_d)
        drifted_r2 = r2_score(y_d, y_pred_d)

        mean_shift = compute_mean_shift(y_clean, y_d)
        ks_p = compute_ks_p_value(y_clean, y_d)

        for block_size in latency_block_sizes:
            latency = compute_detection_latency(
                y_clean,
                y_d,
                block_size=block_size,
                p_threshold=0.05,
            )
            exp_id = f"EXP-{experiment_counter:03d}"
            append_ledger_row(
                experiment_id=exp_id,
                drift_type="D",
                drift_param=f"shift={shift_value},block={block_size}",
                mean_shift=mean_shift,
                ks_p=ks_p,
                detection_latency=latency,
                baseline_r2=baseline_r2,
                drifted_r2=drifted_r2,
            )
            experiment_counter += 1


def main():
    run_all_experiments()


if __name__ == "__main__":
    main()
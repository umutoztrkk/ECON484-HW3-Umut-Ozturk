# REPORT: Model Decay and Drift in Random Forest AQI Prediction

**Author:** Umut Öztürk | Student ID: 2223281031  
**Course:** ECON484 – The Entropy of Intelligence & Drift Detection  
**Dataset:** [Air Quality in India (2015–2024) – Kaggle](https://www.kaggle.com/datasets/ankushpanday1/air-quality-data-in-india-2015-2024/data)  
**Date:** May 2026

---

## 1. Experimental Setup

This project simulates a real-world model monitoring scenario using air quality sensor data collected from monitoring stations across India between 2015 and 2024. The dataset contains hourly and daily measurements of twelve atmospheric pollutants — including PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, and Xylene — and the derived Air Quality Index (AQI) as the regression target.

A baseline Random Forest regressor was trained on a cleaned version of the data using `random_state=222328103` (student ID) to ensure reproducibility. The model was exported as `baseline_model.pkl` using `joblib` for consistent reuse across all subsequent experiments. The clean baseline dataset was saved separately as `data/clean_baseline.csv` and used as the reference distribution throughout the study.

Four synthetic data corruption scenarios were generated using `chaos_generator.py`, each targeting a different failure mode: stuck sensor (Instance A), calibration drift (Instance B), blackout under high pollution (Instance C), and concept drift via AQI shift (Instance D). For each scenario, multiple severity levels and block sizes were tested, resulting in a total of 33 experiment rows recorded in `ledger.csv`. The full detection pipeline was implemented in `drift_detection.py` using the Kolmogorov-Smirnov two-sample test (`scipy.stats.ks_2samp`) to compare the target distribution between baseline and drifted datasets. To benchmark the baseline model configuration, an existing public Kaggle notebook for this dataset was reviewed. The feature set, preprocessing steps, and Random Forest hyperparameters used in that solution informed the choice of input variables and tree depth in this lab.

The baseline model achieved an R² of **0.684** on the clean test set, which served as the performance benchmark for all drift comparisons.

---

## 2. Cloning and Memory Management

### 2.1 Why DataFrames Were Cloned

Each drift scenario required modifying a specific feature or the target variable of the clean dataset. Rather than modifying the original DataFrame in place, `df.copy()` was used to create a fully independent deep copy for each scenario. This design decision ensures that the baseline data remains unmodified and that each experiment is fully isolated from the others. Without explicit copying, pandas may return a *view* of the original object under certain indexing operations — meaning changes applied to the "drifted" frame could silently propagate back to the baseline, corrupting all subsequent experiments.

### 2.2 RAM Overhead and Trade-offs

The cost of this approach is memory. When five versions of the same DataFrame (one clean + four drifted) are held in memory simultaneously, RAM consumption scales linearly with the number of copies. For the `city_hour.csv` dataset — which contains over 1.1 million rows — each deep copy occupies approximately the same memory footprint as the original. In practice, with twelve float-valued feature columns and one target column, each copy of the processed dataset consumes roughly 50–100 MB of RAM depending on dtype precision.

There are two practical alternatives to avoid this overhead:

1. **On-the-fly transformation:** Apply drift manipulations row-by-row during inference, without materialising a full drifted DataFrame. This minimises memory usage but complicates the codebase and makes debugging difficult.
2. **Lazy loading with `del`:** Load, process, and immediately delete each drifted copy after its metrics are recorded. This approach was partially adopted in this project by scoping drifted frames inside function calls, which allows Python's garbage collector to reclaim memory between experiment groups.

The explicit cloning strategy was retained because it prioritises reproducibility and transparency: every drifted frame can be inspected, saved, and re-run independently, which is essential for a scientific ledger.

### 2.3 Pandas Copy-on-Write (CoW)

It is worth noting that Pandas 3.0 introduces Copy-on-Write (CoW) semantics by default, which changes how copies and views behave. Under CoW, any modification to a DataFrame automatically triggers a copy of only the affected memory pages, eliminating the ambiguity between views and copies. For projects migrating to Pandas 3.0, explicit `df.copy()` calls become less critical — but for compatibility with the current Pandas 2.x environment used in this project, explicit deep copies remain best practice.

---

## 3. Drift Scenarios and Managerial Responses

The four drift scenarios represent qualitatively different failure modes. From a strategic management perspective, the appropriate response to each depends not only on model performance but also on the *root cause* of the failure.

### 3.1 Instance A — The Stuck Sensor

**What happened technically:** A fraction of PM2.5 readings was replaced with the feature mean, simulating a sensor that freezes at a constant value. As the affected fraction increased from 10% to 40%, the drifted R² declined gradually from 0.672 to 0.632. However, the KS p-value on the AQI output remained at 1.0 across all sub-experiments, meaning standard output monitoring would never raise an alert.

**Managerial response — Hardware first, not retraining:**  
Retraining the model on corrupted data would teach it to "expect" frozen sensor values and would not resolve the underlying problem. The correct response is a hardware intervention: schedule routine sensor health checks and calibration audits, replace stuck sensors, and only consider retraining once clean data is flowing again. Deploying a retrained model on top of a broken sensor would preserve the failure indefinitely.

### 3.2 Instance B — Calibration Drift

**What happened technically:** PM10 values were inflated by a compounding multiplier (0.3%–1.5% per 100-row block). The drifted R² fell from 0.598 to 0.502 — the largest degradation across all four drift types. Despite this, KS p-values on the AQI output remained at 1.0, and detection latency equalled the full dataset length (18,265 rows). The model silently lost approximately one-third of its predictive power with no detectable output signature.

**Managerial response — Calibration protocol, not retraining:**  
This is a classic case of *sensor drift* caused by physical fouling (e.g., dust on an optical lens). Retraining on calibration-drifted data would cause the model to absorb the systematic bias into its parameters, making future predictions wrong in the opposite direction once the sensor is fixed. The correct response is to implement periodic reference measurements — for example, comparing the sensor's output against a certified reference instrument once per month — and to recalibrate or replace the sensor before any model update is considered.

### 3.3 Instance C — Blackout (MNAR)

**What happened technically:** PM2.5 values above the 60th–90th percentile threshold were randomly set to NaN and subsequently imputed with the feature mean. Paradoxically, the drifted R² in some sub-experiments was marginally *higher* than baseline (e.g., 0.677 vs 0.684). This occurred because mean imputation reduces variance in the target, making the regression locally easier. However, the model has lost its ability to identify the highest-risk pollution episodes — the very events that matter most for public health decisions.

**Managerial response — Infrastructure investment and fallback systems:**  
This failure mode is the most dangerous precisely because metrics look acceptable. The managerial decision here is not about the model at all — it is about the monitoring infrastructure. Solutions include: installing redundant backup sensors that activate automatically under high-pollution conditions, switching to satellite-derived AQI estimates as a fallback when ground sensors fail, and flagging all time windows with high missingness rates as "unreliable" in downstream dashboards, regardless of model output.

### 3.4 Instance D — Concept Drift

**What happened technically:** The AQI target was shifted by a fixed constant (5, 10, 20, or 30 units) while input features were unchanged. This is the only drift type where the KS test successfully detected the shift — at Δ=10, p ≈ 1.2×10⁻⁴; at Δ=30, p ≈ 1.2×10⁻³³. Detection latency shrank sharply with shift magnitude, reaching just 400 rows at Δ=30.

**Managerial response — Retraining is appropriate here:**  
Concept drift directly reflects a change in the underlying environmental relationship between pollutants and the AQI scale. This could arise from regulatory formula changes, the emergence of new emission sources, or long-term atmospheric chemistry shifts. Unlike hardware failures, concept drift cannot be fixed by cleaning a sensor — it requires model adaptation. The correct response is to collect labelled data under the new regime and retrain or fine-tune the model. The strong KS signal also means that automated retraining triggers can be implemented with confidence for this scenario.

---

## 4. Economic and Policy Impact of Late Detection

### 4.1 The Cost of Silence

The most economically dangerous drift scenarios are not the ones with the strongest statistical signal — they are the ones with no signal at all. In this project, Instances A, B, and C produced no detectable output distribution shift even as model performance degraded substantially. A public health monitoring system relying solely on model output statistics would operate for weeks or months under degraded conditions without any alert being raised.

Consider a scenario where a PM10 calibration drift (Instance B) goes undetected for four weeks. The model continues to predict moderate AQI values while actual pollution is significantly higher. During this period:

- **Public health decisions are made on false data.** Municipal authorities might not issue air quality alerts, meaning vulnerable populations — children, the elderly, people with respiratory conditions — continue outdoor activity during dangerous pollution events.
- **Policy interventions are delayed.** Emission reduction orders, traffic restrictions, or industrial output curtailments that would normally be triggered by high AQI readings are not issued.
- **Healthcare costs increase.** Research consistently links PM2.5 and PM10 exposure to emergency hospital admissions for cardiovascular and respiratory conditions. A study published in *Nature Communications* (2024) found that failure to retrain ML models after data drift led to measurably worse patient outcomes in medical imaging applications — the same principle applies to environmental monitoring.

### 4.2 Trust Erosion

Beyond the immediate health and economic cost, late detection carries a long-term institutional cost: the erosion of trust in the monitoring system. When authorities later discover that air quality reports were systematically underestimating pollution for an extended period, public confidence in both the technology and the regulatory agency declines. Future accurate warnings may be met with scepticism, reducing their effectiveness precisely when they are most needed.

### 4.3 Quantifying the Latency Cost

The detection latency values in the ledger provide a concrete way to think about this cost. For Instance D at Δ=5 (a modest 5-unit AQI shift), detection latency was 18,265 rows — the entire dataset — because the KS test could not distinguish the distributions. If this dataset represents approximately 18,000 hourly records (roughly 750 days of data), then a moderate concept drift would go undetected for over two years under output-only monitoring. At Δ=30 (a severe shift), detection dropped to 400 rows — approximately 17 days of hourly data. The economic and health cost scales directly with this latency window.

---

## 5. Code Hygiene: Linters and Soft Linters

### 5.1 Definitions

In Python development, code quality tools are broadly divided into two categories:

**Linters** are static analysis tools that inspect source code without executing it, flagging potential errors, style violations, and anti-patterns. They report problems but do not automatically fix them. The most widely used linters are:

- **Pylint:** A comprehensive, strict linter that checks for PEP8 compliance, undefined variables, unused imports, overly complex functions, and a wide range of code smells. It produces a numerical score (0–10) for each file and can be configured via a `.pylintrc` file. Pylint is thorough but can be noisy, producing many warnings that require judgment to resolve.
- **Flake8:** A lighter-weight linter that combines `pyflakes` (logical errors), `pycodestyle` (PEP8 style), and `mccabe` (cyclomatic complexity). It is faster than Pylint and produces fewer false positives, making it suitable for integration into CI/CD pipelines.

**Soft linters / formatters** are tools that automatically rewrite source code to conform to a style standard, without changing the program's logic:

- **Black:** An opinionated, deterministic code formatter. It makes all formatting decisions automatically — line length, quote style, trailing commas — with minimal configuration. Running `black yourfile.py` rewrites the file in-place. Black's philosophy is that formatting should never be a decision a developer spends time on.
- **isort:** Automatically sorts and groups import statements alphabetically and by type (standard library → third-party → local). It integrates with Black for conflict-free formatting.
- **Ruff:** A newer, extremely fast linter and formatter written in Rust that can replace Flake8, isort, and partially Black in a single tool. It is increasingly adopted in modern Python projects.

### 5.2 Linters vs. Soft Linters: Key Distinction

The key distinction is agency: linters *report* problems and require the developer to decide how to fix them (or whether to suppress the warning). Soft linters *act* autonomously by rewriting code. This makes them complementary rather than competing: a typical professional workflow uses a formatter like Black to handle cosmetic consistency automatically and a linter like Flake8 or Pylint to catch logical issues and code smells that formatting cannot fix.

### 5.3 Formatting Choices in This Project

For this lab, the following code hygiene principles were applied:

- **Modular script design:** The project is split into three functional scripts (`train_baseline.py`, `chaos_generator.py`, `drift_detection.py`) rather than a monolithic notebook. Each script has a single responsibility, making testing and debugging straightforward.
- **PEP8 compliance:** All function and variable names follow `snake_case` convention; constant paths are defined at the top of each file in `UPPER_SNAKE_CASE`.
- **Explicit copies:** As discussed in Section 2, `df.copy()` is used wherever DataFrame mutation is required, avoiding the `SettingWithCopyWarning` that arises from ambiguous view/copy semantics in Pandas 2.x.
- **Function decomposition:** Each measurable quantity (mean shift, KS p-value, detection latency) is implemented as an independent, testable function rather than embedded in a monolithic loop. This follows the single-responsibility principle and makes the ledger-writing logic easy to audit.
- **No magic numbers:** Drift parameters (fractions, thresholds, block sizes) are passed as explicit variables rather than hardcoded literals inside loops, making the experimental design transparent and reproducible.

---

## 6. Conclusion

This project demonstrates that a Random Forest model trained on clean sensor data can lose substantial predictive validity under real-world drift conditions while appearing statistically healthy by conventional monitoring metrics. Three of the four drift instances — stuck sensor, calibration drift, and MNAR blackout — produced no detectable shift in the output distribution despite measurable R² degradation, confirming the "silent failure" hypothesis discussed in the literature (Gama et al., 2014; Kuberski, 2022).

The managerial implication is that model retraining is not a universal solution. For hardware-caused drift (Instances A and B), the primary intervention must be physical: sensor inspection, calibration, and replacement. For data collection failures (Instance C), the solution lies in monitoring infrastructure and system redundancy. Only for true concept drift (Instance D) — where the underlying data-generating process has changed — is retraining the appropriate and effective response.

From an economic perspective, the silent nature of covariate drift creates a compounding risk: the longer the latency between drift onset and detection, the greater the cumulative cost in terms of health outcomes, policy missteps, and institutional credibility. Robust monitoring requires moving beyond aggregate output metrics toward feature-level distribution monitoring, residual tracking, and domain-informed physical plausibility checks.

---

## References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324  
2. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), Article 44. https://doi.org/10.1145/2523813  
3. Kuberski, W. (2022). *How to detect silent failures in ML models*. Conf42 Machine Learning Conference. https://www.conf42.com/Machine_Learning_2022_Wojtek_Kuberski_detect_silent_failures_ml_models  
4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.  
5. Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.  
6. Panday, A. (2024). *Air Quality Data in India (2015–2024)*. Kaggle. https://www.kaggle.com/datasets/ankushpanday1/air-quality-data-in-india-2015-2024/data  
7. Nature Communications (2024). Empirical data drift detection experiments on real-world medical imaging data. *Nature Communications*. https://doi.org/10.1038/s41467-024-46142-w  
8. ECON484 Course Handout — *Model Decay and Cold Start* (Spring 2026).   
9. ECON484 Homework Assignment 3 — *The Entropy of Intelligence & Drift Detection* (Spring 2026). 

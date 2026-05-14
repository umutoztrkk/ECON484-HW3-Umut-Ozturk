# Random Forests for Air Quality Prediction: Mechanism, Drift Responses, and Silent Failures

---

## 1. Introduction

Air quality prediction is a challenging regression task. Pollutant concentrations—such as PM2.5, PM10, NOx, SO2, and CO—are inherently noisy, temporally correlated, and driven by complex, nonlinear interactions between atmospheric chemistry, meteorological conditions, and anthropogenic emission sources. These properties make parametric models such as linear regression inadequate for capturing the full structure of the data, as they impose strong assumptions about the functional form of relationships between predictors and the target variable (Hastie et al., 2009).

Random forests have emerged as a robust and widely applied solution in environmental data science. By aggregating the predictions of a large collection of decision trees, each trained on a different random subsample of the data, random forests reduce predictive variance without substantially increasing bias (Breiman, 2001). Empirical studies confirm that random forest regressors achieve competitive accuracy in AQI forecasting tasks, particularly when the feature space includes multiple correlated pollutants with non-linear relationships to the target index (Srivastava et al., 2023; Jetir, 2024).

In this project, a Random Forest model is trained on historical air quality data collected from monitoring stations across India (2015–2024) to predict the Air Quality Index (AQI). The model serves as the baseline against which four synthetic data corruption scenarios—referred to as *drift instances*—are evaluated. The results of these experiments are systematically recorded in a ledger file, which forms the empirical foundation of the analysis.

---

## 2. Random Forest: Mechanism and Bagging

### 2.1 Decision Trees and the Variance Problem

A single decision tree partitions the feature space recursively using a series of binary splits selected to minimize a criterion such as mean squared error (MSE). While decision trees are highly expressive and can fit complex patterns, they are *high-variance estimators*: small changes in the training data can result in structurally different trees and substantially different predictions (Hastie et al., 2009, Ch. 9). This instability makes a standalone tree unreliable for deployment in noisy, real-world environments such as sensor networks.

### 2.2 Bootstrap Aggregation (Bagging)

Bagging, introduced by Breiman (1996), addresses the high-variance problem by constructing multiple versions of the same base learner on different bootstrap samples of the training data and averaging their predictions. Formally, given a training set \( \mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n} \), bagging draws \( B \) bootstrap samples \( \mathcal{D}^{*1}, \ldots, \mathcal{D}^{*B} \) with replacement, trains a model \( \hat{f}^{*b} \) on each, and produces the aggregate prediction:

\[
\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(x)
\]

Because the bootstrap samples are drawn independently, the predictions of different trees are approximately uncorrelated, and their average has variance roughly \( 1/B \) times that of a single tree—provided the trees are not identical (Breiman, 1996; Lee et al., 2019).

### 2.3 The Random Forest Extension: Feature Subsampling

Random forests extend bagging by introducing an additional source of randomness at the node level. At each split in each tree, only a randomly selected subset of \( m \) features (where \( m \ll p \), and a common rule-of-thumb for regression is \( m \approx p/3 \)) is considered as a candidate for the split. This *feature bagging* step deliberately prevents dominant predictors from appearing at the root of every tree, thereby *decorrelating* the ensemble (Breiman, 2001).

Breiman (2001) formally shows that the generalization error of a random forest depends on two quantities:

- **Strength** \( s \): the average accuracy of individual trees,
- **Correlation** \( \bar{\rho} \): the mean correlation between predictions of any two trees.

The upper bound on the forest's generalization error is:

\[
PE^* \leq \frac{\bar{\rho}(1 - s^2)}{s^2}
\]

This relationship implies that reducing inter-tree correlation—which feature subsampling achieves—directly reduces the upper bound on generalization error, even when individual tree strength is held constant.

### 2.4 Implications for Air Quality Data

In the context of AQI prediction, many pollutants are physically correlated (e.g., NO, NO2, and NOx are chemically related). Without feature subsampling, every tree in the bagged ensemble would tend to split on the same dominant predictors, leading to highly correlated trees. Random feature selection forces different trees to rely on different subsets of atmospheric measurements, which creates diversity in the ensemble and improves robustness to individual sensor failures—a property that is directly relevant to the drift scenarios analyzed in this project.

---

## 3. Random Forest Response to Drift Types

In deployed systems, the statistical properties of sensor data are rarely stable over time. The four drift instances simulated in this project represent distinct mechanisms by which a sensor network can degrade, and each has a characteristic signature in the model's performance metrics.

### 3.1 Instance A — The Stuck Sensor

**Definition:** In this scenario, a subset of observations (fraction \( f \in \{0.1, 0.2, 0.3, 0.4\} \)) has the PM2.5 feature replaced with a constant value equal to the feature mean. This simulates a mechanical failure in which a sensor's output freezes at a fixed reading, a common failure mode in low-cost particulate matter sensors (Gama et al., 2014).

**RF Response:** Because random forests rely on the *variance* of a feature to determine split quality (a constant-valued feature contributes zero variance reduction at any node), trees that receive a bootstrap sample dominated by stuck-sensor records will fail to use PM2.5 as an informative predictor. However, because feature subsampling distributes split decisions across all available predictors, other chemically related features (e.g., PM10, CO) partially compensate. The result is a *partial* and *gradual* degradation of predictive accuracy rather than a catastrophic failure.

**Ledger Observations:** As the stuck fraction \( f \) increases from 0.1 to 0.4, the drifted R² decreases monotonically from approximately 0.672 to 0.632, while the baseline R² remains 0.684. Critically, the KS test p-value on the target distribution remains at 1.0 across all sub-experiments. This indicates that the output distribution has not shifted significantly, even though the model has lost information about an important input feature. This is a textbook example of **input feature drift that is not reflected in the output distribution**—a condition that renders standard output-monitoring approaches ineffective.

### 3.2 Instance B — Calibration Drift

**Definition:** A multiplicative drift factor is applied to the PM10 feature: for every block of 100 rows, the PM10 values are multiplied by \( (1 + \delta \cdot k) \), where \( \delta \in \{0.003, 0.005, 0.010, 0.015\} \) is the drift rate and \( k \) is the block index. This simulates the gradual loss of sensor calibration caused by physical fouling (e.g., dust accumulation on an optical lens), a phenomenon well-documented in long-term environmental monitoring deployments (Gama et al., 2014).

**RF Response:** Calibration drift is particularly insidious for tree-based models. Decision tree splits are defined as threshold comparisons (e.g., \( \text{PM10} > 120 \)); as the actual PM10 values inflate systematically, observations begin crossing split thresholds that were learned on clean data. Trees trained on the baseline distribution assign records to incorrect leaf nodes, producing predictions that are systematically biased. Unlike stuck sensors, calibration drift affects every observation in which PM10 is used as a split variable, and the cumulative error grows monotonically with the drift rate.

**Ledger Observations:** The drifted R² declines from approximately 0.598 at \( \delta = 0.003 \) to 0.502 at \( \delta = 0.015 \)—the most severe performance degradation of all four drift types. Despite this substantial model degradation, KS test p-values on the AQI target remain at 1.0 across all B-type experiments, and detection latency equals the total dataset length (18,265 rows). This demonstrates that a KS test applied only to the model *output* will fail to detect calibration drift even when the model has lost approximately one-third of its predictive power—a critical limitation for real-world monitoring systems.

### 3.3 Instance C — Blackout (Missing Not At Random)

**Definition:** PM2.5 values above a threshold \( \tau \) (set at the 60th, 70th, 80th, and 90th percentiles of the PM2.5 distribution) are probabilistically replaced with NaN, after which the missing values are imputed using the feature mean. This simulates a "blackout" failure in which hardware overloads under high-pollution conditions, causing the sensor to report no reading precisely when air quality is worst—a Missing Not At Random (MNAR) mechanism in the sense of Little and Rubin (2002).

**RF Response:** Mean imputation under MNAR collapses the upper tail of the PM2.5 distribution: the most extreme and epidemiologically significant values are replaced with a value close to the centre of the distribution. Because decision trees in the forest use splits to isolate high-PM2.5 regions of the feature space, the removal of these extreme values causes the forest to lose its ability to discriminate dangerous pollution episodes from moderate ones. The model may continue to perform adequately in the majority of the input space while silently failing on the most safety-critical subset of observations.

**Ledger Observations:** Paradoxically, the drifted R² in some C-type experiments is nearly equal to or marginally above the baseline (e.g., 0.676 at the 90th-percentile threshold versus 0.684 baseline). This occurs because mean imputation reduces the variance in the target variable associated with extreme PM2.5 events, making the regression problem locally "easier." However, this statistical improvement masks a practical failure: the model can no longer identify the highest-risk air quality events. KS p-values remain at 1.0 and detection latency equals the full dataset length, confirming that this drift type is entirely invisible to output-based monitoring.

### 3.4 Instance D — Concept Drift

**Definition:** The AQI target variable is shifted by a constant \( \Delta \in \{5, 10, 20, 30\} \) while all input features remain unchanged. This simulates concept drift—a change in the functional relationship between inputs and the prediction target—which can arise in practice from revisions to the regulatory formula used to compute AQI, changes in the chemical composition of pollution (e.g., emergence of new industrial emission sources), or long-term environmental transitions (Gama et al., 2014).

**RF Response:** Concept drift represents the most direct challenge to a frozen model. The forest has internalized a mapping \( f: X \to \hat{y} \), where \( \hat{y} \approx y_{\text{train}} \). When the true target shifts to \( y + \Delta \), every prediction produced by the forest is systematically low by approximately \( \Delta \) units, regardless of the quality of the input features. Unlike the other drift types, concept drift directly alters the *output* distribution, making it detectable through distributional tests.

**Ledger Observations:** This is the only drift type for which the KS test p-value falls below conventional significance thresholds. At \( \Delta = 10 \), the p-value is approximately \( 1.2 \times 10^{-4} \); at \( \Delta = 20 \), it drops to \( 1.1 \times 10^{-15} \); at \( \Delta = 30 \), it reaches \( 1.2 \times 10^{-33} \). Detection latency decreases sharply with shift magnitude: at \( \Delta = 30 \), drift is detected within 400 rows regardless of block size. The drifted R² also declines with \( \Delta \), reaching approximately 0.643 at the largest shift. This confirms that concept drift, among the four instances, is the scenario for which KS-based output monitoring is most effective.

---

## 4. Silent Failures in Ensemble Models

The experimental results presented above reveal a structural limitation of using aggregate performance metrics to monitor deployed ensemble models. For three of the four drift types (A, B, and C), the KS test on the output distribution fails entirely to detect degradation, and detection latency equals the full length of the data stream. This phenomenon is referred to in the literature as a **silent failure**: the model continues operating, and its aggregate statistics appear acceptable, while its predictions have become systematically unreliable (Kuberski, 2022; Gama et al., 2014).

Two mechanisms explain why random forests are particularly prone to silent failures under covariate drift:

1. **Ensemble averaging absorbs local errors.** When a subset of trees degrades due to a distributional shift in one or more input features, the remaining trees—trained on different bootstrap samples and relying on different feature subsets—continue to produce reasonable predictions. The average output of the forest changes only modestly, even when a significant fraction of trees have lost their predictive validity for the affected observations. This is the same variance-reduction mechanism that makes random forests robust under normal conditions, but it also suppresses the signal of gradual degradation.

2. **Output monitoring is blind to input distribution changes.** KS tests and similar distribution tests applied to model *outputs* can only detect shifts in the predicted or true target variable. Covariate drift—changes in the distribution of input features, as in scenarios A, B, and C—does not necessarily manifest as a shift in the output distribution, particularly when the affected features are partially redundant with other predictors. As Gama et al. (2014) note, "changes in the input distribution may leave the conditional distribution of the target variable approximately unchanged, making drift invisible to output-only monitors."

The practical implication is that robust monitoring of a deployed random forest requires a multi-layered strategy:

- **Feature-level distribution monitoring:** Apply statistical tests (e.g., KS test, Population Stability Index) independently to each input feature. Calibration drift in PM10 or a stuck sensor in PM2.5 would be detectable immediately through per-feature monitoring, even when the output distribution is stable.
- **Residual monitoring:** Track the distribution of model residuals \( (y - \hat{y}) \) over time using control-chart methods. Systematic bias in residuals is an early indicator of concept drift.
- **Domain-informed alerts:** Integrate physical plausibility constraints. A PM2.5 sensor reporting the same value for an extended period is mechanically implausible and should trigger a hardware inspection, independent of any statistical test on the model output.

The results of this project empirically demonstrate the gap between statistical detectability and practical model reliability. A system that monitors only model outputs would miss three out of four degradation scenarios simulated here, including the one (calibration drift) that produces the largest reduction in R².

---

## 5. References

1. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140. https://doi.org/10.1007/BF00058655

2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

3. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), Article 44. https://doi.org/10.1145/2523813

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

5. Kuberski, W. (2022). *How to detect silent failures in ML models*. Conf42 Machine Learning Conference. https://www.conf42.com/Machine_Learning_2022_Wojtek_Kuberski_detect_silent_failures_ml_models

6. Lee, T.-H., Ullah, A., & Wang, R. (2019). Bootstrap aggregating and random forest. In *Macroeconomic Forecasting in the Era of Big Data*, Springer. https://doi.org/10.1007/978-3-030-31150-6_19

7. Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.

8. Srivastava, A., et al. (2023). Prediction of Air Quality Index using Random Forest Algorithm. *Academia.edu*. Retrieved from https://www.academia.edu/103758903

9. ECON484 Course Handout — *Model Decay and Cold Start* (Spring 2026). 

10. ECON484 Homework Assignment 3 — *The Entropy of Intelligence & Drift Detection* (Spring 2026). 

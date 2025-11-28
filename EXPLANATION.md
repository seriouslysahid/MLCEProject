# EXPLANATION of Notebook Code Cells

This document explains each code cell in `notebooks/notebook.ipynb`. For each code cell I provide: a short title, purpose, main inputs, main outputs, and important notes or assumptions.

---

Code Cell 1 — Imports & environment setup
- Purpose: Import all required Python libraries and set display / plotting options.
- Inputs: none (library imports only).
- Outputs: modules in memory; prints `Libraries imported successfully`.
- Notes: suppresses warnings and configures pandas/seaborn/matplotlib display options. Requires the packages listed in `requirements.txt`.

Code Cell 2 — Basic configuration
- Purpose: Define a `CONFIG` dictionary with random seed, chunk size, CV folds, and other parameters; create `'/content/data'` and `'/content/outputs'` directories.
- Inputs: none (writes filesystem paths and sets random seed via numpy).
- Outputs: `CONFIG` variable, directories created, prints `Configuration loaded`.
- Notes: Paths use `/content` (Colab-style). If running locally, adapt paths or ensure `/content` exists or change to project-relative paths.

Code Cell 3 — Helper functions and utilities
- Purpose: Define helper functions used throughout the notebook (path helpers, metric computation, training/evaluation wrapper, scaling, dtype optimization, outlier detection, and confusion matrix saving).
- Inputs: none (defines functions and imports additional metric utilities such as `time`).
- Outputs: functions available in the environment (e.g., `compute_binary_metrics`, `train_and_evaluate`, `scale_data_robust`, `optimize_dtypes`, `detect_outliers_iqr`, `save_confusion_matrix`).
- Notes: These functions centralize repeated logic (metric calculations, robust scaling, I/O helpers) used by later cells.

Code Cell 4 — Kaggle credentials placeholder
- Purpose: Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables for dataset download (Colab environment example).
- Inputs: user must replace `<YOUR_KAGGLE_USERNAME>` and optionally update the key.
- Outputs: environment variables set; prints confirmation.
- Notes: Never commit real API keys; prefer secure secrets management. The notebook includes a dummy key — replace with your own.

Code Cell 5 — Dataset download and verification
- Purpose: Define a robust dataset verification function, check for existing CSVs in `../data/CICIDS2017`, and if missing/invalid attempt to download via `kagglehub` and copy CSVs into the target directory.
- Inputs: `kagglehub` dataset download, `target_dir = ../data/CICIDS2017` on disk.
- Outputs: CSV files copied into `../data/CICIDS2017` (if downloaded); final verification result printed.
- Notes: This cell implements multiple sanity checks (file count, per-file size, total size range). It raises an error if verification fails.

Code Cell 6 — Load and concatenate CSV files
- Purpose: Read all CSV files from the data directory in chunks (to avoid memory issues), concatenate into a single DataFrame, perform initial cleaning (strip column names, drop duplicates, replace infinities, optimize dtypes).
- Inputs: CSV files in `../data/CICIDS2017`, `CONFIG['CHUNK_SIZE']`.
- Outputs: `df` DataFrame containing all concatenated rows; prints counts and columns.
- Notes: Uses `pd.read_csv(..., chunksize=...)` to limit memory usage; runs `optimize_dtypes` to reduce memory footprint.

Code Cell 7 — Display first rows
- Purpose: Show the first 5 rows of `df` for quick inspection.
- Inputs: `df`.
- Outputs: visual display of `df.head()` in notebook output.
- Notes: No mutation.

Code Cell 8 — Dataset info (dtypes & missing)
- Purpose: Print `df.info()` to show dtypes and non-null counts.
- Inputs: `df`.
- Outputs: `df.info()` console output.
- Notes: Useful to identify problematic columns (object types, large floats).

Code Cell 9 — Print column names
- Purpose: Clean column names again and print an enumerated list of column names.
- Inputs: `df.columns`.
- Outputs: printed list of column names.
- Notes: Ensures whitespace was removed from column labels.

Code Cell 10 — Missing values summary
- Purpose: Compute missing-value counts and percentages; present a sorted DataFrame of columns with missing values.
- Inputs: `df`.
- Outputs: `missing_df` DataFrame and printed totals.
- Notes: `missing_df` is used later for imputation decisions.

Code Cell 11 — Summary statistics (sampled describe)
- Purpose: Produce summary statistics by sampling `n=100000` rows and running `.describe()` to speed up computation on large datasets.
- Inputs: `df.sample(n=100000, random_state=42)`.
- Outputs: printed `.describe().T` of the sample (mean, std, min, max, etc.).
- Notes: Sampling makes quick EDA feasible on large datasets.

Code Cell 12 — Target variable (binary) analysis
- Purpose: Identify the target column (`' Label'` or `'Label'`), create a temporary `Binary_Class_Display` column that maps all non-BENIGN labels to `ATTACK`, and print distribution counts.
- Inputs: `df[target_col]`.
- Outputs: `Binary_Class_Display` temporary column, `class_summary` DataFrame printed.
- Notes: This groups all attack types into a single `ATTACK` class for binary classification.

Code Cell 13 — Visualize binary target distribution
- Purpose: Create bar and pie plots showing `BENIGN` vs `ATTACK` counts/percentages and save the figure to `outputs/target_distribution.png`.
- Inputs: `class_dist` computed earlier; output helper `get_output_path`.
- Outputs: saved image `outputs/target_distribution.png` and displayed plot.
- Notes: The cell creates `'/content/outputs'` earlier; if running locally, ensure `outputs/` exists in the path used by `get_output_path`.

Code Cell 14 — Remove temporary display column
- Purpose: Clean up the temporary `Binary_Class_Display` column added for visualization.
- Inputs: `df`.
- Outputs: `df` without `Binary_Class_Display`.
- Notes: Prevents leakage into feature set.

Code Cell 15 — Univariate outlier detection (IQR) for top features
- Purpose: Compute IQR-based outlier bounds for a sample of numeric features (top 20 by variance), summarize outlier counts/percentages, and save `outputs/outlier_analysis.csv`.
- Inputs: numeric columns from `df`; `CONFIG['CHUNK_SIZE']` not needed here.
- Outputs: `outlier_df` DataFrame, saved CSV `outputs/outlier_analysis.csv`.
- Notes: The cell prioritizes top features by variance to reduce runtime; it records Lower/Upper bounds per feature for later capping.

Code Cell 16 — Visualize outliers (boxplots)
- Purpose: Plot boxplots for the top 6 features by outlier percentage and save `outputs/outlier_boxplots.png`.
- Inputs: `outlier_df` and `df` (sampled for plotting).
- Outputs: saved image `outputs/outlier_boxplots.png` and displayed plots.
- Notes: Sampling is used for speed; the plot helps to visually inspect extreme values.

Code Cell 17 — Correlation analysis (compute correlations & high-corr pairs)
- Purpose: Select top 20 numeric features by variance, compute a correlation matrix on a sample, find highly correlated pairs with |corr| > 0.7, and save `outputs/high_correlations.csv`.
- Inputs: `df` (sampled), `top_20_features` list.
- Outputs: `corr_matrix`, `high_corr_df` (saved CSV if pairs found).
- Notes: High-correlation pairs may indicate redundancy; results are used for feature selection guidance.

Code Cell 18 — Correlation heatmap plot
- Purpose: Plot the correlation heatmap for the selected top 20 features and save `outputs/correlation_heatmap.png`.
- Inputs: `corr_matrix` computed earlier.
- Outputs: saved image `outputs/correlation_heatmap.png` and displayed plot.
- Notes: Heatmap opts to avoid annotations (20x20 would be cluttered).

Code Cell 19 — Distribution analysis (histograms)
- Purpose: Plot histograms of nine numerical features to inspect distributions and skewness; save `outputs/feature_distributions.png`.
- Inputs: `numerical_cols` and `df`.
- Outputs: saved image `outputs/feature_distributions.png` and displayed plots.
- Notes: The cell picks the first nine numerical columns; adjust selection for other features if desired.

Code Cell 20 — Target column sanity checks
- Purpose: Print the `target_col`, classes, and number of unique classes (reminder before converting to binary).
- Inputs: `df`.
- Outputs: printed target column info.
- Notes: Verifies expected label column name and values.

Code Cell 21 — Handle missing values (drop & impute)
- Purpose: Drop columns with >50% missing values, then impute remaining missing values with per-column median. Avoid modifying the target.
- Inputs: `missing_df` from earlier and `df`.
- Outputs: updated `df` with fewer columns and no (or fewer) missing values.
- Notes: Strategy chosen for this notebook; different datasets may require alternative imputation.

Code Cell 22 — Outlier handling (Winsorization / capping)
- Purpose: Load `outputs/outlier_analysis.csv`, select features with >10% outliers, and cap values to computed lower/upper bounds.
- Inputs: `outputs/outlier_analysis.csv`; `df`.
- Outputs: `df` with capped values for specified features.
- Notes: If `outlier_analysis.csv` is missing, the step is skipped with a warning.

Code Cell 23 — Feature engineering (new features)
- Purpose: Create three derived features: `Packet_Rate`, `Byte_Rate`, and `Packet_Size_Ratio` when required source columns exist.
- Inputs: `Flow Duration`, `Total Fwd Packets`, `Total Length of Fwd Packets`, `Total Length of Bwd Packets` (if present).
- Outputs: new columns added to `df` and printed confirmation.
- Notes: Each new feature uses +1 in denominators to avoid division-by-zero.

Code Cell 24 — Encode target to binary
- Purpose: Convert the target to `Label_Binary` (0 for BENIGN, 1 for ATTACK) and save a JSON mapping `outputs/label_mapping_binary.json`.
- Inputs: `df[target_col]`.
- Outputs: `Label_Binary` column in `df`; saved mapping file.
- Notes: This is the canonical binary target used downstream for modeling.

Code Cell 25 — Encode other categorical variables
- Purpose: Find other object-type columns and apply `LabelEncoder` to convert them to numeric codes.
- Inputs: `df`.
- Outputs: encoded categorical columns in `df` (if any) and printed list.
- Notes: The cell excludes the original target column from encoding.

Code Cell 26 — Separate features (`X`) and target (`y`)
- Purpose: Build feature matrix `X` (drop original and binary target where appropriate) and target vector `y` = `Label_Binary`.
- Inputs: `df`.
- Outputs: `X` (DataFrame), `y` (Series); printed shapes.
- Notes: Ensures the modeling code doesn't accidentally include label columns as features.

Code Cell 27 — Stratified train-test split (70-30)
- Purpose: Split `X`/`y` into stratified training and test sets preserving class ratios; delete temporary large variables to free memory and save `feature_names` for later.
- Inputs: `X`, `y`, `random_state=42`, `test_size=0.3`.
- Outputs: `X_train`, `X_test`, `y_train`, `y_test`, `feature_names` list; prints dataset sizes.
- Notes: Uses `stratify=y` to ensure balanced class distribution in both splits.

Code Cell 28 — Feature scaling (StandardScaler via utility)
- Purpose: Fit `StandardScaler` on training data using `scale_data_robust`, transform both training and test sets, preventing data leakage.
- Inputs: `X_train`, `X_test`.
- Outputs: `X_train_scaled` (numpy array), `scaler` object, `X_test_scaled` (numpy array).
- Notes: The helper function replaces infinities and fills NaNs before scaling.

Code Cell 29 — Save processed datasets (Feather format)
- Purpose: Convert scaled numpy arrays back to DataFrames and save `X_train.feather`, `X_test.feather`, `y_train_binary.feather`, `y_test_binary.feather` to `outputs/` for fast reload later.
- Inputs: `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`, `feature_names`.
- Outputs: Feather files under `outputs/` and printed file summaries.
- Notes: Feather is preferred here for speed and dtype preservation.

Code Cell 30 — Load preprocessed data (Feather) for modeling
- Purpose: Demonstrate fast loading of the saved feather files (reads `X_train/X_test` and `y_train/y_test`), convert DataFrames to numpy arrays for model training.
- Inputs: Feather files created earlier.
- Outputs: `X_train`, `X_test`, `y_train`, `y_test`, `feature_names_loaded` and printed memory usage.
- Notes: This cell rehydrates a saved preprocessing state so training can be run independently of earlier preprocessing steps.

Code Cell 31 — Print input/output summary
- Purpose: Print a summary describing the problem type, number of features, sample feature names, target info, and models to implement.
- Inputs: `X_train_scaled`, `feature_names_loaded`, `y_train`, `label_mapping`.
- Outputs: descriptive console output.
- Notes: Useful human-readable summary to confirm data shape and intended models.

Code Cell 32 — Model 1: Logistic Regression training & evaluation
- Purpose: Scale features (again) specifically for logistic regression, train a `LogisticRegression` with optimized options, predict on test set, compute metrics via `compute_binary_metrics`, and save confusion matrix image.
- Inputs: `X_train`, `X_test`, `y_train`, `y_test` (uses `StandardScaler` and `LogisticRegression` settings tuned for speed).
- Outputs: fitted `lr_model`, `lr_metrics` dictionary with Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrix saved as `outputs/confusion_matrix_lr.png`.
- Notes: Uses `solver='saga'` and relaxed tolerances for faster training; ensures `predict_proba` is used where available.

Code Cell 33 — Model 2: Linear SVM approximation via `SGDClassifier`
- Purpose: Train an SVM-like classifier using `SGDClassifier` (hinge loss), predict, compute metrics, and save confusion matrix image.
- Inputs: `X_train`, `X_test`, `y_train`, `y_test` scaled separately for SVM.
- Outputs: fitted `svm_model`, `svm_metrics`, saved `outputs/confusion_matrix_svm.png`.
- Notes: Uses `decision_function` as a score for ROC-AUC; `SGDClassifier` is chosen for scalability and speed.

Code Cell 34 — Model 3: PCA + Logistic Regression pipeline
- Purpose: Build and train a pipeline that performs PCA (n_components=20) followed by LogisticRegression; predict, compute metrics, and save confusion matrix.
- Inputs: `X_train`, `X_test`, `y_train`, `y_test`, PCA pipeline configuration.
- Outputs: `pca_lr_pipeline` fitted, `pca_lr_metrics`, saved `outputs/confusion_matrix_pca_lr.png`.
- Notes: PCA reduces dimensionality before logistic regression to speed up training and reduce noise.

Code Cell 35 — Model comparison table
- Purpose: Aggregate `lr_metrics`, `svm_metrics`, and `pca_lr_metrics` into a `comparison_df`, print it, determine the best model by F1-Score, and save `outputs/model_comparison.csv`.
- Inputs: metrics dictionaries from the three trained models.
- Outputs: `comparison_df` and saved CSV.
- Notes: This is the canonical results table used to pick the final model.

Code Cell 36 — Plot confusion matrices for all models
- Purpose: Create a 1×3 plot showing confusion matrices side-by-side for the three models and save `outputs/confusion_matrices.png`.
- Inputs: `lr_metrics`, `svm_metrics`, `pca_lr_metrics` (each contains a `Confusion Matrix`).
- Outputs: saved image and displayed heatmaps.
- Notes: Helpful visual comparison across models.

Code Cell 37 — ROC curves for all models
- Purpose: Compute and plot ROC curves for each model (using `predict_proba` or `decision_function` as available), save `outputs/roc_curves.png`.
- Inputs: models and their respective test feature matrices (`X_test_lr`, `X_test_svm_scaled`, `X_test_pca_scaled`) and `y_test`.
- Outputs: saved ROC figure and printed confirmation.
- Notes: ROC-AUC computed via `roc_auc_score` for each model; ensure `y_proba` is a 1D array representing positive-class scores.

Code Cell 38 — (Short) Save comparison table
- Purpose: Save `comparison_df` again to `outputs/model_comparison.csv` (idempotent save).
- Inputs/Outputs: same as Code Cell 35.
- Notes: Redundant but ensures the CSV is present.

Code Cell 39 — Generate prediction probability results and plot distributions (first variant)
- Purpose: Build `prediction_results` dict for each model containing train/test probabilities and actual labels, normalize SVM decision scores via `MinMaxScaler`, and plot train/test predicted-probability distributions for each model; save `outputs/prediction_distributions.png`.
- Inputs: fitted models and scaled training/test sets (the earlier computed arrays like `X_train_lr`, `X_test_lr`, etc.).
- Outputs: `prediction_results` and saved figure.
- Notes: Visualizes how well model probabilities separate classes (good separation = clear peaks near 0 and 1 for respective classes).

Code Cell 40 — Generate prediction probability results (second variant)
- Purpose: Another cell that prepares `prediction_results` and demonstrates generation of model-specific probability arrays; essentially duplicates Code Cell 39's behavior with explicit printouts.
- Inputs/Outputs: same as Code Cell 39.
- Notes: Appears twice in the notebook (two similar cells) — both prepare predictions for distribution analysis.

Code Cell 41 — Cross-validation (5-fold, sampled)
- Purpose: Perform 5-fold stratified cross-validation on ~10% of training data for each model to get mean F1 and a simple empirical 95% CI estimate; save `outputs/cv_results_table.csv`.
- Inputs: sampled training arrays (`X_train_sample_*`) and `y_train_sample`.
- Outputs: `cv_df` DataFrame and saved CSV `outputs/cv_results_table.csv`.
- Notes: Sampling reduces compute time while still giving a representative estimate. Uses `cross_val_score(..., scoring='f1')`.

Code Cell 42 — Visualize CV results
- Purpose: Plot the mean CV F1 scores with error bars (CI_95) for the three models and save `outputs/cv_results.png`.
- Inputs: `cv_df` from previous cell.
- Outputs: saved image `outputs/cv_results.png`.
- Notes: Visualizes cross-validation stability and differences between models.

---

If you want, I can:
- Add inline references to each code-cell number inside `notebooks/notebook.ipynb` as comments, or
- Generate a simplified summary mapping markdown headings in the notebook to the code cells, or
- Convert this explanation into a notebook cell-by-cell JSON file (per your notebook-format instructions) so that each explanation appears as a markdown cell next to the corresponding code cell.

Which of these next steps would you like? If you prefer the EXPLANATION as a notebook with paired markdown cells, tell me and I will create the JSON-formatted notebook cells per your instructions.
# Detailed Technical Explanation: Network Intrusion Detection System

This document provides an in-depth, cell-by-cell technical analysis of the `NIDS_Complete_Workflow_Final.ipynb` notebook. It goes beyond code comments to explain the *mathematical intuition*, *engineering justifications*, and *design patterns* used in this project.

---

## 1. Introduction and Setup

### Problem Statement & Motivation
The notebook begins by defining the objective: building a Network Intrusion Detection System (NIDS) using the CIC-IDS2017 dataset.
-   **Why NIDS?** Modern cyber threats are sophisticated. Traditional firewalls (signature-based) cannot detect new, unknown attacks ("zero-day" exploits). Machine Learning (ML) allows us to detect *anomalies*—patterns that deviate from normal behavior—potentially catching new threats.
-   **Why CIC-IDS2017?** It is a high-quality, realistic dataset containing benign traffic and common attacks like DDoS, PortScan, and Botnet traffic, providing a robust benchmark for NIDS.

### Library Imports
We import essential Python libraries:
-   **`numpy` & `pandas`**: For efficient numerical computation and data manipulation (DataFrames).
-   **`matplotlib` & `seaborn`**: For data visualization (plots, heatmaps).
-   **`sklearn` (Scikit-Learn)**: The core ML library used for:
    -   `model_selection`: Splitting data (`train_test_split`).
    -   `preprocessing`: Scaling data (`StandardScaler`).
    -   `linear_model`: Implementing Logistic Regression.
    -   `svm`: Implementing Support Vector Machines (via `SGDClassifier`).
    -   `decomposition`: Dimensionality reduction (`PCA`).
    -   `metrics`: Evaluating performance (Accuracy, F1-Score, ROC-AUC).
-   **`kagglehub`**: A specific API tool to download datasets directly from Kaggle.

### Configuration
We define a `CONFIG` dictionary to centralize parameters:
-   `SEED`: Ensures **reproducibility**. Setting a random seed (e.g., 42) guarantees that random operations (like data splitting) yield the same result every time the code is run.
-   `TEST_SIZE`: Set to 0.3 (30%), meaning 70% of data is for training and 30% for testing.
-   `TARGET`: 'Label' is the column we want to predict.

---

## 2. Data Loading and Verification

### Dataset Download
**Code**: `kagglehub.dataset_download("cic-dataset/cic-ids-2017")`
-   **Explanation**: Automates the retrieval of the dataset. This ensures the code is portable and doesn't rely on manual file placement.

### Data Loading Strategy
**Code**: `pd.read_csv(file, dtype=dtype_dict)`
-   **Optimization**: The dataset is massive (>2GB). We specify data types (`dtype`) during loading (e.g., forcing `float32` instead of `float64`) to reduce memory usage by ~50%.
-   **Concatenation**: The dataset comes in multiple CSV files (one per day of capture). We loop through them and combine them into a single DataFrame using `pd.concat`.

---

## 3. Exploratory Data Analysis (EDA)

### Initial Inspection
**Code**: `df.head()`, `df.info()`, `df.describe()`
-   **Purpose**: To understand the data structure.
    -   `head()` shows sample rows.
    -   `info()` reveals missing values and data types.
    -   `describe()` provides statistical summaries (mean, std, min, max) for each feature.

### Correlation Analysis
**Code**: `sns.heatmap(correlation_matrix)`
-   **Technical Term**: **Correlation** measures the linear relationship between two variables. A value of 1.0 means they move perfectly together; -1.0 means they move in opposite directions.
-   **Insight**: We identify "redundant" features—pairs with very high correlation (e.g., `Total Fwd Packets` and `Subflow Fwd Packets`). Redundant features add computational cost without adding new information (multicollinearity).

### Class Distribution
**Code**: `df['Label'].value_counts()`
-   **Observation**: The dataset is **imbalanced**. 'BENIGN' traffic vastly outnumbers attack traffic.
-   **Implication**: A naive model could achieve 80%+ accuracy just by predicting "BENIGN" for everything. We must use metrics like **F1-Score** (which balances precision and recall) rather than just accuracy to evaluate performance fairly.

---

## 4. Data Preprocessing

### Handling Missing Values
**Code**: `SimpleImputer(strategy='median')`
-   **Problem**: Real-world data often has gaps (`NaN`). ML models generally cannot handle missing inputs.
-   **Solution**: We replace missing values with the **median** of that column.
-   **Justification**: We use median instead of mean because the mean is highly sensitive to outliers (extreme values), whereas the median is robust.

### Outlier Detection & Handling
**Code**: `IQR Method` and `Winsorization`
-   **Technical Term**: **IQR (Interquartile Range)** is the range between the 25th percentile (Q1) and 75th percentile (Q3). Outliers are defined as points falling below $Q1 - 1.5*IQR$ or above $Q3 + 1.5*IQR$.
-   **Action**: We "cap" (Winsorize) extreme values to the upper/lower bounds.
-   **Justification**: Network attacks often manifest as extreme bursts of traffic. However, *statistical* outliers can also be errors. Capping preserves the "extremeness" without allowing a single infinite value to ruin the model's calculations.

### Feature Scaling
**Code**: `StandardScaler()`
-   **Action**: Transforms data to have Mean = 0 and Variance = 1.
-   **Justification**:
    -   Features have different units (e.g., "Duration" in seconds vs. "Bytes" in millions).
    -   Linear models (Logistic Regression, SVM) are sensitive to scale. Without scaling, the model would unfairly prioritize features with larger raw numbers.
-   **Critical Detail**: We fit the scaler on **Training Data ONLY** and then apply it to Test Data. This prevents **Data Leakage**—accidentally using information from the test set (like its mean) during training.

---

## 5. Model Training

### Model 1: Logistic Regression
-   **Mathematical Intuition**: Logistic Regression estimates the probability $P(y=1|X)$ using the sigmoid function:
    $$ P(y=1|X) = \frac{1}{1 + e^{-(w^T X + b)}} $$
    where $w$ are the weights and $b$ is the bias. It finds the "best fit" S-curve that separates the classes.
-   **Configuration**:
    -   `solver='saga'`: An algorithm optimized for large datasets.
    -   `class_weight='balanced'`: Automatically adjusts weights inversely proportional to class frequencies, helping the model pay more attention to the minority (Attack) class.

### Model 2: Linear SVC (SGDClassifier)
-   **Mathematical Intuition**: A Support Vector Machine (SVM) finds the optimal hyperplane (line in high dimensions) that maximizes the **margin** (distance) between the two classes.
    -   **Hinge Loss**: The loss function used is $max(0, 1 - y(w^T x + b))$. It penalizes points that are on the wrong side of the margin.
-   **Optimization**: We use `SGDClassifier` with `loss='hinge'`. This implements a Linear SVM using Stochastic Gradient Descent, which is much faster than traditional SVM solvers (`SVC`) for millions of samples.

### Model 3: PCA + Logistic Regression
-   **Technical Term**: **PCA (Principal Component Analysis)** is a dimensionality reduction technique. It creates new, uncorrelated features ("principal components") that capture the maximum variance in the data.
-   **Action**: We reduce the 78 original features down to a smaller set that explains 95% of the variance, then feed this into Logistic Regression.
-   **Goal**: To see if we can achieve similar accuracy with less data (faster training).

---

## 6. Evaluation and Comparison

### Metrics
We evaluate models using:
-   **Accuracy**: Overall % correct. (Can be misleading for imbalanced data).
-   **Precision**: Of all predicted attacks, how many were actually attacks? (Low precision = False Alarms).
-   **Recall**: Of all actual attacks, how many did we detect? (Low recall = Missed Threats).
-   **F1-Score**: The harmonic mean of Precision and Recall. The best single metric for this problem.
    $$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} $$
-   **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve. Measures the model's ability to distinguish between classes across all threshold settings.

### Calibration Analysis
-   **Concept**: A well-calibrated model produces probabilities that reflect reality. If it predicts "80% chance of attack" for 100 events, ~80 of them should actually be attacks.
-   **Brier Score**: Measures the accuracy of probabilistic predictions (lower is better).
    $$ Brier = \frac{1}{N} \sum_{t=1}^{N} (f_t - o_t)^2 $$
    where $f_t$ is the forecast probability and $o_t$ is the outcome (0 or 1).
-   **Result**: Our models showed good calibration, validating their reliability for real-world risk assessment.

---

## 7. Conclusion
The **Linear SVC** proved to be the best model, offering the highest F1-Score and Accuracy. This suggests that the boundary between benign and malicious traffic in this feature space is largely linear. The project successfully demonstrates a complete ML pipeline from raw data to a deployable, high-performance intrusion detection model.

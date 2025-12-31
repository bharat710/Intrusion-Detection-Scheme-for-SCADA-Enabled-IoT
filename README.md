# DNP3 Intrusion Detection System

A comprehensive machine learning pipeline for detecting intrusions in DNP3 (Distributed Network Protocol 3) industrial control systems. This project implements data aggregation, preprocessing, feature selection, and multi-model evaluation to achieve high-accuracy intrusion detection.

## Project Overview

This system processes network traffic data from DNP3 protocols, extracts relevant features, and trains multiple machine learning models to classify traffic as normal or malicious. The pipeline handles imbalanced datasets and evaluates model performance across different intrusion rate scenarios.

## Features

- **Comprehensive Data Aggregation**: Combines data from multiple attack scenarios and parsers (CICFlowMeter and Custom DNP3 Parser)
- **Robust Preprocessing**: Handles missing values, infinite values, and mixed data types
- **Intelligent Feature Selection**: Reduces dimensionality to 99 most informative features using statistical methods
- **Multi-Model Training**: Evaluates Decision Tree, Random Forest, XGBoost, and CatBoost classifiers
- **Imbalanced Data Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance training data
- **Multiple Intrusion Rate Testing**: Evaluates models at 5%, 10%, and 15% intrusion rates
- **Comprehensive Evaluation**: Generates classification reports, confusion matrices, and performance comparison charts

## Project Structure

```
DNP3_Intrusion_Detection/
├── aggregate_data.py              # Phase 1: Data aggregation from raw files
├── preprocess_data.py             # Phase 2: Data cleaning and preprocessing
├── feature_selection.py           # Phase 3: Feature reduction to 99 features
├── train_evaluate_models.py       # Phase 4: Model training and evaluation
├── verify_features.py             # Utility to verify selected features
├── run_full_project.py            # Complete pipeline in single script
└── DNP3_Intrusion_Detection_Dataset_Final/  # Raw data directory
    ├── Attack_Command_Response_Flood/
    ├── Attack_DNS_Flood/
    ├── Attack_HTTP_Flood/
    ├── Attack_ICMP_Flood/
    ├── Attack_MITM/
    ├── Attack_TCP_Flood/
    ├── Attack_UDP_Flood/
    ├── Attack_UDP_Lag/
    ├── Attack_Unsolicited_Response/
    └── Training_Testing_Balanced_CSV_Files/
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost catboost matplotlib seaborn tqdm
```

Or install all dependencies at once:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost catboost matplotlib seaborn tqdm
```

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Execute the entire pipeline from data aggregation to model evaluation:

```bash
python run_full_project.py
```

This single script will:
1. Aggregate all raw CSV files
2. Preprocess and clean the data
3. Select the top 99 features
4. Train and evaluate all models
5. Generate performance reports and visualizations

### Option 2: Run Individual Phases

Execute each phase separately for more control:

#### Phase 1: Data Aggregation
```bash
python aggregate_data.py
```
Output: `master_dataset_all_files.csv`

#### Phase 2: Data Preprocessing
```bash
python preprocess_data.py
```
Output: `dataset_processed.csv`, `label_mapping.json`

#### Phase 3: Feature Selection
```bash
python feature_selection.py
```
Output: `dataset_final_for_training.csv`

#### Phase 4: Model Training & Evaluation
```bash
python train_evaluate_models.py
```
Output: `final_results_summary.csv`, confusion matrices, performance charts

### Verify Selected Features

To view the 99 selected features:

```bash
python verify_features.py
```

## Configuration

### Key Parameters

Modify these parameters in the scripts or `run_full_project.py`:

```python
# Number of features to select
NUMBER_OF_FEATURES_TO_SELECT = 99

# Intrusion rate scenarios to test
INTRUSION_RATES = [0.05, 0.10, 0.15]  # 5%, 10%, 15%

# Models to evaluate
MODELS = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, 
                            eval_metric='mlogloss', n_jobs=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=100, 
                                   allow_writing_files=False)
}
```

## Pipeline Details

### Phase 1: Data Aggregation

- Recursively searches for CSV files in attack folders
- Extracts metadata (attack type, timeout, parser type)
- Combines all files using outer join to preserve all columns
- Handles two different parser outputs (CICFlowMeter and Custom DNP3 Parser)

### Phase 2: Data Preprocessing

- Cleans column names (removes special characters, standardizes format)
- Creates final label: "Normal" for benign traffic, attack type for malicious traffic
- Encodes labels and saves mapping to JSON file
- Converts all object columns to numeric format
- Removes identifier columns (IPs, timestamps, flow IDs)
- Handles infinite values and missing data

### Phase 3: Feature Selection

- Applies StandardScaler for feature normalization
- Removes zero-variance features using VarianceThreshold
- Selects top 99 features using SelectKBest with ANOVA F-value scoring
- Ensures features are statistically significant for classification

### Phase 4: Model Training & Evaluation

For each intrusion rate (5%, 10%, 15%):
1. Creates imbalanced training set with specified attack/normal ratio
2. Applies SMOTE to balance the training data
3. Trains all four models
4. Evaluates on held-out test set (30% of original data)
5. Generates classification reports and confusion matrices

## Output Files

| File | Description |
|------|-------------|
| `master_dataset_all_files.csv` | Aggregated raw data from all sources |
| `dataset_processed.csv` | Cleaned and preprocessed data |
| `label_mapping.json` | Mapping of numeric labels to attack types |
| `dataset_final_for_training.csv` | Final dataset with 99 selected features |
| `final_results_summary.csv` | Performance metrics for all models and rates |
| `final_performance_comparison.png` | Bar chart comparing model accuracies |
| `confusion_matrix_xgb_5pct.png` | XGBoost confusion matrix at 5% intrusion |
| `confusion_matrix_xgb_10pct.png` | XGBoost confusion matrix at 10% intrusion |
| `confusion_matrix_xgb_15pct.png` | XGBoost confusion matrix at 15% intrusion |

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision across all classes
- **Recall**: Weighted average recall across all classes
- **F1-Score**: Weighted average F1-score across all classes

## Attack Types Detected

The system detects the following attack types:
- Command Response Flood
- DNS Flood
- HTTP Flood
- ICMP Flood
- Man-in-the-Middle (MITM)
- TCP Flood
- UDP Flood
- UDP Lag
- Unsolicited Response

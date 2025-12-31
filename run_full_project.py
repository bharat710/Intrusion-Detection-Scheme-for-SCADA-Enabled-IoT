# ==============================================================================
# SCRIPT METADATA
# ==============================================================================
# Description: Full pipeline to replicate the DNP3 Intrusion Detection paper.
#              This single script handles data aggregation, preprocessing,
#              feature selection, and model training/evaluation.
# Author:      Siddivinayak
# Date:        October 17, 2025
# ==============================================================================

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import re
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
# --- File and Directory Names ---
BASE_DIRECTORY = 'DNP3_Intrusion_Detection_Dataset_Final'
AGGREGATED_FILENAME = 'master_dataset_final_corrected.csv'
PROCESSED_FILENAME = 'dataset_processed.csv'
FINAL_TRAINING_FILENAME = 'dataset_final_for_training.csv'
MAPPING_FILENAME = 'label_mapping.json'
RESULTS_CSV_FILENAME = 'final_results_summary.csv'
COMPARISON_CHART_FILENAME = 'final_performance_comparison.png'

# --- Model and Experiment Parameters ---
NUMBER_OF_FEATURES_TO_SELECT = 99
INTRUSION_RATES = [0.05, 0.10, 0.15] # 5%, 10%, 15%

MODELS = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=100, allow_writing_files=False)
}

# ==============================================================================
# PHASE 1: DATA AGGREGATION
# ==============================================================================
def run_aggregation():
    """
    Combines all raw CSV files from attack folders and the pre-balanced
    training folders into a single master CSV file.
    """
    print("="*80)
    print("🚀 Starting Phase 1: Comprehensive Data Aggregation...")
    print("="*80)

    if not os.path.isdir(BASE_DIRECTORY):
        print(f"❌ ERROR: The directory '{BASE_DIRECTORY}' was not found.")
        return False

    all_dfs = []
    total_files_processed = 0
    training_testing_folder = os.path.join(BASE_DIRECTORY, 'Training_Testing_Balanced_CSV_Files')

    # Part A: Aggregate raw attack files
    print("\n--- Part A: Processing raw attack folders ---")
    attack_folders = [f for f in os.listdir(BASE_DIRECTORY) if os.path.isdir(os.path.join(BASE_DIRECTORY, f)) and 'Training_Testing' not in f]
    for folder in tqdm(attack_folders, desc="Attack Folders"):
        search_pattern = os.path.join(BASE_DIRECTORY, folder, '**', '*.csv')
        csv_files = glob.glob(search_pattern, recursive=True)
        if not csv_files: continue
        print(f"  - Processing {len(csv_files)} files from: {folder}")
        for file_path in tqdm(csv_files, desc=f"Files in {folder}", leave=False):
            try:
                df = pd.read_csv(file_path, low_memory=False)
                df['attack_type'] = '_'.join(folder.split('_')[1:])
                all_dfs.append(df)
                total_files_processed += 1
            except Exception as e:
                print(f"    - ❗️ Could not process {os.path.basename(file_path)}. Error: {e}")

    # Part B: Aggregate pre-balanced files to get 'BENIGN' data
    print("\n--- Part B: Processing Training_Testing_Balanced_CSV_Files ---")
    if os.path.isdir(training_testing_folder):
        search_pattern = os.path.join(training_testing_folder, '**', '*.csv')
        balanced_files = glob.glob(search_pattern, recursive=True)
        if balanced_files:
            print(f"  - Processing {len(balanced_files)} pre-balanced files...")
            for file_path in tqdm(balanced_files, desc="Balanced Files", leave=False):
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    all_dfs.append(df)
                    total_files_processed += 1
                except Exception as e:
                    print(f"    - ❗️ Could not process {os.path.basename(file_path)}. Error: {e}")

    if not all_dfs:
        print("\n❌ ERROR: No data could be loaded.")
        return False

    master_df = pd.concat(all_dfs, ignore_index=True, join='outer', sort=False)
    master_df.to_csv(AGGREGATED_FILENAME, index=False)
    
    print("\n-----------------------------------------")
    print(f"✅ Phase 1 Complete! Saved to '{AGGREGATED_FILENAME}'")
    print(f"   Total rows: {len(master_df):,}")
    print(f"   Label distribution:\n{master_df['Label'].value_counts()}")
    print("-----------------------------------------")
    return True

# ==============================================================================
# PHASE 2: DATA PREPROCESSING
# ==============================================================================
def run_preprocessing():
    """
    Cleans the aggregated data, converts it to a numeric format,
    and creates the final label column.
    """
    print("\n" + "="*80)
    print("🚀 Starting Phase 2: Data Preprocessing...")
    print("="*80)

    try:
        df = pd.read_csv(AGGREGATED_FILENAME, low_memory=False)
        print(f"✅ Successfully loaded '{AGGREGATED_FILENAME}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{AGGREGATED_FILENAME}' was not found.")
        return False

    # Create the final text label BEFORE cleaning column names
    normal_mask = df['Label'].str.strip().str.upper().isin(['BENIGN', 'NORMAL'])
    normal_rows = df[normal_mask].shape[0]
    if normal_rows == 0:
        print("❌ CRITICAL ERROR: No 'BENIGN' or 'NORMAL' rows found. Cannot create 'Normal' class.")
        return False
    print(f"Step 1: Found {normal_rows} normal traffic rows.")
    print("  - Creating final label text...")
    df['final_label_text'] = df.apply(lambda row: 'Normal' if str(row.get('Label', '')).strip().upper() in ['BENIGN', 'NORMAL'] else row.get('attack_type', 'Unknown'), axis=1)

    # Clean column names
    print("  - Cleaning column names...")
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(col).strip().replace('/', '_').replace('.', '_').replace(' ', '_')) for col in df.columns]
    
    # Encode labels and save mapping
    print("  - Encoding labels...")
    label_encoder = LabelEncoder()
    df['final_label'] = label_encoder.fit_transform(df['final_label_text'])
    label_mapping = {int(index): label for index, label in enumerate(label_encoder.classes_)}
    with open(MAPPING_FILENAME, 'w') as f: json.dump(label_mapping, f, indent=4)
    print(f"Step 2: Encoded labels and saved mapping to '{MAPPING_FILENAME}'.")
    
    # Convert object columns to numeric
    print("  - Converting object columns to numeric...")
    object_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in ['attack_type', 'Label', 'final_label_text']]
    for col in tqdm(object_cols, desc="Numeric Conversion"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df.fillna(0, inplace=True)

    # Drop unnecessary columns
    print("  - Dropping unnecessary columns...")
    cols_to_drop = ['attack_type', 'Label', 'final_label_text', 'parser_type', 'timeout', 'FlowID', 'SrcIP', 'DstIP', 'Timestamp', 'frameSrc', 'frameDst']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols, inplace=True)

    # Final validation for infinite values
    print("  - Replacing infinite values...")
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    df.to_csv(PROCESSED_FILENAME, index=False)
    print("\n-----------------------------------------")
    print(f"✅ Phase 2 Complete! Saved to '{PROCESSED_FILENAME}'")
    print(f"   Final shape: {df.shape}")
    print("-----------------------------------------")
    return True

# ==============================================================================
# PHASE 3: FEATURE SELECTION
# ==============================================================================
def run_feature_selection():
    """
    Reduces the feature set to the top 99 most informative features.
    """
    print("\n" + "="*80)
    print(f"🚀 Starting Phase 3: Feature Selection (to {NUMBER_OF_FEATURES_TO_SELECT} features)...")
    print("="*80)
    
    try:
        df = pd.read_csv(PROCESSED_FILENAME)
        print(f"✅ Successfully loaded '{PROCESSED_FILENAME}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{PROCESSED_FILENAME}' was not found.")
        return False
        
    X = df.drop('final_label', axis=1)
    y = df['final_label']
    
    # Scale data first to prevent calculation errors
    print("Step 1: Scaling the data...")
    scaler_start = time.time()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    scaler_end = time.time()
    print(f"Step 1: Scaled the data in {scaler_end - scaler_start:.2f} seconds.")
    
    # Apply Variance Threshold to remove zero-variance features
    print("Step 2: Removing constant features...")
    var_start = time.time()
    selector_var = VarianceThreshold(threshold=0.0)
    X_no_const = pd.DataFrame(selector_var.fit_transform(X_scaled), columns=X_scaled.columns[selector_var.get_support()])
    var_end = time.time()
    print(f"Step 2: Removed constant features in {var_end - var_start:.2f} seconds. Features remaining: {X_no_const.shape[1]}")
    
    # Select the top 99 features
    print(f"Step 3: Selecting top {NUMBER_OF_FEATURES_TO_SELECT} features...")
    kbest_start = time.time()
    selector_kbest = SelectKBest(score_func=f_classif, k=NUMBER_OF_FEATURES_TO_SELECT)
    X_selected = selector_kbest.fit_transform(X_no_const, y)
    final_df = pd.DataFrame(X_selected, columns=X_no_const.columns[selector_kbest.get_support()])
    final_df['final_label'] = y.values
    kbest_end = time.time()
    print(f"Step 3: Selected top {final_df.shape[1]-1} features in {kbest_end - kbest_start:.2f} seconds.")
    
    final_df.to_csv(FINAL_TRAINING_FILENAME, index=False)
    print("\n-----------------------------------------")
    print(f"✅ Phase 3 Complete! Saved to '{FINAL_TRAINING_FILENAME}'")
    print(f"   Final shape: {final_df.shape}")
    print("-----------------------------------------")
    return True

# ==============================================================================
# PHASE 4: MODEL TRAINING & EVALUATION
# ==============================================================================
def run_training_and_evaluation():
    """
    Trains and evaluates the specified models on the final dataset.
    """
    print("\n" + "="*80)
    print("🚀 Starting Phase 4: Model Training and Evaluation...")
    print("="*80)
    
    try:
        df = pd.read_csv(FINAL_TRAINING_FILENAME)
        with open(MAPPING_FILENAME, 'r') as f:
            label_mapping = json.load(f)
        normal_label_index = int([k for k, v in label_mapping.items() if v == 'Normal'][0])
        print(f"✅ Loaded data and mapping. 'Normal' traffic is label: {normal_label_index}")
    except (FileNotFoundError, IndexError):
        print(f"❌ ERROR: Could not load required files ('{FINAL_TRAINING_FILENAME}' or '{MAPPING_FILENAME}').")
        return

    X = df.drop('final_label', axis=1)
    y = df['final_label']
    print("  - Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    results_log = []
    for rate in tqdm(INTRUSION_RATES, desc="Intrusion Rates"):
        print(f"\n--- Processing for {int(rate*100)}% Intrusion Rate ---")
        
        normal_data_X = X_train[y_train == normal_label_index]
        attack_data_X = X_train[y_train != normal_label_index]
        num_attack_samples = int(len(normal_data_X) * rate)
        attack_sample_X = attack_data_X.sample(n=num_attack_samples, random_state=42)
        X_train_imbalanced = pd.concat([normal_data_X, attack_sample_X])
        y_train_imbalanced = y_train.loc[X_train_imbalanced.index]

        print("  - Applying SMOTE for oversampling...")
        smote_start = time.time()
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imbalanced, y_train_imbalanced)
        smote_end = time.time()
        print(f"  - SMOTE completed in {smote_end - smote_start:.2f} seconds. Resampled training set size: {len(X_train_resampled)} rows.")

        for name, model in tqdm(MODELS.items(), desc="Models"):
            print(f"  - Training {name}... (this may take a while)")
            train_start = time.time()
            model.fit(X_train_resampled, y_train_resampled)
            train_end = time.time()
            print(f"  - {name} training completed in {train_end - train_start:.2f} seconds.")
            print(f"  - Evaluating {name}...")
            eval_start = time.time()
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results_log.append({ 'Model': name, 'Intrusion Rate': f"{int(rate*100)}%", 'Accuracy': report['accuracy'], 'Precision': report['weighted avg']['precision'], 'Recall': report['weighted avg']['recall'], 'F1-Score': report['weighted avg']['f1-score'] })
            if name == "XGBoost": save_confusion_matrix(y_test, y_pred, rate)
            eval_end = time.time()
            print(f"  - {name} evaluation completed in {eval_end - eval_start:.2f} seconds.")

    # --- Process and Save Final Results ---
    results_df = pd.DataFrame(results_log)
    print("\n" + "-"*40)
    print("--- Final Performance Summary ---")
    print(results_df.round(4).to_string())
    results_df.round(4).to_csv(RESULTS_CSV_FILENAME, index=False)
    print(f"\n✅ Results summary saved to '{RESULTS_CSV_FILENAME}'")
    
    results_pivot = results_df.pivot(index='Intrusion Rate', columns='Model', values='Accuracy')[list(MODELS.keys())]
    results_pivot.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Performance Comparison of Models', fontsize=16); plt.ylabel('Accuracy Rate'); plt.xlabel('Intrusion Rate')
    plt.xticks(rotation=0); plt.ylim(0.95, 1.005)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(COMPARISON_CHART_FILENAME, dpi=300); plt.close()
    print(f"✅ Final comparison chart saved to '{COMPARISON_CHART_FILENAME}'")
    
    print("\n-----------------------------------------")
    print("🎉 Phase 4 Complete! 🎉")
    print("-----------------------------------------")

def save_confusion_matrix(y_true, y_pred, rate):
    """Generates and saves a heatmap of the confusion matrix for XGBoost."""
    plt.figure(figsize=(12, 9))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for XGBoost at {int(rate*100)}% Intrusion Rate', fontsize=16)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    filename = f'confusion_matrix_xgb_{int(rate*100)}pct.png'
    plt.savefig(filename, dpi=300); plt.close()

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Orchestrates the entire pipeline."""
    if run_aggregation():
        if run_preprocessing():
            if run_feature_selection():
                run_training_and_evaluation()

if __name__ == '__main__':
    main()
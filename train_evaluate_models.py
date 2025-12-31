import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')

# --- Main Configuration Block ---
INPUT_FILENAME = 'dataset_final_for_training.csv'
MAPPING_FILENAME = 'label_mapping.json' # The file that holds our answer
RESULTS_CSV_FILENAME = 'final_results_summary.csv'
COMPARISON_CHART_FILENAME = 'final_performance_comparison.png'

MODELS = { "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1), "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1), "CatBoost": CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False) }
INTRUSION_RATES = [0.05, 0.10, 0.15]

def run_experiments():
    print("🚀 Starting Phase 4: Model Training and Evaluation...")
    
    # --- Step 1: Load Data ---
    try:
        df = pd.read_csv(INPUT_FILENAME)
        print(f"✅ Successfully loaded '{INPUT_FILENAME}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{INPUT_FILENAME}' was not found.")
        return

    # --- Step 2: Load the Label Mapping from the JSON file ---
    # --- THIS IS THE CORRECT AND EFFICIENT FIX ---
    try:
        with open(MAPPING_FILENAME, 'r') as f:
            label_mapping = json.load(f)
        # Find the key (numeric label) for the value 'Normal'
        normal_label_index = int([k for k, v in label_mapping.items() if v == 'Normal'][0])
        print(f"✅ Successfully loaded label mapping. 'Normal' traffic is label: {normal_label_index}\n")
    except (FileNotFoundError, IndexError):
        print(f"❌ ERROR: Could not load '{MAPPING_FILENAME}' or find 'Normal' label. Please run the preprocessing script again.")
        return

    # --- Step 3: Split Data ---
    X = df.drop('final_label', axis=1)
    y = df['final_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Step 3: Split data into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

    # --- Step 4: Run Experiments ---
    results_log = []
    print("\n--- Starting Model Training Loop ---\n")
    for rate in INTRUSION_RATES:
        print(f"--- Processing for {int(rate*100)}% Intrusion Rate ---")
        
        normal_data_X = X_train[y_train == normal_label_index]
        attack_data_X = X_train[y_train != normal_label_index]
        num_attack_samples = int(len(normal_data_X) * rate)
        attack_sample_X = attack_data_X.sample(n=num_attack_samples, random_state=42)
        X_train_imbalanced = pd.concat([normal_data_X, attack_sample_X])
        y_train_imbalanced = y_train.loc[X_train_imbalanced.index]
        print(f"  - Imbalanced training set size: {len(X_train_imbalanced)} rows.")

        smote = SMOTE(random_state=42, n_jobs=-1)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imbalanced, y_train_imbalanced)
        print(f"  - Resampled training set size: {len(X_train_resampled)} rows.")

        for name, model in MODELS.items():
            print(f"\n  Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results_log.append({ 'Model': name, 'Intrusion Rate': f"{int(rate*100)}%", 'Accuracy': report['accuracy'], 'Precision': report['weighted avg']['precision'], 'Recall': report['weighted avg']['recall'], 'F1-Score': report['weighted avg']['f1-score'] })
            print(f"  -> {name} @ {int(rate*100)}% - Accuracy: {report['accuracy']:.4f}, F1-Score: {report['weighted avg']['f1-score']:.4f}")
            if name == "XGBoost": save_confusion_matrix(y_test, y_pred, rate)

    print("\n--- Model Training Complete ---\n")
    process_and_save_results(results_log)

def save_confusion_matrix(y_true, y_pred, rate):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for XGBoost at {int(rate*100)}% Intrusion Rate', fontsize=16)
    plt.ylabel('True Label', fontsize=12); plt.xlabel('Predicted Label', fontsize=12)
    filename = f'confusion_matrix_xgb_{int(rate*100)}pct.png'
    plt.savefig(filename, dpi=300); plt.close()
    print(f"     -> Saved confusion matrix to '{filename}'")

def process_and_save_results(results_log):
    results_df = pd.DataFrame(results_log)
    print("--- Final Performance Summary ---")
    print(results_df.round(4).to_string())
    results_df.round(4).to_csv(RESULTS_CSV_FILENAME, index=False)
    print(f"\n✅ Results summary saved to '{RESULTS_CSV_FILENAME}'")
    
    results_pivot = results_df.pivot(index='Intrusion Rate', columns='Model', values='Accuracy')[list(MODELS.keys())]
    plt.style.use('seaborn-v0_8-whitegrid')
    results_pivot.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title('Performance Comparison of Models', fontsize=16)
    plt.ylabel('Accuracy Rate', fontsize=12); plt.xlabel('Intrusion Rate', fontsize=12)
    plt.xticks(rotation=0); plt.ylim(0.95, 1.005)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(COMPARISON_CHART_FILENAME, dpi=300); plt.close()

    print(f"✅ Final comparison chart saved to '{COMPARISON_CHART_FILENAME}'")
    print("\n-----------------------------------------")
    print("🎉 Evaluation Complete! 🎉")
    print("-----------------------------------------")

if __name__ == '__main__':
    run_experiments()
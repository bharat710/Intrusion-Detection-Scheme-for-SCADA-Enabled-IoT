import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
input_filename = 'dataset_processed.csv'
output_filename = 'dataset_final_for_training.csv'
# The number of features used in the paper's final model
NUMBER_OF_FEATURES_TO_SELECT = 99

print(f"🚀 Starting Phase 3: Feature Selection (to select {NUMBER_OF_FEATURES_TO_SELECT} features)...")

# --- Load the Processed Dataset ---
try:
    df = pd.read_csv(input_filename)
    print(f"✅ Successfully loaded '{input_filename}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ ERROR: The file '{input_filename}' was not found.")
    exit()

# --- 1. Separate Features (X) and Target (y) ---
X = df.drop('final_label', axis=1)
y = df['final_label']
print(f"Step 1: Separated features (X) and target (y). Original features: {X.shape[1]}")

# --- 2. Scale the Data First ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("Step 2: Scaled the data using StandardScaler.")

# --- 3. Apply Variance Threshold ---
# This removes useless features that have no variance
selector_var = VarianceThreshold(threshold=0.0)
X_no_const = selector_var.fit_transform(X_scaled)

# Get the names of the columns that were kept
kept_columns_after_var = X_scaled.columns[selector_var.get_support()]
X_no_const = pd.DataFrame(X_no_const, columns=kept_columns_after_var)

print(f"Step 3: Applied Variance Threshold. Features remaining: {X_no_const.shape[1]}")

# --- 4. Select the Top 99 Features using SelectKBest ---
# This method selects features based on their statistical scores (ANOVA F-value).
# This aligns our feature count with the paper's final model.
selector_kbest = SelectKBest(score_func=f_classif, k=NUMBER_OF_FEATURES_TO_SELECT)
X_selected = selector_kbest.fit_transform(X_no_const, y)

# Get the names of the final 99 columns
final_kept_columns = X_no_const.columns[selector_kbest.get_support()]
X_final = pd.DataFrame(X_selected, columns=final_kept_columns)

print(f"Step 4: Applied SelectKBest. Final number of features: {X_final.shape[1]}")

# --- 5. Create Final DataFrame and Save ---
final_df = X_final
final_df['final_label'] = y.values # Use .values to ensure correct alignment

final_df.to_csv(output_filename, index=False)

print("\n-----------------------------------------")
print("✅ Feature Selection Complete!")
print(f"Final dataset for training saved as: {output_filename}")
print(f"Final dataset shape: {final_df.shape}")
print("-----------------------------------------")
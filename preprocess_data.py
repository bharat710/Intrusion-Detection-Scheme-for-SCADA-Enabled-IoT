import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import json # Import the json library
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
input_filename = 'master_dataset_all_files.csv'
output_filename = 'dataset_processed.csv'
mapping_filename = 'label_mapping.json' # File to save our label mapping

print("🚀 Starting Phase 2: Data Preprocessing (Final Robust Version)...")

# --- Load the Dataset ---
try:
    df = pd.read_csv(input_filename, low_memory=False)
    print(f"✅ Successfully loaded '{input_filename}'.")
except FileNotFoundError:
    print(f"❌ ERROR: The file '{input_filename}' was not found.")
    exit()

# --- 1. Clean Column Names ---
def clean_col_names(column):
    column = str(column).strip()
    column = re.sub(r'[/.\s-]', '_', column)
    column = re.sub(r'[^a-zA-Z0-9_]', '', column)
    return column
df.columns = [clean_col_names(col) for col in df.columns]
print("Step 1: Cleaned column names.")

# --- 2. Create the Final Label FIRST (Critical Change) ---
# We create the label before any data is converted to numbers to ensure it works correctly.
def create_final_label(row):
    # Check the 'Label' column for 'BENIGN' to identify normal traffic.
    if str(row['Label']).strip().upper() == 'BENIGN':
        return 'Normal'
    # Otherwise, use the attack type derived from the folder.
    else:
        return row['attack_type']
df['final_label_text'] = df.apply(create_final_label, axis=1)
print("Step 2: Created text-based final label.")

# --- 3. Encode the Label and Save the Mapping ---
label_encoder = LabelEncoder()
df['final_label'] = label_encoder.fit_transform(df['final_label_text'])
# Create the mapping and save it to a file for the next script to use
label_mapping = {int(index): label for index, label in enumerate(label_encoder.classes_)}
with open(mapping_filename, 'w') as f:
    json.dump(label_mapping, f, indent=4)
print(f"Step 3: Encoded labels and saved the mapping to '{mapping_filename}'.")
print("   -> Label Mapping:", label_mapping)

# --- 4. Convert all other object columns to numeric ---
print("Step 4: Converting remaining object columns to numeric format...")
for col in df.select_dtypes(include=['object']).columns:
    if col not in ['attack_type', 'Label', 'final_label_text']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
df.fillna(0, inplace=True)

# --- 5. Drop Unnecessary Columns ---
columns_to_drop = [
    'attack_type', 'Label', 'parser_type', 'timeout', 'final_label_text',
    'FlowID', 'SrcIP', 'DstIP', 'Timestamp', 'frameSrc', 'frameDst'
]
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=existing_cols_to_drop, inplace=True)
print(f"Step 5: Dropped {len(existing_cols_to_drop)} redundant/identifier columns.")

# --- 6. Final Validation and Cleaning ---
df.replace([np.inf, -np.inf], 0, inplace=True)
print("Step 6: Scanned and replaced all infinity values.")

# --- Save the Processed Data ---
df.to_csv(output_filename, index=False)
print("\n-----------------------------------------")
print("✅ Preprocessing Complete!")
print(f"Cleaned dataset saved as: '{output_filename}'")
print(f"Final dataset shape: {df.shape}")
print("-----------------------------------------")
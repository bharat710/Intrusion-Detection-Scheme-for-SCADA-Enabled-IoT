import pandas as pd

# --- Configuration ---
input_filename = 'dataset_final_for_training.csv'

print(f"🔎 Loading '{input_filename}' to verify selected features...")

# --- Load the final dataset ---
try:
    df = pd.read_csv(input_filename)
except FileNotFoundError:
    print(f"❌ ERROR: The file '{input_filename}' was not found. Please run the feature selection script first.")
    exit()

# Get all column names from the dataframe
all_columns = df.columns.tolist()

# The last column is our target label, so we exclude it to get just the features
feature_columns = all_columns[:-1]

# --- Print the Results ---
print("\n-----------------------------------------")
print(f"✅ Verification Complete!")
print(f"Total features found: {len(feature_columns)}")
print("-----------------------------------------")

print("\nList of the 99 selected features:\n")
# Print the feature names, 5 per line for easy reading
for i in range(0, len(feature_columns), 5):
    print(" | ".join(feature_columns[i:i+5]))
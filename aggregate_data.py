import os
import pandas as pd
import glob

# --- Configuration ---
base_directory = 'DNP3_Intrusion_Detection_Dataset_Final'
output_filename = 'master_dataset_all_files.csv'

print(f"🚀 Starting comprehensive aggregation from: {os.path.abspath(base_directory)}\n")

if not os.path.isdir(base_directory):
    print(f"❌ ERROR: The directory '{base_directory}' was not found.")
    print("Please make sure this script is in the 'Siddivinayak_RIS' folder.")
    exit()

# Get all subfolders, excluding the pre-balanced training folder.
attack_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f)) and 'Training_Testing' not in f]

all_dfs = []
total_files_processed = 0

# Loop through each of the 9 attack folders.
for folder in attack_folders:
    # Use glob to find ALL .csv files recursively within this attack folder.
    search_pattern = os.path.join(base_directory, folder, '**', '*.csv')
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"🟡 Info: No CSV files found for '{folder}'. Skipping.")
        continue

    print(f"Processing {len(csv_files)} files from: {folder}")
    
    # Extract the base attack name.
    attack_name = '_'.join(folder.split('_')[1:])

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # --- Add Metadata Columns ---
            df['attack_type'] = attack_name
            
            # Extract timeout and parser type from the file path.
            path_parts = file_path.split(os.sep)
            
            # Find timeout (e.g., '45_timeout')
            timeout = next((part for part in path_parts if 'timeout' in part), 'unknown_timeout')
            df['timeout'] = timeout.split('_')[0] # Get just the number '45'
            
            # Determine the parser type based on keywords in the file path
            if 'CIC' in file_path:
                df['parser_type'] = 'CICFlowMeter'
            # The custom parser files have DNP3_FLOWLABELED in the name
            elif 'DNP3_FLOWLABELED' in os.path.basename(file_path):
                df['parser_type'] = 'Custom_DNP3_Parser'
            else:
                df['parser_type'] = 'unknown'

            all_dfs.append(df)
            total_files_processed += 1
        except Exception as e:
            print(f"   - ❗️ Could not read or process {os.path.basename(file_path)}. Error: {e}")

# --- Final Combination ---
if not all_dfs:
    print("\n❌ ERROR: No data could be loaded.")
else:
    # IMPORTANT: The two parsers (CICFlowMeter and Custom_DNP3_Parser) create different sets of columns.
    # We use pd.concat with 'outer' join to keep all columns from all files, filling missing values with NaN.
    master_df = pd.concat(all_dfs, ignore_index=True, join='outer', sort=False)
    
    master_df.to_csv(output_filename, index=False)
    
    print("\n-----------------------------------------")
    print("✅ Comprehensive Aggregation Successful!")
    print(f"Processed a total of {total_files_processed} files.")
    print(f"Master dataset saved as: {output_filename}")
    print(f"Total rows: {len(master_df):,}")
    print(f"Total columns (from all parsers): {len(master_df.columns)}")
    print("-----------------------------------------")
    
    print("\n📊 Attack types found and their counts:")
    print(master_df['attack_type'].value_counts())
    
    print("\n📊 Parser types found and their counts:")
    print(master_df['parser_type'].value_counts())

    print("\n📊 Timeout values found and their counts:")
    print(master_df['timeout'].value_counts())
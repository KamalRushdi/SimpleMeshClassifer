import os
import pandas as pd
import os
import csv
import shutil


def filter_csv(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # remove the rows with the same name
    df = df.drop_duplicates(subset=["name"], keep="first")

    # save the filtered CSV and overwrite the original one
    df.to_csv(csv_path, index=False)

    print("Filtered csv saved to {}".format(csv_path))




# --- CONFIGURE THESE ---
csv_path = '../../../Desktop/new_data/kifoz.csv'
src_dir = '../../../../../media/mhd-kamal-rushdi/Hephaestus2TB/ALL_OZGUR_3D_SCANS_RAW'
dst_dir = '../../../Desktop/new_data/kifoz'

filter_csv(csv_path)

# ------------------------
# Step 1: Read CSV and extract names
names = set()
with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row:  # skip empty lines
            names.add(row[0])  # the name is the first column

# Step 2: Go through source directories
for folder_name in os.listdir(src_dir):
    folder_path = os.path.join(src_dir, folder_name)

    if os.path.isdir(folder_path) and folder_name in names:
        src_file = os.path.join(folder_path, 'scan_cut.obj')
        dst_file = os.path.join(dst_dir, f'{folder_name}.obj')

        if os.path.isfile(src_file):
            shutil.copyfile(src_file, dst_file)
            print(f'Copied: {src_file} → {dst_file}')
        else:
            print(f'Warning: scan_cut.obj not found in {folder_path}')

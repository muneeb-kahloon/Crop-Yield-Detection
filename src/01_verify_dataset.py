import os
import pandas as pd
from PIL import Image

# Automatically check for the dataset path regardless of slight folder nesting variations
DATASET_PATH = os.path.join("data", "raw", "PlantVillage")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("data", "raw")

def inspect_dataset(data_dir):
    categories = []
    image_counts = []
    corrupted_files = []

    print(f"Scanning dataset in: {os.path.abspath(data_dir)}\n")

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist! Please check folder paths.")
        return

    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f not in ['processed', 'PlantVillage']]

    if not subfolders and os.path.exists(os.path.join(data_dir, "PlantVillage")):
        data_dir = os.path.join(data_dir, "PlantVillage")
        subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        images = os.listdir(folder_path)
        
        valid_count = 0
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_count += 1
            except Exception:
                corrupted_files.append(img_path)

        categories.append(folder)
        image_counts.append(valid_count)

    df = pd.DataFrame({"Class_Name": categories, "Image_Count": image_counts})
    df = df.sort_values(by="Image_Count", ascending=False).reset_index(drop=True)

    print("--- Dataset Verification Summary ---")
    print(f"Total Classes Found: {len(df)}")
    print(f"Total Valid Images: {df['Image_Count'].sum()}")
    print(f"Corrupted Images Found: {len(corrupted_files)}\n")

    print(df.head(15))  # Shows top 15 classes

if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)
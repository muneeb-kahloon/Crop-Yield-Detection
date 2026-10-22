import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Path setup with auto-detection for nested folders
DATASET_PATH = os.path.join("data", "raw", "PlantVillage")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("data", "raw")

PROCESSED_PATH = os.path.join("data", "processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

def perform_eda_and_cleaning(data_dir):
    categories = []
    image_counts = []
    sample_images = {}

    # Handle nested PlantVillage folder if present
    if not any(os.path.isdir(os.path.join(data_dir, f)) and f not in ['processed', 'PlantVillage'] for f in os.listdir(data_dir)):
        if os.path.exists(os.path.join(data_dir, "PlantVillage")):
            data_dir = os.path.join(data_dir, "PlantVillage")

    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    print(f"Cleaning corrupted files and performing EDA in: {os.path.abspath(data_dir)}\n")

    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        items = os.listdir(folder_path)
        
        valid_images = []
        for item in items:
            img_path = os.path.join(folder_path, item)
            
            # Ensure it is a file, not a directory, before attempting image operations
            if os.path.isfile(img_path):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_images.append(img_path)
                except Exception:
                    print(f"Removing corrupted image file: {img_path}")
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"Could not remove {img_path}: {e}")

        categories.append(folder)
        image_counts.append(len(valid_images))
        if valid_images:
            sample_images[folder] = valid_images[0]

    # Save summary stats
    df = pd.DataFrame({"Class_Name": categories, "Image_Count": image_counts})
    df = df.sort_values(by="Image_Count", ascending=True).reset_index(drop=True)
    csv_path = os.path.join(PROCESSED_PATH, "class_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"Class distribution saved to {csv_path}")

    # Plot 1: Class Distribution Bar Chart
    plt.figure(figsize=(12, 8))
    plt.barh(df["Class_Name"], df["Image_Count"], color="teal")
    plt.xlabel("Number of Images")
    plt.ylabel("Crop & Disease Class")
    plt.title("PlantVillage Class Distribution")
    plt.tight_layout()
    chart_path = os.path.join(PROCESSED_PATH, "class_distribution.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Distribution chart saved to {chart_path}")

    # Plot 2: Sample Image Grid
    num_samples = min(15, len(sample_images))
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.ravel()

    for idx, (class_name, img_path) in enumerate(list(sample_images.items())[:num_samples]):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(class_name.replace("___", "\n"), fontsize=8)
            axes[idx].axis("off")
        except Exception as e:
            print(f"Could not load image preview for {class_name}: {e}")

    # Hide unused grid subplots if classes < 15
    for i in range(num_samples, 15):
        axes[i].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(PROCESSED_PATH, "sample_images_grid.png")
    plt.savefig(grid_path)
    plt.close()
    print(f"Sample grid saved to {grid_path}")

if __name__ == "__main__":
    perform_eda_and_cleaning(DATASET_PATH)
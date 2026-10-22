import os
import json

DATA_DIR = os.path.join("data", "raw", "PlantVillage")
CLASS_INDICES_PATH = os.path.join("database", "class_indices.json")

# Extract only valid subfolder names (excluding files)
folders = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])

class_indices = {str(i): folder_name for i, folder_name in enumerate(folders)}

os.makedirs("database", exist_ok=True)
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f, indent=4)

print(f"✅ Successfully mapped {len(class_indices)} clean crop classes!")
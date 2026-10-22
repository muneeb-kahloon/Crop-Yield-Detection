import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

DATA_DIR = os.path.join("data", "raw", "PlantVillage")
MODEL_PATH = os.path.join("models", "crop_disease_model.h5")
CLASS_INDICES_PATH = os.path.join("database", "class_indices.json")

print("📂 Syncing dataset classes...")

# 1. Load Dataset
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# 2. Save Exact Class Mapping
class_names = train_ds.class_names
class_indices = {str(i): name for i, name in enumerate(class_names)}

os.makedirs("database", exist_ok=True)
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f, indent=4)
print(f"✅ Saved exact {len(class_indices)} class mappings.")

# Normalize datasets
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# 3. Build Fresh MobileNetV2 for exact 36 classes
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Fast-Training 36-Class Model (3 Epochs)...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print("🎉 Model trained and synced perfectly with all 36 classes!")
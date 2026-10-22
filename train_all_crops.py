import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from keras.models import Sequential, Model

# 1. Path Setup
DATA_DIR = os.path.join("data", "raw", "PlantVillage")
MODEL_SAVE_PATH = os.path.join("models", "crop_disease_model.h5")
CLASS_INDICES_PATH = os.path.join("database", "class_indices.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("📂 Loading dataset classes...")

# 2. Dataset Loading
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. Save Class Mapping to JSON
class_names = train_ds.class_names
class_indices = {str(i): name for i, name in enumerate(class_names)}

os.makedirs("database", exist_ok=True)
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f, indent=4)
print(f"✅ Saved {len(class_indices)} crop classes to {CLASS_INDICES_PATH}")

# 4. Data Augmentation & Normalization Pipeline
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

def process_data(ds, augment=False):
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = process_data(train_ds, augment=True)
val_ds = process_data(val_ds, augment=False)

# 5. MobileNetV2 Architecture
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

# 6. Model Training
print("🚀 Training Model on All Crops & Trees Dataset...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 7. Save Updated Model
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"🎉 Successfully trained and saved model to {MODEL_SAVE_PATH}")
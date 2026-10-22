import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

# Keras references using core TensorFlow module
MobileNetV2 = tf.keras.applications.MobileNetV2
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Project Paths
DATASET_PATH = os.path.join("data", "raw", "PlantVillage")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("data", "raw")

MODEL_SAVE_PATH = os.path.join("models", "crop_disease_model.h5")
CLASS_INDICES_PATH = os.path.join("database", "class_indices.json")
PROCESSED_PATH = os.path.join("data", "processed")

os.makedirs("models", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Training Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

def train():
    data_dir = DATASET_PATH
    if not any(os.path.isdir(os.path.join(data_dir, f)) and f not in ['processed', 'PlantVillage'] for f in os.listdir(data_dir)):
        if os.path.exists(os.path.join(data_dir, "PlantVillage")):
            data_dir = os.path.join(data_dir, "PlantVillage")

    print(f"Loading images from: {os.path.abspath(data_dir)}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save Class Mapping for UI
    class_indices = train_generator.class_indices
    indices_to_class = {v: k for k, v in class_indices.items()}
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(indices_to_class, f, indent=4)
    print(f"Class indices saved to {CLASS_INDICES_PATH}")

    num_classes = len(class_indices)

    # MobileNetV2 Architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting Model Training...\n")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Save Final Model
    model.save(MODEL_SAVE_PATH)
    print(f"\nTrained model successfully saved to {MODEL_SAVE_PATH}")

    # Export Training Curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(PROCESSED_PATH, "training_curves.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Training performance curves saved to {curve_path}")

if __name__ == "__main__":
    train()
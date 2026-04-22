# =========================================
# BRAIN TUMOR CLASSIFICATION USING RESNET50
# Full Google Colab Code
# Classes: glioma, meningioma, notumor, pituitary
# =========================================

# =========================
# 1. Install dependencies
# =========================

# =========================
# 2. Import libraries
# =========================
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# =========================
# 3. Upload dataset zip
# =========================
import os

# Replace with your actual folder path
data_dir = r"C:\Users\swaga\OneDrive\Desktop\brain\brain_tumor_data"

print("Dataset path:", data_dir)

# Check contents
print("Folders inside dataset:", os.listdir(data_dir))

# =========================
# 5. Define dataset paths
# =========================
extract_path=r"C:\Users\swaga\OneDrive\Desktop\brain\brain_tumor_data"

train_dir = os.path.join(extract_path, "Training")
test_dir = os.path.join(extract_path, "Testing")

print("Train directory exists:", os.path.exists(train_dir))
print("Test directory exists:", os.path.exists(test_dir))
print("Train classes:", os.listdir(train_dir))
print("Test classes:", os.listdir(test_dir))

# =========================
# 6. Parameters
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# =========================
# 7. Create training + validation datasets
#    from Training folder
# =========================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# 8. Create test dataset
# =========================
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", num_classes)

# =========================
# 9. Optimize dataset pipeline
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# =========================
# 10. Visualize sample images
# =========================
plt.figure(figsize=(12, 8))
for images, labels in train_dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

# =========================
# 11. Data augmentation
# =========================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# =========================
# 12. Build ResNet50 model
# =========================
# Important:
# ResNet50 expects preprocessing via preprocess_input
# We do that inside the model.

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model for initial training
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)   # ResNet50 preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.summary()

# =========================
# 13. Compile model
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 14. Callbacks
# =========================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        "best_resnet50_brain_tumor.keras",
        monitor='val_accuracy',
        save_best_only=True
    )
]

# =========================
# 15. Train top layers first
# =========================
initial_epochs = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=initial_epochs,
    callbacks=callbacks
)

# =========================
# 16. Fine-tune ResNet50
# =========================
base_model.trainable = True

# Freeze first many layers, train only deeper layers
for layer in base_model.layers[:140]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=callbacks
)

# =========================
# 17. Plot accuracy and loss
# =========================
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# =========================
# 18. Evaluate on test dataset
# =========================
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# 19. Predictions on test set
# =========================
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# 20. Classification report
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# 21. Confusion matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 22. Save final model
# =========================
model.save("brain_tumor_resnet50_final.keras")
print("Model saved as brain_tumor_resnet50_final.keras")

# =========================
# 23. Predict on one image
# =========================
def predict_image(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} | Confidence: {confidence:.4f}")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

# Example usage:
# sample_image = "/content/brain_tumor_data/Testing/glioma/Te-gl_1.jpg"
# predict_image(sample_image, model, class_names)
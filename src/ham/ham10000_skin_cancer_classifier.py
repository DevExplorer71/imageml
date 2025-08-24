"""
Skin Cancer Classification using HAM10000 Dataset
Assumes images and metadata are in data/ham1000/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# Paths
DATA_DIR = 'data/ham1000/'
IMG_DIR_1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMG_DIR_2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMG_SIZE = 64
BATCH_SIZE = 32
from src.shared.cnn_model import build_cnn_model

df = pd.read_csv(METADATA_PATH)
img_paths = []
for img_id in df['image_id']:
    path1 = os.path.join(IMG_DIR_1, img_id + '.jpg')
    path2 = os.path.join(IMG_DIR_2, img_id + '.jpg')
    if os.path.exists(path1):
        img_paths.append(path1)
    elif os.path.exists(path2):
        img_paths.append(path2)
    else:
        img_paths.append(None)
df['img_path'] = img_paths
df = df.dropna(subset=['img_path'])

# Use only a subset for quick training (optional)
df = df.sample(2000, random_state=42)

# --- Data Visualization ---
# --- Data Visualization ---

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Show class distribution
plt.figure(figsize=(8,4))
df['dx'].value_counts().plot(kind='bar')
plt.title('HAM10000 Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('ham_class_distribution.png')
plt.close()

# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flatten()):
    img_path = df.iloc[i]['img_path']
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    ax.imshow(img)
    ax.set_title(df.iloc[i]['dx'])
    ax.axis('off')
plt.suptitle('Sample HAM10000 Images')
plt.tight_layout()
plt.savefig('ham_sample_images.png')
plt.close()

# Image generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_dataframe(
    df,
    x_col='img_path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_dataframe(
    df,
    x_col='img_path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(train_gen.class_indices)
model = build_model(num_classes)

# Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# Evaluate
loss, acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {acc:.2f}")


# Example: Predict on a single image
# Change this path to an image you want to test
test_img_path = df.iloc[0]['img_path']  # Using first image from dataframe as example
img = image.load_img(test_img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
pred_probs = model.predict(img_array)
pred_class = np.argmax(pred_probs, axis=1)[0]
class_labels = list(train_gen.class_indices.keys())
print(f"Predicted class for {test_img_path}: {class_labels[pred_class]}")


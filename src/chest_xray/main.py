
import os
import matplotlib.pyplot as plt
from src.chest_xray.data_loader import get_generators
from src.shared.cnn_model import build_pneumonia_cnn

DATA_DIR = 'data/chest_xray'
IMG_SIZE = 64
BATCH_SIZE = 32

if __name__ == "__main__":
    # Force TensorFlow to use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        train_gen, val_gen, test_gen = get_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # --- Data Visualization ---
    # Class distribution
    import numpy as np
    labels = []
    for cls, idx in train_gen.class_indices.items():
        labels.append((cls, np.sum(train_gen.labels == idx)))
    classes, counts = zip(*labels)
    plt.figure(figsize=(6,4))
    plt.bar(classes, counts)
    plt.title('Chest X-ray Class Distribution (Train)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('chest_xray_class_distribution.png')
    plt.close()

    # Sample images
    fig, axes = plt.subplots(2, 5, figsize=(12,5))
    for i, ax in enumerate(axes.flatten()):
        img = train_gen[i][0][0]
        label_idx = np.argmax(train_gen[i][1][0]) if train_gen.class_mode == 'categorical' else int(train_gen[i][1][0])
        label = list(train_gen.class_indices.keys())[label_idx]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(label)
        ax.axis('off')
    plt.suptitle('Sample Chest X-ray Images')
    plt.tight_layout()
    plt.savefig('chest_xray_sample_images.png')
    plt.close()

    # --- Training ---
    try:
        model = build_pneumonia_cnn(IMG_SIZE)
        model.fit(train_gen, validation_data=val_gen, epochs=5)
        loss, acc = model.evaluate(test_gen)
        print(f"Test Accuracy: {acc:.2f}")
    except Exception as e:
        print(f"Error during training/evaluation: {e}")
        exit(1)

    # --- Grad-CAM Visualization ---
    try:
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import Model
        def get_img_array(img):
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = tf.expand_dims(img, axis=0)
            return img

        # Pick a test image
        test_img, _ = test_gen[0]
        # Ensure input is a TensorFlow tensor with correct shape
        img = test_img[0]
        if not isinstance(img, tf.Tensor):
            img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.expand_dims(img, axis=0)

        # Grad-CAM
        grad_model = Model([
            model.input], [model.get_layer('conv5_block3_out').output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8

        # Overlay heatmap
        import cv2
        img_np = img[0]
        if hasattr(img_np, 'numpy'):
            img_np = img_np.numpy()
        img_np = (img_np * 255).astype('uint8')
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        plt.imshow(superimposed_img)
        plt.title('Grad-CAM: What the model focuses on')
        plt.axis('off')
        plt.savefig('chest_xray_gradcam.png')
        plt.close()
        print("Grad-CAM visualization saved as chest_xray_gradcam.png")
    except Exception as e:
        print(f"Error during Grad-CAM visualization: {e}")

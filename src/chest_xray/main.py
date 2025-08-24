import os
from src.chest_xray.data_loader import get_generators
from src.chest_xray.cnn_model import build_pneumonia_cnn

DATA_DIR = 'data/chest_xray'
IMG_SIZE = 64
BATCH_SIZE = 32

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_pneumonia_cnn(IMG_SIZE)
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc:.2f}")

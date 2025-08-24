import os
from src.data.ham10000_loader import load_metadata
from src.models.cnn_model import build_cnn_model
from src.utils.image_gen import get_image_generators

DATA_DIR = 'data/ham1000/'
IMG_DIR_1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMG_DIR_2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMG_SIZE = 64
BATCH_SIZE = 32
SAMPLE_SIZE = 2000

if __name__ == "__main__":
    df = load_metadata(METADATA_PATH, IMG_DIR_1, IMG_DIR_2, sample_size=SAMPLE_SIZE)
    train_gen, val_gen = get_image_generators(df, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    model = build_cnn_model(IMG_SIZE, num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=10)

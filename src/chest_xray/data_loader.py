import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_generators(data_dir, img_size=64, batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    test_gen = datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_gen, val_gen, test_gen

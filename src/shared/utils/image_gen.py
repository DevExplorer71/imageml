import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def get_image_generators(df, img_size, batch_size):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_dataframe(
        df,
        x_col='img_path',
        y_col='dx',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_dataframe(
        df,
        x_col='img_path',
        y_col='dx',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_gen, val_gen

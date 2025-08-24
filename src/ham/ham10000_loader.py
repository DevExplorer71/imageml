import os
import pandas as pd

def load_metadata(metadata_path, img_dir_1, img_dir_2, sample_size=None, random_state=42):
    df = pd.read_csv(metadata_path)
    img_paths = []
    for img_id in df['image_id']:
        path1 = os.path.join(img_dir_1, img_id + '.jpg')
        path2 = os.path.join(img_dir_2, img_id + '.jpg')
        if os.path.exists(path1):
            img_paths.append(path1)
        elif os.path.exists(path2):
            img_paths.append(path2)
        else:
            img_paths.append(None)
    df['img_path'] = img_paths
    df = df.dropna(subset=['img_path'])
    if sample_size:
        df = df.sample(sample_size, random_state=random_state)
    return df

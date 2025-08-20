# ImageML Project

## Overview
This project demonstrates machine learning for cancer detection using image datasets, including HAM10000 (skin cancer) and PatchCamelyon (PCam). Scripts are provided for training and evaluating models using TensorFlow/Keras.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd imageml
```

### 2. Set Up Python Environment
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Data

#### HAM10000 (Skin Cancer MNIST)
- Go to [HAM10000 Kaggle Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Download and extract the following files:
  - `HAM10000_images_part_1`
  - `HAM10000_images_part_2`
  - `HAM10000_metadata.csv`
- Place them in the following directory:
  ```
  /home/james/Desktop/imageml/data/ham1000/
  ```
- Your folder structure should look like:
  ```
  data/ham1000/HAM10000_images_part_1/
  data/ham1000/HAM10000_images_part_2/
  data/ham1000/HAM10000_metadata.csv
  ```

#### PatchCamelyon (PCam)
- Go to [PCam Dataset](https://github.com/basveeling/pcam) or [Camelyon16 Grand Challenge](https://data.camelyon16.grand-challenge.org/)
- Download and extract the `.h5` files:
  - `camelyonpatch_level_2_split_train_x.h5`
  - `camelyonpatch_level_2_split_train_y.h5`
- Place them in:
  ```
  /home/james/Desktop/imageml/data/pcam/
  ```

### 5. Run the Scripts

#### Skin Cancer Classification (HAM10000)
```bash
python ham10000_skin_cancer_classifier.py
```

#### PatchCamelyon Cancer Detection
```bash
python pcam_cancer_detection.py
```

#### Tabular Example (Breast Cancer)
```bash
python breast_cancer_tabular_example.py
```

## Notes
- Large data files are excluded from the repository (see `.gitignore`).
- If you use a different directory, update the script paths accordingly.
- For more details, see comments in each script.

## Troubleshooting
- Ensure your virtual environment is activated before installing packages or running scripts.
- If you encounter missing package errors, install them with `pip install <package-name>`.
- For dataset download issues, check the official dataset pages for access or alternative links.

## License
This project is for educational purposes. Please respect dataset licenses and terms of use.

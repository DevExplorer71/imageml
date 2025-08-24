# ImageML Project

## Overview
This project demonstrates machine learning for medical image analysis using:
- HAM10000 (skin cancer)
- Chest X-ray (pneumonia detection)
- MedMNIST (mini medical image datasets)

Scripts are modular and organized for easy training and evaluation using TensorFlow/Keras and PyTorch.

---

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

---

## Data Download & Organization

### HAM10000 (Skin Cancer)
1. Download from [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Place files in:
   ```
   data/ham/HAM10000_images_part_1/
   data/ham/HAM10000_images_part_2/
   data/ham/HAM10000_metadata.csv
   ```

### Chest X-ray (Pneumonia)
1. Download from [Kaggle Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Place folders in:
   ```
   data/chest_xray/train/NORMAL/
   data/chest_xray/train/PNEUMONIA/
   data/chest_xray/test/NORMAL/
   data/chest_xray/test/PNEUMONIA/
   data/chest_xray/val/NORMAL/
   data/chest_xray/val/PNEUMONIA/
   ```

### MedMNIST
1. No manual download needed. Datasets are downloaded automatically by the scripts.

---


## Running the Workflows

### 1. Skin Cancer Classification (HAM10000)
**Script:** `src/ham/ham10000_skin_cancer_classifier.py`
```bash
/home/james/Desktop/clonedprojects/imageml/.venv/bin/python -m src.ham.ham10000_skin_cancer_classifier
```

### 2. Chest X-ray Pneumonia Detection
**Script:** `src/chest_xray/main.py`
```bash
/home/james/Desktop/clonedprojects/imageml/.venv/bin/python -m src.chest_xray.main
```

### 3. MedMNIST (ChestMNIST Example)
**Script:** `src/medmnist/chestmnist_example.py`
```bash
/home/james/Desktop/clonedprojects/imageml/.venv/bin/python src/medmnist/chestmnist_example.py
```

### 4. Breast Cancer Tabular Example
**Script:** `breast_cancer_tabular_example.py`
```bash
/home/james/Desktop/clonedprojects/imageml/.venv/bin/python breast_cancer_tabular_example.py
```

---

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

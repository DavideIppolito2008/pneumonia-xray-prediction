# Pneumonia Detection from Chest X-Rays

This project provides a complete deep learning workflow for detecting **pneumonia** from chest X-ray images. It includes data preprocessing, model training, evaluation, and prediction via both command-line/GUI scripts and a web integration.

---

## 📂 Project Structure

```
├── data/                 # Dataset directory (train/val/test folders)
├── models/               # Saved models and checkpoints
├── scripts/              # Training, evaluation, and utility scripts
│   ├── preprocess_and_train.py   # End-to-end preprocessing, training, evaluation
│   └── predict_image_gui.py      # GUI-based prediction tool (Tkinter)
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📊 Dataset

The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains chest X-ray images labeled as **NORMAL** or **PNEUMONIA**.

Dataset structure after extraction:

```
chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## ⚙️ Training Pipeline

The script [`preprocess_and_train.py`](scripts/preprocess_and_train.py):

- Loads training, validation, and test datasets with augmentations and normalization.
- Uses **ResNet-18** (pretrained on ImageNet) with a fine-tuned classification head.
- Trains with cross-entropy loss and Adam optimizer.
- Tracks validation accuracy and saves the best model as `models/best_model.pth`.
- Evaluates performance on the test set, printing:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix visualization

Run training with:

```bash
python scripts/preprocess_and_train.py
```

---

## 🔮 Prediction Tools

### 1. Command-line/Web Prediction

The [`site/predict.py`](site/predict.py) script can be called directly or via the PHP backend (`upload_predict.php`) to classify uploaded images.

Example usage:

```bash
python site/predict.py path/to/image.jpg
```

Output:

```json
{"prediction": "PNEUMONIA"}
```

### 2. GUI-based Prediction

The [`predict_image_gui.py`](scripts/predict_image_gui.py) script provides a simple **Tkinter GUI**:

- User selects a chest X-ray file.
- The model predicts **NORMAL** or **PNEUMONIA**.
- Result is shown in a popup window.

Run GUI:

```bash
python scripts/predict_image_gui.py
```

---

## 🛠️ Requirements

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:

- Python 3.8+
- PyTorch
- Torchvision
- scikit-learn
- matplotlib
- Pillow
- Tkinter (for GUI)

---

## 📈 Results & Evaluation

- Model: ResNet-18 (fine-tuned)
- Dataset: Kaggle Chest X-ray (Pneumonia)
- Metrics: Accuracy, Precision, Recall, F1-score
- Visualization: Confusion matrix plotted during evaluation

---

## 📌 Notes

- The model currently distinguishes **NORMAL** vs. **PNEUMONIA** cases.
- Ensure the dataset is structured correctly before training.
- For deployment, consider using a production-ready server (e.g., Flask API, FastAPI, or Docker container).

---

## 👨‍💻 Authors

Developed by **Davide Ippolito** as part of a deep learning project for medical image classification.


# Pneumonia Detection from Chest X-Rays

The project delivers end-to-end deep learning functionalities to effortlessly detect pneumonia from chest X-rays, along with pre- and post-processing for deep learning models, and interfacing through CLI, GUI, and Web Apps.
---

## 📂 Project Structure

```
├── data/                 # Dataset directory (train/val/test dirs)
├── models/               # Saved models and check points
├── scripts/              # Training, eval, and utility scripts
│   ├── preprocess_and_train.py   # End-to-end process, train, and evaluate
│   └── predict_image_gui.py      # Prediction GUI tool (Tkinter)
├── requirements.txt      # Python requirements
└── README.md             # Project readme
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

-Pre-processing and normalizing the training, validation, and testing datasets along with the augmentations.
-Implements a version of ResNet-18 with a classification head that has been fine-tuned and pretrained on ImageNet.
-Utilizes cross-entropy loss and Adam optimizer for training.
-Monitors validation accuracy and saves the best performing model as models/best_model.pth. 
-Analyzes test set performance by reporting:
-Accuracy, precision, recall, F1-score, and
-Visualization of the confusion matrix.

Run training with:

```bash
python scripts/preprocess_and_train.py
```

---

## Prediction Tools
### 1. GUI-based Prediction
The [`predict_image_gui.py`](scripts/predict_image_gui.py) script shows a simple **Tkinter GUI**:

- User selects a chest X-ray file.

- Model predicts **NORMAL** or **PNEUMONIA**.
- Show result in a pop-up window.

Run GUI:

```bash
python scripts/predict_image_gui.py
```

---

## 🛠 Requirements

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


- The model is discriminating **NORMAL** vs. **PNEUMONIA** cases at this moment.
- Ensure that the dataset is formatted correctly before training.
- Deploy with a production-ready server
---

## Authors

Built by **Davide Ippolito** as an exercise in medical image classification using deep learning.

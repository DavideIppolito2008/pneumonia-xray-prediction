import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Configurazione
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Trasformazioni per l'immagine
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
    return pred_class

def select_and_predict():
    file_path = filedialog.askopenfilename(title='Seleziona una radiografia', filetypes=[('Immagini', '*.jpg *.jpeg *.png *.bmp')])
    if not file_path:
        return
    if not os.path.isfile(file_path):
        messagebox.showerror('Errore', f'File non trovato: {file_path}')
        return
    try:
        prediction = predict_image(file_path, model, CLASS_NAMES)
        messagebox.showinfo('Risultato', f'Predizione: {prediction}')
    except Exception as e:
        messagebox.showerror('Errore', str(e))

if __name__ == '__main__':
    model = load_model(MODEL_PATH, len(CLASS_NAMES))
    root = tk.Tk()
    root.title('Predizione Cancro ai Polmoni da Radiografia')
    root.geometry('400x200')
    btn = tk.Button(root, text='Seleziona Immagine e Predici', command=select_and_predict, font=('Arial', 14))
    btn.pack(expand=True)
    root.mainloop()

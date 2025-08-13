"""
Train ResNet50 transfer learning model for tomato leaf disease classification.
"""
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODEL_PATH = "models/resnet50_best.pt"
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_loaders():
    train_ds = ImageFolder(f"{DATA_DIR}/train", transform=transform)
    val_ds = ImageFolder(f"{DATA_DIR}/val", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader

def build_model():
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

def train():
    train_loader, val_loader = get_loaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    metrics = {
        'acc': Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE),
        'prec': Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE),
        'rec': Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE),
        'f1': F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
    }
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_acc, val_prec, val_rec, val_f1 = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_acc += metrics['acc'](preds, labels).item()
                val_prec += metrics['prec'](preds, labels).item()
                val_rec += metrics['rec'](preds, labels).item()
                val_f1 += metrics['f1'](preds, labels).item()
        n = len(val_loader)
        print(f"Val Acc: {val_acc/n:.4f}, Prec: {val_prec/n:.4f}, Rec: {val_rec/n:.4f}, F1: {val_f1/n:.4f}")
        if val_acc/n > best_acc:
            best_acc = val_acc/n
            torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete. Best model saved.")

if __name__ == "__main__":
    train()

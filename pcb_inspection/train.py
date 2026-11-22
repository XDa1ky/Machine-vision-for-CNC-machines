"""
train.py
--------
Обучение PCBNet на синтетическом датасете тайлов.

Перед запуском сгенерируйте датасет:
    python generate_dataset.py

Затем обучите модель:
    python train.py

Результат: models/pcbnet_best.pt  (лучшая модель по val-accuracy)
           results/training_curves.png
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import PCBNet, count_parameters

# ─────────────────────────── Настройки ───────────────────────────
TILE_SIZE   = 64
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "tiles")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RES_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR,   exist_ok=True)

CLASSES = ["capacitor", "chip", "empty", "resistor"]   # алфавитный порядок ImageFolder


# ─────────────────────── Аугментация данных ───────────────────────

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.5, 0.2], std=[0.2, 0.2, 0.1]),
])


# ───────────────────────── Загрузка данных ────────────────────────

def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Классы: {train_ds.classes}")
    print(f"  Train: {len(train_ds)} изображений | Val: {len(val_ds)} изображений")
    return train_loader, val_loader, train_ds.classes


# ──────────────────────── Один эпоха обучения ─────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────── Валидация ───────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ───────────────────────── Сохранение графиков ───────────────────

def save_curves(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = os.path.join(RES_DIR, "training_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Графики сохранены: {out}")


# ─────────────────────────── Главная ─────────────────────────────

def main():
    print("=" * 55)
    print("  Обучение PCBNet")
    print("=" * 55)
    print(f"  Устройство : {DEVICE}")
    print(f"  Эпох       : {EPOCHS}")
    print(f"  Батч       : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print("-" * 55)

    train_loader, val_loader, classes = get_loaders()

    model     = PCBNet(num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"  Параметров : {count_parameters(model):,}")
    print("-" * 55)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "pcbnet_best.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Сохраняем лучшую модель
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "classes":    classes,
                "val_acc":    va_acc,
            }, best_model_path)
            mark = " *"
        else:
            mark = ""

        elapsed = time.time() - t0
        print(
            f"  Эпоха {epoch:3d}/{EPOCHS} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"Val loss={va_loss:.4f} acc={va_acc:.3f} | "
            f"{elapsed:.1f}s{mark}"
        )

    print("=" * 55)
    print(f"  Лучшая Val Accuracy: {best_val_acc:.3f}")
    print(f"  Модель сохранена   : {best_model_path}")
    save_curves(history)
    print("=" * 55)


if __name__ == "__main__":
    main()

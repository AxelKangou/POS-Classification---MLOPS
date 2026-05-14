import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

#import mlflow
#import mlflow.pytorch

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# CONFIG
# =====================
DATA_DIR = "dataset_pos_train/"
BATCH_SIZE = 16
EPOCHS = 3
N_SPLITS = 3
MODEL_NAME = "resnet18"  # or efficientnet_b0

# =====================
# TRANSFORMS
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# MODEL FACTORY
# =====================
def get_model(name, num_classes):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Unsupported model")

    return model.to(device)

# =====================
# TRAIN / EVAL
# =====================
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        loss = loss_fn(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()

    return correct / len(loader.dataset)

# =====================
# MAIN TRAINING
# =====================
def main():

    base_dir = os.getcwd() 
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)


    dataset = ImageFolder(DATA_DIR, transform=train_transform)
    targets = dataset.targets
    num_classes = len(dataset.classes)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_scores = []



    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):

            print(f"\n🔥 Fold {fold+1}")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_subset.dataset.transform = train_transform
            val_subset.dataset.transform = val_transform

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            model = get_model(MODEL_NAME, num_classes)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            best_acc = 0

            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
                acc = evaluate(model, val_loader)

                print(f"Epoch {epoch} | Loss {loss:.4f} | Acc {acc:.4f}")



                if acc > best_acc:
                    best_acc = acc
                    #save_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
                    #torch.save(model.state_dict(), save_path)
                    torch.save(model.state_dict(), f"models/best_model_fold{fold}.pth")

            fold_scores.append(best_acc)

    avg_acc = np.mean(fold_scores)
    print(f"\n✅ CV Accuracy: {avg_acc:.4f}")
    with open("models/results.json", "w") as f:
        json.dump(
            {"cv accuracy": float(avg_acc)},
            f#, indent=4
        )


if __name__ == "__main__":
    main()
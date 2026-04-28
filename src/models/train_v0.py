import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import shutil
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

OUTPUT_DIR = "/content/dataset_images_train"

train_dataset = ImageFolder(OUTPUT_DIR, transform=train_transform)
val_dataset = ImageFolder(OUTPUT_DIR, transform=val_transform)


# Add a check to ensure the dataset is not empty
if len(train_dataset) == 0:
    raise RuntimeError(
        f"Error: No valid images found in '{OUTPUT_DIR}'. "
        "Please ensure images were successfully downloaded and are in a supported format. "
        "Refer to the output of the image download cell for details on download failures."
        "A common cause is corrupted or malformed image files preventing PIL from identifying them."
    )

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

if num_classes == 0:
    raise RuntimeError(
        f"Error: No classes found in '{OUTPUT_DIR}'. "
        "Ensure your images are organized into subdirectories representing classes "
        "(e.g., {OUTPUT_DIR}/class1/image.jpg, {OUTPUT_DIR}/class2/image.png).")


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class EfficientNetClassifier0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

# =====================
# TRAIN FUNCTION
# =====================
def train_one_epoch(model, loader, optimizer, loss_fn, accumulation_steps):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss = loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * xb.size(0) * accumulation_steps

    return total_loss / len(loader.dataset)

# =====================
# VALIDATION
# =====================
def evaluate(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yb).sum().item()

    return correct / len(loader.dataset)

# =====================
# K-FOLD TRAINING
# =====================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset), 1):

    print(f"\n🔥 Fold {fold}")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)

    model = EfficientNetClassifier(num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    loss_fn = nn.CrossEntropyLoss()

    best_score = 0

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, accumulation_steps=2)

        val_score = evaluate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch} | Loss {train_loss:.4f} | Acc {val_score:.4f}")

        # Save best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

    fold_scores.append(best_score)

print(f"\n✅ CV Accuracy: {np.mean(fold_scores):.4f}")



# =====================
# K-FOLD TRAINING
# =====================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset), 1):

    print(f"\n🔥 Fold {fold}")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)

    model = ResNetClassifier(num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    loss_fn = nn.CrossEntropyLoss()

    best_score = 0

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, accumulation_steps=2)

        val_score = evaluate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch} | Loss {train_loss:.4f} | Acc {val_score:.4f}")

        # Save best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

    fold_scores.append(best_score)

print(f"\n✅ CV Accuracy: {np.mean(fold_scores):.4f}")



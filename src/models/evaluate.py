import torch
import os
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset_pos_test/"
MODEL_PATH = "models/best_model_fold0.pth"

def load_model(path, num_classes):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    #model.load_state_dict(torch.load(path, map_location=device))
    # map_location=device is CRITICAL for switching between GPU/CPU
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded weights from {path}")
    else:
        raise FileNotFoundError(f"❌ Model file not found at {path}")
    
    model.to(device)
    model.eval()
    return model

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
            print(f"❌ Test directory {DATA_DIR} not found.")
            return
    
    dataset = ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16)

    model = load_model(MODEL_PATH, len(dataset.classes))

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds.cpu().numpy())
    report = classification_report(y_true, y_pred, target_names=dataset.classes)
    print(report)
    os.makedirs("models", exist_ok=True)
    with open("models/test_report.txt", "w") as f:
        f.write(report)
    print("✅ Report saved to models/test_report.txt")

if __name__ == "__main__":
    evaluate()
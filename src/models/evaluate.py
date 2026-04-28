import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/test/"

def load_model(path, num_classes):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16)

    model = load_model("models/best_model_fold0.pth", len(dataset.classes))

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(yb.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
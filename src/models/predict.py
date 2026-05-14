import torch
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "test_prediction/"
CLASSES = ["supermarket", "minimarket", "kiosk", "grocery"]

def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("models/best_model_fold0.pth"))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)

    conf, pred = torch.max(probs, 1)

    return {
        "predicted_label": CLASSES[pred.item()],
        "confidence": float(conf.item())
    }

if __name__ == "__main__":
    model = load_model()
    result = predict("TEST_DIR/channel1.png", model)
    print(result)
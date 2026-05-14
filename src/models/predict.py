import torch
import os
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "test_prediction/"
CLASSES = ["grocery", "roasteries", "vatrine"]

def load_model(path="models/best_model_fold0.pth"):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(path, map_location=device))
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
        "confidence": round(float(conf.item()), 4)
    }

if __name__ == "__main__":
    model = load_model()
    for image_to_test in os.listdir(TEST_DIR):    
    #image_to_test = os.path.join(TEST_DIR,"channel1.png")
        result = predict(os.path.join(TEST_DIR,image_to_test), model)
        print(f"{image_to_test}:{result['predicted_label']} ({result['confidence']})")
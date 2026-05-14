import torch
import os
import json 
from pathlib import Path  
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "test_prediction/"
REPORT_DIR = "reports"
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
    results = {}
    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(exist_ok=True)
    print(f"🚀 Starting batch prediction in {TEST_DIR}...")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    for img_name in os.listdir(TEST_DIR):
        if img_name.lower().endswith(valid_extensions):
            image_path = os.path.join(TEST_DIR, img_name)
            
            # Run prediction
            prediction = predict(image_path, model)
            
            # Store result using filename as the key
            results[img_name] = prediction
            print(f"Processed: {img_name} -> {prediction['predicted_label']}")
    
    output_path = report_dir/"batch_predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n ✅ Batch prediction complete. result saved to : {output_path}")


"""
FastAPI backend for tomato leaf disease detection.
Endpoints:
- /predict: POST image, returns top-3 classes + probabilities + Grad-CAM
- /health: GET health check
"""
import io
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from models.train_vgg16 import build_model as build_vgg16
from models.train_resnet50 import build_model as build_resnet50
from models.gradcam import GradCAM
import numpy as np
import base64

app = FastAPI()

# Load models
vgg16 = build_vgg16()
vgg16.load_state_dict(torch.load("models/vgg16_best.pt", map_location="cpu"))
vgg16.eval()
resnet50 = build_resnet50()
resnet50.load_state_dict(torch.load("models/resnet50_best.pt", map_location="cpu"))
resnet50.eval()

CLASS_NAMES = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Spider mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_gradcam_overlay(model, input_tensor, arch):
    if arch == "vgg16":
        target_layer = model.features[-1]
    else:
        target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor)
    gradcam.remove_hooks()
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = input_tensor.squeeze().permute(1,2,0).numpy()
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    overlay = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    _, buf = cv2.imencode('.jpg', overlay)
    return base64.b64encode(buf).decode('utf-8')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(image: UploadFile = File(...), model: str = "vgg16"):
    img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    if model == "vgg16":
        net = vgg16
    else:
        net = resnet50
    with torch.no_grad():
        outputs = net(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
    gradcam_overlay = get_gradcam_overlay(net, input_tensor, model)
    return JSONResponse({
        "top3": top3,
        "gradcam": gradcam_overlay
    })

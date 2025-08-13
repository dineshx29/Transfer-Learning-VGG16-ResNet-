"""
Export trained VGG16 and ResNet50 models to ONNX and TorchScript formats.
"""
import torch
import torchvision
from models.train_vgg16 import build_model as build_vgg16
from models.train_resnet50 import build_model as build_resnet50

NUM_CLASSES = 10
DEVICE = torch.device("cpu")
DUMMY_INPUT = torch.randn(1, 3, 224, 224, device=DEVICE)

VGG16_PATH = "models/vgg16_best.pt"
RESNET50_PATH = "models/resnet50_best.pt"

EXPORT_DIR = "models/exported"
import os
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_onnx(model, out_path):
    torch.onnx.export(model, DUMMY_INPUT, out_path, input_names=['input'], output_names=['output'], opset_version=11)
    print(f"Exported ONNX: {out_path}")

def export_torchscript(model, out_path):
    traced = torch.jit.trace(model, DUMMY_INPUT)
    traced.save(out_path)
    print(f"Exported TorchScript: {out_path}")

def main():
    # VGG16
    vgg16 = build_vgg16()
    vgg16.load_state_dict(torch.load(VGG16_PATH, map_location=DEVICE))
    vgg16.eval()
    export_onnx(vgg16, f"{EXPORT_DIR}/vgg16.onnx")
    export_torchscript(vgg16, f"{EXPORT_DIR}/vgg16.pt")
    # ResNet50
    resnet50 = build_resnet50()
    resnet50.load_state_dict(torch.load(RESNET50_PATH, map_location=DEVICE))
    resnet50.eval()
    export_onnx(resnet50, f"{EXPORT_DIR}/resnet50.onnx")
    export_torchscript(resnet50, f"{EXPORT_DIR}/resnet50.pt")

if __name__ == "__main__":
    main()

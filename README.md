# Tomato Leaf Disease Detection (Transfer Learning: VGG16 & ResNet50)

## Overview
This project detects common diseases in tomato leaves using transfer learning (VGG16, ResNet50) and real-world datasets. It includes:
- Automated dataset download/preparation (PlantVillage & PlantDoc)
- Training pipelines for VGG16 and ResNet50
- Data augmentation, evaluation (accuracy, precision, recall, F1)
- Grad-CAM explainability
- Model export (ONNX, TorchScript)
- FastAPI backend for inference
- React + Vite frontend for image upload, prediction, Grad-CAM overlays
- Dockerized deployment (backend + frontend)

## Directory Structure
- `data/` — Dataset scripts and storage
- `models/` — Training, evaluation, export scripts
- `api/` — FastAPI backend
- `web/` — React frontend
- `configs/` — Config files

## Quick Start
1. **Clone repo & install dependencies**
   ```pwsh
   pip install -r requirements.txt
   ```
2. **Prepare dataset**
   ```pwsh
   python data/prepare_data.py
   ```
3. **Train models**
   ```pwsh
   python models/train_vgg16.py
   python models/train_resnet50.py
   ```
4. **Export models**
   ```pwsh
   python models/export_models.py
   ```
5. **Run locally (Docker Compose)**
   ```pwsh
   docker-compose up --build
   ```
6. **Open [http://localhost:3000](http://localhost:3000) in browser**

## Documentation
- All code is commented.
- See README sections for details on each step: dataset, training, evaluation, serving, running the app.

## Requirements
- Python 3.8+
- Node.js 18+
- Docker

## Performance
- Models reach >95% accuracy on validation
- Inference <1s on CPU

---

## Steps
### 1. Dataset Preparation
- Downloads PlantVillage tomato subset (all diseases + healthy)
- Optionally integrates PlantDoc for real-world images
- Splits into train/val/test

### 2. Model Training
- VGG16 & ResNet50 transfer learning
- Data augmentation
- Metrics: accuracy, precision, recall, F1
- Grad-CAM explainability

### 3. Model Export
- Exports both models to ONNX & TorchScript

### 4. Backend (FastAPI)
- Endpoints: `/predict`, `/health`
- Accepts image upload, returns top-3 classes + Grad-CAM

### 5. Frontend (React + Vite)
- Upload image, view predictions & Grad-CAM overlays

### 6. Deployment
- Dockerfiles for backend & frontend
- `docker-compose.yml` for one-command launch

---

## How to Run
See above Quick Start. All steps are automated and ready to run end-to-end.

## Contact
For questions, open an issue or contact the author.

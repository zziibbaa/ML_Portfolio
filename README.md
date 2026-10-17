# 🚀 Computer Vision & End-to-End Machine Learning Portfolio

> **Computer Vision • Object Detection • Deep Learning • Machine Learning • MLOps • Deployment**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO26-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MachineLearning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![Docker](https://img.shields.io/badge/Docker-Deployment-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-purple)
![SQL](https://img.shields.io/badge/SQL-Database-lightgrey)

</p>

---

# ⭐ Portfolio Highlights

This repository showcases my journey toward becoming a Computer Vision Engineer, built on complete, reproducible Machine Learning and Deep Learning projects.

### Highlights

- 🤖 **15 End-to-End AI Projects**, 4 of them in Computer Vision
- 🦺 Object Detection with Ultralytics YOLO26 (**0.978 mAP@50** on a held-out test set)
- 🏷 Dataset Quality & Annotation Correction (FiftyOne + CVAT)
- 🎥 Real-Time Video Processing with OpenCV
- 🧠 Representation Learning: Siamese Networks & Few-Shot Classification
- 🚀 FastAPI & Docker Deployment, Experiment Tracking with MLflow
- 📈 Systematic Model Evaluation & Error Analysis

# 📌 Portfolio Overview

```text
📂 ML_Portfolio

├── 👁 Computer Vision
│   ├── Object Detection (YOLO)
│   ├── Real-Time Video Processing (OpenCV)
│   ├── Image Classification (CNN)
│   └── Few-Shot Learning (Metric Learning)
│
├── 🤖 Machine Learning
│   ├── Classification
│   ├── Regression
│   ├── Clustering
│   └── Feature Engineering
│
├── 🚀 Deployment & MLOps
│   ├── FastAPI
│   ├── Docker
│   └── MLflow
│
├── 💬 Natural Language Processing
│
└── 🗄 SQL Analytics
```

---

# 📊 Portfolio Statistics

| Category | Projects |
|----------|---------:|
| 👁 Computer Vision | 4 |
| 🤖 Machine Learning | 4 |
| 🚀 Deployment & MLOps | 2 |
| 💬 NLP | 2 |
| 🌀 Clustering | 3 |
| 🗄 SQL | 1 |

**Total Independent Projects: 15**

> Some projects appear in more than one category, so the column does not sum to the total.

---

# 👋 About Me

Hi, I'm **Ziba Hatamian**.

Aspiring Computer Vision Engineer with an M.Sc. in Biotechnology, focused on object detection, reproducible training pipelines, and dataset quality. I build complete projects, from data inspection and annotation correction to evaluation and error analysis.

---

# 🏅 Best Projects

If you're visiting this portfolio for the first time, I recommend starting with:

🥇 **Helmet Detection with YOLO26**

An Object Detection project covering leakage-safe data splitting, annotation correction with FiftyOne + CVAT, model comparison, test-time augmentation, and detailed error analysis.

🥈 **Driver Drowsiness Detection**

A real-time OpenCV system that combines Haar Cascade face/eye detection, a CNN eye-state classifier, and temporal logic to warn the driver.

🥉 **Heart Disease Prediction (End-to-End MLOps)**

Complete Machine Learning lifecycle, from raw data to a deployed service:

Data → Training → MLflow → FastAPI → Docker

# 🏆 Featured Projects

## 🦺 Helmet Detection with YOLO26 (Object Detection)

A single-class Object Detection project that detects safety helmets in construction-site images, with a focus on dataset quality and fair model evaluation.

Highlights

- YOLO26n baseline → fine-tuning → custom augmentation → YOLO26s
- Leakage-safe train/validation split using MD5 and pHash near-duplicate detection
- Annotation review and correction with FiftyOne + CVAT (validation boxes: 2,920 → 3,008)
- Confidence-threshold analysis and Test-Time Augmentation (Weighted Boxes Fusion)
- Detailed false-positive / false-negative analysis, including incomplete ground truth
- **0.978 mAP@50 and 0.676 mAP@50–95** on the held-out test set (1,766 images, YOLO26s)

**Technologies**

- Ultralytics YOLO
- PyTorch
- FiftyOne
- CVAT
- Albumentations

📂 Project

> [Deep_Learning/Object_Detection_Error_Analysis/helmet_detection/](Deep_Learning/Object_Detection_Error_Analysis/helmet_detection/)

---

## 🚗 Driver Drowsiness Detection (Real-Time Computer Vision)

A real-time prototype that detects prolonged eye closure from a webcam feed and triggers an audible and visual warning.

Highlights

- Face and eye detection with OpenCV Haar Cascades
- CNN eye-state classifier (open / closed) on 24×24 grayscale eye crops
- Temporal logic: an alarm triggers only when both eyes stay closed for 5 seconds, separating prolonged closure from single-frame errors
- Audio alarm (Pygame), flashing visual warning, and automatic frame capture
- Documented limitations and roadmap: facial landmarks (EAR), PERCLOS, head pose, and quantitative evaluation

**Technologies**

- OpenCV
- TensorFlow / Keras
- CNN
- Haar Cascade
- Pygame

📂 Project

> [OpenCV/Diagnosis of drowsiness/](OpenCV/Diagnosis of drowsiness/)

---

## ❤️ Heart Disease Prediction (End-to-End MLOps)

A complete Machine Learning project covering the entire workflow, from raw data to a deployed service.

Highlights

- Data preprocessing pipeline
- Logistic Regression baseline
- PyTorch Feed Forward Neural Network
- Threshold optimization
- MLflow experiment tracking
- FastAPI REST API
- Docker deployment

**Technologies**

- PyTorch
- Scikit-Learn
- FastAPI
- Docker
- MLflow

📂 Project

> [Machine_Learning/Heart_Disease_End_to_End/](Machine_Learning/Heart_Disease_End_to_End/)

---

# 🧰 Technical Skills

## Programming

- Python
- SQL

## Computer Vision

- Object Detection (Ultralytics YOLO)
- Image Classification (CNN)
- Real-Time Video Processing (OpenCV)
- Classical Detection (Haar Cascades)
- Few-Shot & Metric Learning (Siamese Networks, Embeddings)
- Dataset Quality Checks (duplicate and near-duplicate detection, leakage-safe splits)
- Annotation Review & Correction (FiftyOne, CVAT)
- Data Augmentation (Albumentations)
- Model Evaluation, Threshold Analysis & Error Analysis
- Test-Time Augmentation (Weighted Boxes Fusion)

## Deep Learning

- PyTorch
- TensorFlow / Keras
- Convolutional Neural Networks
- Transfer Learning
- Feed Forward Neural Networks

## Machine Learning

- Scikit-Learn
- Classification
- Regression
- Clustering
- Feature Engineering
- Cross Validation
- Hyperparameter Optimization
- Pipelines

## NLP

- TF-IDF
- Naive Bayes
- Logistic Regression
- Linear SVM

## Data Analysis

- Pandas
- NumPy
- Matplotlib
- Plotly

## Deployment & MLOps

- FastAPI
- Docker
- MLflow
- Flask

## Development Tools

- Git
- GitHub
- VS Code
- Jupyter Notebook

---

# 📂 Portfolio Projects

## 👁 Computer Vision

| Project | Description |
|---------|-------------|
| 🦺 Helmet Detection (YOLO26) | Object Detection |
| 🚗 Driver Drowsiness Detection | Real-Time OpenCV + CNN |
| 🧠 Few-Shot Image Recognition | Siamese Network + ResNet50 |
| 🔤 EMNIST Letters Classification | CNN (PyTorch), 95.30% Test Accuracy |

---

## 🤖 Machine Learning

| Project | Description |
|---------|-------------|
| 🏡 Ames Housing Price Prediction | ElasticNet Regression |
| 🧬 Gene Expression Classification | Multi-class Classification |
| 🪨 Rock Density Regression | Regression |
| 📈 Advertising Sales Prediction | End-to-End ML Deployment |

---

## 🚀 Deployment & MLOps

| Project | Description |
|---------|-------------|
| ❤️ Heart Disease Prediction | PyTorch + MLflow + FastAPI + Docker |
| 📈 Advertising Sales Prediction | FastAPI + Docker + MLflow |

---

## 💬 Natural Language Processing

| Project | Description |
|---------|-------------|
| ✈️ Twitter Airline Sentiment Analysis | Text Classification |
| 🎬 IMDB Movie Review Sentiment Analysis | Sentiment Analysis |

---

## 🌀 Clustering

| Project | Description |
|---------|-------------|
| 🏦 Bank Marketing | K-Means |
| 🌍 World Factbook | Hierarchical Clustering |
| 🛒 Wholesale Customers | DBSCAN |

---

## 🗄 SQL

| Project | Description |
|---------|-------------|
| 🏢 Northwind Business Analysis | SQL Analytics |

---

# 🎯 Current Focus

I'm currently focusing on:

- Computer Vision & Object Detection
- Model Deployment
- Classical Image Processing with OpenCV
- Experiment Tracking & MLOps

---

# 🛣 Learning Roadmap

Upcoming projects planned for this portfolio:

- Planned: deploy the Helmet Detection model by exporting it to ONNX and TensorRT, benchmarking latency, serving it with FastAPI, and packaging it with Docker
- Object tracking on video (ByteTrack)
- Image segmentation (U-Net) and defect inspection combining OpenCV with deep learning
- Image retrieval with CLIP embeddings and FAISS
- Drowsiness detection upgrade: facial landmarks, EAR and PERCLOS
- EfficientNet and Vision Transformer (ViT)
- Recommendation Systems
- Time Series Forecasting
- LSTM and Transformer-based Models

---

# 📈 Repository Philosophy

Rather than collecting isolated notebooks, this repository emphasizes:

- Modular project organization
- Reusable code
- Clean software engineering practices
- Reproducible experiments
- Meaningful evaluation and honest reporting of limitations
- End-to-end Machine Learning workflows

---

# 📫 Contact

## Ziba Hatamian

📧 Email

z.228hatamian@gmail.com

🐙 GitHub

https://github.com/zziibbaa

---

⭐ If you find these projects useful, consider giving the repository a star!

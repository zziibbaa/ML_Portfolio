# ❤️ Heart Disease Prediction

An end-to-end Machine Learning project for predicting heart disease using the Cleveland Heart Disease Dataset.

The project compares a traditional Machine Learning approach (**Logistic Regression**) with a **Feed Forward Neural Network (PyTorch)** and demonstrates a complete deployment workflow including preprocessing pipelines, experiment tracking, REST API development, and Docker containerization.

---

# 🚀 Project Highlights

* End-to-end Machine Learning workflow
* Exploratory Data Analysis (EDA)
* Scikit-Learn preprocessing pipeline
* Logistic Regression baseline
* Feed Forward Neural Network (PyTorch)
* Validation-based threshold optimization
* Experiment tracking with MLflow
* REST API using FastAPI
* Dockerized deployment

---

# 🔄 Project Workflow

```text
                Cleveland Dataset
                       │
                       ▼
               Data Cleaning & EDA
                       │
                       ▼
         Train / Validation / Test Split
                       │
                       ▼
          Scikit-Learn Preprocessing Pipeline
      (StandardScaler + OneHotEncoder)
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
 Logistic Regression         Feed Forward Network
          │                         │
          └────────────┬────────────┘
                       ▼
            Threshold Optimization
         (Validation F1-Score Search)
                       │
                       ▼
              Model Evaluation
                       │
                       ▼
                MLflow Tracking
                       │
                       ▼
              FastAPI Deployment
                       │
                       ▼
                    Docker
```

---

# 📊 Dataset

**Cleveland Heart Disease Dataset**

### Features

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* Slope
* Number of Major Vessels (ca)
* Thalassemia (thal)

### Target

* **0** → No Heart Disease
* **1** → Heart Disease

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Missing value removal
* Stratified train/validation/test split
* Standardization of numerical features
* One-Hot Encoding of categorical variables
* Threshold optimization based on Validation F1-score

A Scikit-Learn Pipeline is used to guarantee identical preprocessing during both training and inference.

---

# 🤖 Models

## Logistic Regression

* LogisticRegressionCV
* 10-fold Cross Validation
* Automatic C selection
* Validation threshold optimization

---

## Feed Forward Neural Network (PyTorch)

Architecture

```text
Input
   │
Linear → 32
   │
LeakyReLU
   │
Dropout(0.3)
   │
Linear → 16
   │
LeakyReLU
   │
Dropout(0.3)
   │
Linear → 2
```


Training

* CrossEntropyLoss
* Adam Optimizer
* Best model selected using Validation Accuracy

---

# 📈 Model Performance

| Model                | Accuracy | Precision | Recall | F1-score |  ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | -------: |
| Logistic Regression  |     0.70 |      0.62 |   0.93 | **0.74** | **0.90** |
| Feed Forward Network | **0.80** |  **0.83** |   0.71 | **0.77** |     0.89 |

---

# 📌 Key Findings

* Logistic Regression achieved the highest Recall for detecting heart disease.
* The Feed Forward Neural Network achieved the best overall balance between Precision and F1-score.
* Despite its simplicity, Logistic Regression remains a strong baseline for small structured clinical datasets.

---

# 📉 ROC Curve

|            Logistic Regression           |         Feed Forward Network        |
| :--------------------------------------: | :---------------------------------: |
| ![](images/RocCurveDisplay_logistic.png) | ![](images/RocCurveDisplay_FFN.png) |

---

## 📊 Experiment Tracking

Model experiments were tracked using **MLflow**.

Logged information includes:

- Hyperparameters
- Validation metrics
- Test metrics
- Best threshold
- ROC-AUC
- Model artifacts

<p align="center">
<img src="images/mlflow_Logistic.png" width="900">
</p>
<p align="center">
<img src="images/mlflow_FFN.png" width="900">
</p>

Experiments are tracked using **MLflow**.

---

# 🚀 REST API

The trained PyTorch model is deployed using **FastAPI**.

## Available Endpoints

| Endpoint   | Description              |
| ---------- | ------------------------ |
| `/health`  | Health check             |
| `/predict` | Heart disease prediction |

### Example Request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "thal": 6,
  "ca": 0
}
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.759,
  "threshold": 0.482,
  "result": "Heart Disease"
}
```

---

# 🐳 Docker

Build the Docker image

```bash
docker build -t heart-disease-api .
```

Run the container

```bash
docker run -p 8000:8000 heart-disease-api
```

Swagger UI

```text
http://localhost:8000/docs
```

---

# 📂 Project Structure

```text
Heart_Disease/
│
├── data/
├── notebooks/
├── images/
├── Logistic_Regression/
├── FFN_Pipeline_v2/
├── mlruns/
└── README.md
```

---

# 🛠 Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* PyTorch
* FastAPI
* MLflow
* Docker
* Joblib

---

# 🚀 Installation

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git

cd ML_Portfolio/Machine_Learning_Project/Heart_Disease

pip install -r FFN_Pipeline_v2/requirements.txt
```

Run the Logistic Regression model

```bash
python Logistic_Regression/main_logistic.py
```

Run the Feed Forward Network

```bash
python FFN_Pipeline_v2/main_skorch.py
```

---

# 👩‍💻 Author

**Ziba**

M.Sc. in Biotechnology transitioning into Machine Learning and AI Engineering.

### Areas of Interest

* Machine Learning
* Deep Learning
* MLOps
* AI for Healthcare

GitHub

https://github.com/zziibbaa

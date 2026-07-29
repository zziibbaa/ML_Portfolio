# 🚀 Advertising Sales Prediction Deployment

An end-to-end Machine Learning deployment project for predicting product sales based on advertising expenditures across multiple marketing channels.

The project demonstrates a complete ML workflow including model training, experiment tracking with MLflow, API development using FastAPI, and Docker containerization for production-ready deployment.

---

# ⭐ Project Highlights

- End-to-End Machine Learning Pipeline
- Sales Prediction using Random Forest Regression
- MLflow Experiment Tracking
- FastAPI REST API Development
- Dockerized Deployment
- Model Serialization with Joblib
- Production-oriented Project Structure

---

# 🔄 Project Workflow

```text
                 Advertising Dataset
                          │
                          ▼
                    Data Analysis
                          │
                          ▼
                   Data Preprocessing
                          │
                          ▼
                 Random Forest Regression
                          │
                          ▼
                     Model Evaluation
                          │
                          ▼
                      RMSE Metric
                          │
                          ▼
                    MLflow Tracking
                          │
                          ▼
                     Saved Model
                          │
                          ▼
                     FastAPI API
                          │
                          ▼
                   Docker Deployment
```

---

# 📊 Dataset

The Advertising dataset contains information about advertising budgets across three different media channels.

### Features

- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

### Target

- Product Sales

The objective of the model is to predict product sales based on advertising investments.

---

# ⚙️ Data Preprocessing

The preprocessing workflow includes:

- Loading and cleaning the dataset
- Feature selection
- Train-Test splitting
- Model training and evaluation
- Model serialization for deployment

---

# 🤖 Model

### Random Forest Regression

The final model was trained using:

- Random Forest Regressor
- Scikit-Learn
- MLflow Experiment Tracking

### Evaluation Metric

The model is evaluated using:

- Root Mean Squared Error (RMSE)

```text
RMSE = 0.7314
```

A lower RMSE indicates better predictive performance.

---

# 📈 Experiment Tracking

Model experiments are tracked using MLflow.

The following information is logged:

- Model parameters
- Evaluation metrics
- RMSE values
- Serialized model artifacts

Run MLflow UI locally:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 🌐 REST API

The trained model is deployed using FastAPI.

### Available Endpoint

| Endpoint | Description |
|----------|------------|
| /predict | Sales Prediction |

---

### Example Request

```json
{
    "TV":150,
    "radio":25,
    "newspaper":10
}
```


### Example Response

```json
{
    "predictions":[18.42]
}
```


The API also supports multiple inputs.

```json
[
    {
        "TV":150,
        "radio":25,
        "newspaper":10
    },
    {
        "TV":200,
        "radio":30,
        "newspaper":5
    }
]
```

---

# 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t advertising-sales-api .
```

Run the container:

```bash
docker run -p 8000:8000 advertising-sales-api
```

After running the container, the API will be available at:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

# 📂 Project Structure

```text
Advertising_Sales_Prediction/
│
├── Advertising.csv
├── Deploy_Model.ipynb
├── Deploy_Model.py
├── final_model.pkl
├── column_name.pkl
├── fast_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🛠 Technologies

- Python
- Scikit-Learn
- Random Forest Regression
- FastAPI
- MLflow
- Docker
- Uvicorn
- Joblib
- Git

---

# 🚀 Production-Ready Features

- Experiment Tracking with MLflow
- REST API Development
- Docker Containerization
- Model Serialization
- Reproducible Environment using requirements.txt
- Separation of Training and Deployment Pipelines
- Batch Prediction Support
- Input Validation with FastAPI

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience in:

- Machine Learning Regression Models
- Experiment Tracking
- Model Deployment
- REST API Development
- Docker Containerization
- Production-oriented ML Pipelines
- MLOps Fundamentals

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

#### Areas of Interest

- Machine Learning
- Deep Learning
- MLOps
- Model Deployment
- Data Science

GitHub:

> https://github.com/zziibbaa

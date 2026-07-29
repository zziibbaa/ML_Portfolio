# 🧬 Gene Expression Classification using K-Nearest Neighbors

A Machine Learning project that applies the K-Nearest Neighbors (KNN) algorithm to classify gene expression data. The project demonstrates a complete machine learning workflow including data preprocessing, pipeline construction, hyperparameter tuning using GridSearchCV, and comprehensive model evaluation.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Scikit-Learn Pipeline Construction
- K-Nearest Neighbors (KNN) Classification
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- ROC Curve Analysis
- Precision-Recall Curve Analysis
- Confusion Matrix Visualization
- Model Performance Evaluation

---

# 🔄 Project Workflow

```text
                 Gene Expression Dataset
                            │
                            ▼
                    Data Exploration
                            │
                            ▼
                     Data Preprocessing
                            │
                            ▼
                   StandardScaler Pipeline
                            │
                            ▼
                  Hyperparameter Tuning
                        (GridSearchCV)
                            │
                            ▼
                    KNN Classification
                            │
                            ▼
                      Cross Validation
                            │
                            ▼
                     Model Evaluation
                            │
                            ▼
        Accuracy • F1-score • ROC Curve
     Precision-Recall Curve • Confusion Matrix
```

---

# 📊 Dataset

The project uses gene expression data containing numerical features associated with gene activity.

The goal of the project is to classify samples into two classes based on gene expression characteristics using the K-Nearest Neighbors algorithm.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Exploratory Data Analysis (EDA)
- Train-Test Split
- Feature Standardization using StandardScaler
- Scikit-Learn Pipeline Construction
- Hyperparameter Tuning using GridSearchCV
- Cross Validation for model selection

Using a Pipeline guarantees identical preprocessing during both training and model evaluation.

---

# 🤖 Model

## K-Nearest Neighbors (KNN)

The project uses:

- KNeighborsClassifier
- StandardScaler
- GridSearchCV
- Cross Validation

### Best Hyperparameters

| Parameter | Value |
|----------|--------|
| Number of Neighbors | 19 |
| Distance Metric | Minkowski |
| p | 2 (Euclidean Distance) |
| Weights | Uniform |

The optimal value of K was determined using GridSearchCV and Cross Validation.

---

# 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.95 |
| Precision | 0.95 |
| Recall | 0.94 |
| F1-score | 0.95 |
| ROC-AUC | 0.98 |
| Average Precision | 0.92 |
| Best K | 19 |

---

# 📌 Key Findings

- The optimal number of neighbors was found to be K=19.
- The model achieved 95% classification accuracy on the test set.
- ROC-AUC of 0.98 indicates excellent class separability.
- Precision-Recall analysis achieved an Average Precision score of 0.92.
- Cross Validation successfully selected the optimal hyperparameters.
- The Scikit-Learn Pipeline guarantees reproducible preprocessing during evaluation.

---

# 📂 Project Structure

```text
Gene_Expression_Classification/
│
├── data/
├── notebooks
└── README.md
```

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- StandardScaler
- KNeighborsClassifier
- GridSearchCV
- Matplotlib
- Seaborn

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git
```

Move to the project directory:

```bash
cd ML_Portfolio/Machine_Learning/Gene_Expression_Classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

---

# 🎯 Skills Demonstrated

This project demonstrates:

- Exploratory Data Analysis
- Data Preprocessing
- Scikit-Learn Pipelines
- Hyperparameter Tuning
- Cross Validation
- KNN Classification
- Model Evaluation
- ROC Curve Analysis
- Precision-Recall Analysis
- Machine Learning Workflow Design

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

#### Areas of Interest

- Machine Learning
- Deep Learning
- Computer Vision
- MLOps
- Data Science

GitHub:

> https://github.com/zziibbaa

# 🏡 Ames Housing Price Prediction

An end-to-end Machine Learning project for predicting house prices using the Ames Housing Dataset.

This project focuses on building a complete regression pipeline including data preprocessing, missing value handling, feature engineering, hyperparameter tuning, model evaluation, and prediction on unseen data using ElasticNet Regression.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Outlier Detection and Removal
- Feature Selection
- ElasticNet Regression
- Hyperparameter Optimization using GridSearchCV
- Model Evaluation using MAE and RMSE
- Residual Analysis
- Model Serialization using Joblib
- Prediction on New Data

---

# 🔄 Project Workflow

```text
                 Ames Housing Dataset
                           │
                           ▼
                    Data Cleaning
                           │
                           ▼
                 Missing Value Handling
                           │
                           ▼
                  Outlier Detection
                           │
                           ▼
                    Feature Analysis
                           │
                           ▼
                      Train/Test Split
                           │
                           ▼
                      Data Scaling
                           │
                           ▼
                    ElasticNet Model
                           │
                           ▼
                      GridSearchCV
                           │
                           ▼
                     Model Evaluation
                           │
                           ▼
                     Residual Analysis
                           │
                           ▼
                   Prediction on New Data
                           │
                           ▼
                    Model Serialization
                         (Joblib)
```

---

# 📊 Dataset

### Ames Housing Dataset

The dataset contains detailed information about residential properties sold in Ames, Iowa.

It includes more than 80 features describing different aspects of houses such as:

- Lot Size
- Neighborhood
- Overall Quality
- Basement Characteristics
- Garage Information
- Year Built
- Living Area
- Sale Condition
- Sale Price

### Target Variable

```text
SalePrice
```

Dataset Source:

> https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data

---

# ⚙️ Data Preprocessing

The preprocessing stage includes:

- Handling missing values
- Outlier removal
- Feature selection
- Train/Test split
- Feature scaling
- Preparing data for model training

---

# 🤖 Model

## ElasticNet Regression

ElasticNet combines the advantages of:

- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

The model was optimized using:

```text
GridSearchCV
        │
        ▼
    alpha tuning
        │
        ▼
   l1_ratio tuning
        │
        ▼
   Best Hyperparameters
        │
        ▼
    Final Model
```

---

# 📈 Model Performance

| Metric | Value |
|--------|------:|
| MAE | 11,166 |
| RMSE | 20,555 |
| Mean Sale Price | 180,815 |

### Relative Performance

```text
RMSE ≈ 11.3%

of the average house price.
```

The prediction error represents approximately 11% of the average house price, indicating that the model captures the underlying patterns of the dataset reasonably well.

---

# 📉 Residual Analysis

Residual analysis was performed to evaluate model behavior.

The following visualizations were used:

- Actual Values vs Residuals Scatter Plot
- Residual Distribution (KDE Plot)

The residual plots indicate:

- No major systematic prediction bias
- Errors are centered around zero
- Residuals show an approximately normal distribution

These observations suggest that the model generalizes reasonably well on unseen data.

---

# 🔮 Prediction on New Data

The trained model can be used to predict house prices for unseen samples.

The prediction pipeline includes:

```text
New Data
    │
    ▼
Feature Scaling
    │
    ▼
ElasticNet Model
    │
    ▼
Predicted Sale Price
```

---

# 💾 Model Serialization

The final trained model was saved using:

```python
joblib.dump()
```

This allows the model to be reused without retraining.

---

# 📂 Project Structure

```text
Ames_Housing_Price_Prediction/
│
├── data/
├── notebooks/
├── images/
├── saved_models/
├── README.md
└── requirements.txt
```

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- ElasticNet
- GridSearchCV
- Matplotlib
- Seaborn
- Joblib

---

# 🚀 Installation

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git

cd ML_Portfolio/Machine_Learning/Ames_Housing_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📌 Key Findings

- ElasticNet successfully captures the relationship between housing features and sale prices.
- Hyperparameter tuning significantly improves model performance.
- Residual analysis indicates stable model behavior.
- The model achieves an RMSE of approximately 11% relative to the average house price.

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

#### Areas of Interest

- Machine Learning
- Deep Learning
- MLOps
- Data Science
- AI for Healthcare

GitHub:

> https://github.com/zziibbaa

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

# 🤖 Model Architecture

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

- RMSE represents approximately 11.3% of the average house price.

- The average prediction error remains relatively small compared to the target variable, indicating that the model captures the underlying relationship between housing features and sale prices reasonably well.
```

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
├── saved_models/
├── README.md
```

---

# 🛠 Technologies Used

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


# 📌 Key Findings

- ElasticNet successfully combines feature selection and regularization, making it well-suited for tabular regression problems.

- Hyperparameter tuning using GridSearchCV improves the model's predictive performance and generalization capability.

- Residual analysis suggests that prediction errors are reasonably centered around zero without major systematic bias.

- The model achieves an RMSE of approximately 11.3% relative to the average house price, demonstrating good predictive performance on unseen data.

- The project highlights the importance of data preprocessing, feature engineering and model evaluation when developing regression pipelines.
---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer | Machine Learning • Deep Learning • MLOps

#### Areas of Interest

- Machine Learning
- Deep Learning
- Computer Vision
- MLOps
- AI Applications

GitHub:

> https://github.com/zziibbaa

# 🪨 Rock Density Prediction using Regression Models

An end-to-end Machine Learning project for predicting rock density using multiple regression algorithms.

The project focuses on comparing the performance of several Machine Learning models for a regression task and investigates how different algorithms capture the relationship between geophysical signal measurements and rock density values.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Multiple Regression Models Comparison
- Polynomial Feature Engineering
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Model Performance Evaluation using RMSE and MAE
- Visualization of Model Predictions
- Comparative Analysis of Regression Algorithms

---

# 🔄 Project Workflow

```text
                 Rock Density Dataset
                           │
                           ▼
                    Data Exploration
                           │
                           ▼
                     Train/Test Split
                           │
                           ▼
                    Data Preparation
                           │
                           ▼
                Multiple Regression Models
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
 Linear Regression                    Polynomial Regression
        │                                     │
        ▼                                     ▼
 KNN Regression                        Decision Tree Regression
        │                                     │
        ▼                                     ▼
 Support Vector Regression              Random Forest Regression
        │                                     │
        ▼                                     ▼
 AdaBoost Regression                  Gradient Boosting Regression
        └─────────────────────────────────────┘
                           │
                           ▼
                 Hyperparameter Optimization
                           │
                           ▼
                     Model Evaluation
                           │
                           ▼
                  Comparative Performance
                           │
                           ▼
                     Best Model Selection
```

---

# 📊 Dataset

The dataset contains two numerical variables:

- Signal Measurements
- Rock Density

### Target Variable

```text
density
```

### Input Feature

```text
signal
```

The goal of this project is to learn the relationship between signal measurements and the corresponding rock density values.

### Average Density Value

| Metric | Value |
|-------|------:|
| Mean Density | 2.225 |

---

# 🔍 Exploratory Data Analysis

The following analyses were performed:

- Distribution analysis
- Scatter plot visualization
- Relationship analysis between signal and density
- Comparison between actual and predicted values

Visualization techniques were also used to investigate how different regression models fit the underlying data distribution.

---

# 🤖 Regression Models

The following models were evaluated:

### Linear Models

- Linear Regression
- Polynomial Regression

### Instance-based Models

- K-Nearest Neighbors Regressor

### Tree-based Models

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

### Kernel-based Models

- Support Vector Regression (SVR)

---

# ⚙️ Hyperparameter Optimization

Hyperparameter tuning was performed for several models.

### Support Vector Regression

The model was optimized using:

```text
GridSearchCV
      │
      ▼
     C tuning
      │
      ▼
   gamma tuning
      │
      ▼
 Best Parameters Selection
```

### Best Parameters

```python
{
    "C":1000,
    "gamma":"scale"
}
```

Polynomial Regression models were also evaluated using polynomial degrees from:

```text
2 → 8
```

Random Forest models were compared using different numbers of trees.

```text
10
64
100
128
```

---

# 📈 Model Performance

| Model | RMSE |
|------|------:|
| Linear Regression | 0.257 |
| Polynomial Regression (Degree=2) | 0.282 |
| Polynomial Regression (Degree=8) | 0.135 |
| KNN Regression (Best) | 0.133 |
| Decision Tree Regression | 0.152 |
| Support Vector Regression | 0.126 |
| Random Forest Regression (10 Trees) | **0.125** |
| AdaBoost Regression | 0.133 |
| Gradient Boosting Regression | 0.133 |

### Linear Regression

| Metric | Value |
|------|------:|
| MAE | 0.211 |
| RMSE | 0.257 |

---

# 📌 Key Findings

The experiments demonstrate that:

- Linear Regression provides a useful baseline model.
- Polynomial Regression significantly improves performance by capturing nonlinear relationships.
- Tree-based ensemble methods outperform simple regression models.
- Support Vector Regression achieves excellent predictive performance after hyperparameter tuning.
- Random Forest Regression achieved the lowest RMSE among all evaluated models.
- Ensemble methods provide more stable predictions compared with simpler models.

Among the evaluated models, Random Forest Regression and Support Vector Regression achieved the best overall performance.

---

# 📉 Visualization

Model predictions were visualized using:

- Scatter plots of the original dataset
- Predicted regression curves
- Comparative analysis of different models

These visualizations provide useful insights into how different algorithms learn the relationship between signal measurements and rock density values.

---

# 📂 Project Structure

```text
Rock_Density_Regression/

│
├── data/
├── notebooks/
├── images/
└── README.md
```

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

### Machine Learning Techniques

- Linear Regression
- Polynomial Regression
- KNN Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression
- AdaBoost Regression
- Gradient Boosting Regression
- GridSearchCV

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

### Areas of Interest

- Machine Learning
- Deep Learning
- MLOps
- Data Science
- AI for Healthcare

GitHub:

```text
https://github.com/zziibbaa
```

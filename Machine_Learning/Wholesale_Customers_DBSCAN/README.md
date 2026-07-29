# 🛒 Wholesale Customers Clustering using DBSCAN

A Machine Learning project that applies the DBSCAN clustering algorithm to explore purchasing patterns among wholesale customers. The project demonstrates the complete unsupervised learning workflow including exploratory data analysis (EDA), feature scaling, hyperparameter analysis, outlier detection, and cluster interpretation through visualization techniques.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Feature Selection
- Data Standardization
- DBSCAN Clustering
- Outlier Detection
- Hyperparameter Analysis (eps)
- Cluster Visualization
- Min-Max Scaling
- Heatmap Analysis

---

# 🔄 Project Workflow

```text
            Wholesale Customers Dataset
                          │
                          ▼
                    Data Exploration
                          │
                          ▼
                   Feature Selection
                  (Remove Channel &
                        Region)
                          │
                          ▼
                  Feature Scaling
                  (StandardScaler)
                          │
                          ▼
                 Hyperparameter Analysis
                     (eps Selection)
                          │
                          ▼
                     DBSCAN Model
                          │
                          ▼
                  Outlier Identification
                          │
                          ▼
                   Cluster Visualization
                          │
                          ▼
                  Cluster Mean Analysis
                          │
                          ▼
                 Min-Max Normalization
                          │
                          ▼
                    Heatmap Visualization
```

---

# 📊 Dataset

The project uses the Wholesale Customers Dataset obtained from the UCI Machine Learning Repository.

### Dataset Information

| Description | Value |
|------------|-------|
| Number of Samples | 440 |
| Learning Type | Unsupervised Learning |
| Algorithm | DBSCAN |

### Features

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen
- Channel
- Region

For clustering purposes, the following variables were removed:

- Channel
- Region

The clustering process focuses exclusively on customer purchasing behavior.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Exploratory Data Analysis (EDA)
- Removing categorical variables (Channel and Region)
- Feature Standardization using StandardScaler
- Hyperparameter analysis for selecting appropriate eps values
- Cluster analysis using DBSCAN
- Min-Max Scaling for cluster comparison

Scaling is particularly important because DBSCAN is sensitive to distance measurements between samples.

---

# 🤖 Model

## DBSCAN Clustering

The project uses:

- DBSCAN
- StandardScaler
- MinMaxScaler

### Hyperparameters

| Parameter | Value |
|----------|--------|
| eps | 2 |
| min_samples | 16 |

The value of eps was investigated across a range of values to analyze its impact on the percentage of detected outliers.

---

# 📈 Results

The project successfully demonstrates:

- Identification of dense regions within the dataset.
- Detection of observations classified as noise points (outliers).
- Cluster visualization using customer purchasing behavior.
- Comparison of cluster characteristics using heatmap analysis.

The percentage of observations classified as outliers was analyzed across different values of eps to facilitate hyperparameter selection.

---

# 📌 Key Findings

- DBSCAN successfully identified clusters without requiring a predefined number of clusters.
- Noise points were automatically detected by the algorithm.
- Grocery, Milk, and Detergents_Paper provide useful visual representations of cluster structures.
- Heatmap visualization highlights differences among cluster purchasing patterns.
- Feature scaling substantially improves clustering performance for distance-based algorithms.

---

# 📊 Visualizations

The project includes:

- Percentage of Outliers vs eps values
- Grocery vs Milk Cluster Visualization
- Milk vs Detergents_Paper Cluster Visualization
- Cluster Mean Heatmap Analysis

These visualizations provide insights into customer purchasing behavior across different clusters.

---

# 📂 Project Structure

```text
Wholesale_Customers_DBSCAN/
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
- DBSCAN
- StandardScaler
- MinMaxScaler
- Matplotlib
- Seaborn

---

# 📚 References

- Wholesale Customers Dataset (UCI Machine Learning Repository)
- Scikit-Learn Documentation
- DBSCAN Documentation

---

# 🎯 Skills Demonstrated

This project demonstrates:

- Exploratory Data Analysis
- Data Preprocessing
- Feature Selection
- Feature Scaling
- Unsupervised Learning
- DBSCAN Clustering
- Outlier Detection
- Cluster Visualization
- Hyperparameter Analysis
- Heatmap Interpretation

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

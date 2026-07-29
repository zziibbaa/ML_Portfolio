# 🏦 Bank Marketing Customer Segmentation using K-Means Clustering

An unsupervised Machine Learning project that applies K-Means clustering to identify hidden customer segments within a bank marketing dataset.

The project includes Exploratory Data Analysis (EDA), data preprocessing, feature engineering, feature scaling, cluster analysis, and customer segmentation using the Elbow Method for selecting the optimal number of clusters.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Feature Engineering
- One-Hot Encoding of Categorical Features
- Feature Scaling
- K-Means Clustering
- Elbow Method (SSD Analysis)
- Customer Segmentation
- Data Visualization

---

# 🔄 Project Workflow

```text
                 Bank Marketing Dataset
                            │
                            ▼
                    Data Cleaning
                            │
                            ▼
                 Exploratory Data Analysis
                            │
                            ▼
                   Feature Engineering
                            │
                            ▼
                  One-Hot Encoding
                            │
                            ▼
                     Feature Scaling
                            │
                            ▼
                    K-Means Clustering
                            │
                            ▼
                     SSD Analysis
                      (Elbow Method)
                            │
                            ▼
                  Optimal Number of Clusters
                            │
                            ▼
                     Cluster Analysis
                            │
                            ▼
                      Business Insights
```

---

# 📊 Dataset

The project uses the Bank Marketing Dataset provided by the UCI Machine Learning Repository.

The dataset contains demographic and financial information about bank customers collected during direct marketing campaigns.

### Examples of Features

- Age
- Job
- Marital Status
- Education
- Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign Information
- Previous Marketing Outcomes

The goal of this project is not prediction but discovering hidden customer groups with similar characteristics.

---

# 🔍 Exploratory Data Analysis

The following analyses were performed:

- Distribution analysis of numerical features
- Correlation analysis
- Missing value inspection
- Outlier analysis
- Customer behavior visualization
- Feature relationship analysis

EDA provides valuable insights before applying clustering algorithms.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical variables
- Feature scaling
- Preparing data for K-Means clustering

Feature scaling was particularly important since K-Means is distance-based and sensitive to feature magnitudes.

---

# 🤖 Model

## K-Means Clustering

The K-Means algorithm was used to segment customers into homogeneous groups based on their characteristics.

The following steps were performed:

- Training multiple K-Means models
- Comparing SSD values
- Applying the Elbow Method
- Selecting the optimal number of clusters

---

# 📈 Evaluation Method

### Elbow Method

The Sum of Squared Distances (SSD) was used to determine the optimal number of clusters.

```text
SSD
 │
 │\
 │ \
 │  \
 │   \
 │    \______
 │______________
        K
```

The optimal number of clusters was selected based on the point where the decrease in SSD begins to stabilize.

---

# 📊 Results

### Optimal Number of Clusters

- Number of Clusters: **6**

The Elbow Method indicates that six clusters provide a reasonable trade-off between model simplicity and within-cluster similarity.

The clustering process successfully segmented customers into six groups with different behavioral characteristics.

---

# 📌 Key Findings

The clustering results reveal that:

- Customers exhibit distinct behavioral patterns.
- Different customer segments may respond differently to marketing campaigns.
- Customer segmentation can improve targeted marketing strategies.
- Clustering provides useful business insights without requiring labeled data.

---

# 📂 Project Structure

```text
Bank_Marketing_Clustering/

│
├── Bank.csv
├── Bank_Marketing.ipynb
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

- K-Means Clustering
- Feature Scaling
- One-Hot Encoding
- Elbow Method
- Exploratory Data Analysis

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

GitHub

> https://github.com/zziibbaa
GitHub:

```text
https://github.com/zziibbaa
```

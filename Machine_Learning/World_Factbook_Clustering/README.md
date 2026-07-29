# 🌍 World Factbook Clustering using KMeans

A Machine Learning project that applies unsupervised learning techniques to explore similarities among countries using demographic, economic, and social indicators from the World Factbook dataset.

The project demonstrates the complete clustering workflow including data preprocessing, exploratory data analysis (EDA), feature scaling, KMeans clustering, and Elbow Method analysis for selecting an appropriate number of clusters.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Data Cleaning and Missing Value Handling
- Feature Scaling and Normalization
- KMeans Clustering
- Elbow Method (SSD Analysis)
- Cluster Visualization
- Unsupervised Learning Workflow

---

# 🔄 Project Workflow

```text
            World Factbook Dataset
                       │
                       ▼
                  Data Cleaning
                       │
                       ▼
               Exploratory Data Analysis
                       │
                       ▼
                 Missing Value Handling
                       │
                       ▼
                  Feature Scaling
                       │
                       ▼
                 KMeans Clustering
                       │
                       ▼
                 SSD Calculation
                  (Elbow Method)
                       │
                       ▼
               Cluster Visualization
                       │
                       ▼
                 Model Evaluation
```

---

# 📊 Dataset

The project uses data derived from the World Factbook containing country-level information.

Examples of features include:

- Population
- GDP
- Literacy Rate
- Birth Rate
- Death Rate
- Internet Usage
- Economic Indicators
- Social Indicators

The dataset provides an opportunity to explore similarities among countries using unsupervised learning techniques.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing unnecessary columns
- Exploratory Data Analysis
- Feature normalization and scaling
- Preparing numerical variables for clustering

Scaling was applied before KMeans since distance-based algorithms are sensitive to feature magnitudes.

---

# 🤖 Clustering Model

## KMeans Clustering

The project uses:

- KMeans Clustering
- SSD (Sum of Squared Distances)
- Elbow Method

The Elbow Method was used to investigate suitable values for the number of clusters.

---

# 📈 Results

The SSD curve suggests that approximately:

> **15 clusters**

provide a reasonable balance between model complexity and within-cluster variance.

The clustering results illustrate that countries can be grouped based on similarities across multiple demographic and economic characteristics.

---

# 📊 Exploratory Data Analysis

EDA includes:

- Distribution analysis of variables
- Missing value analysis
- Feature visualization
- Correlation analysis of numerical variables

These analyses provide insights into the structure of the dataset before applying clustering techniques.

---

# 📂 Project Structure

```text
World_Factbook_Clustering/
│
├── data/
├── images/
├── notebooks/
├── World_Factbook.ipynb
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
- KMeans Clustering

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git
```

Move to the project directory:

```bash
cd ML_Portfolio/Machine_Learning/World_Factbook_Clustering
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# 📚 References

- CIA World Factbook
- Scikit-Learn Documentation
- KMeans Clustering Documentation

---

# 🎯 Skills Demonstrated

This project demonstrates:

- Exploratory Data Analysis
- Data Cleaning
- Missing Value Handling
- Feature Scaling
- Unsupervised Learning
- KMeans Clustering
- Elbow Method Analysis
- Data Visualization

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

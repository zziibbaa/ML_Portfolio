# 🌍 World Factbook Country Clustering using K-Means

An unsupervised Machine Learning project that applies K-Means clustering to identify groups of countries with similar demographic, economic, and social characteristics.

The project includes Exploratory Data Analysis (EDA), data preprocessing, feature scaling, cluster analysis, and country segmentation using the Elbow Method to determine the optimal number of clusters.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Feature Scaling
- K-Means Clustering
- Elbow Method (SSD Analysis)
- Country Segmentation
- Cluster Interpretation
- Data Visualization
- Comparative Analysis of Countries

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
                  Country Group Identification
```

---

# 📊 Dataset

The project uses data derived from the CIA World Factbook.

The dataset contains demographic, economic, and social indicators describing countries around the world.

### Examples of Features

- GDP
- Population
- Literacy Rate
- Birth Rate
- Death Rate
- Internet Usage
- Life Expectancy
- Inflation Rate
- Unemployment Rate
- Other socioeconomic indicators

The objective of this project is to discover hidden patterns among countries rather than performing predictive modeling.

---

# 🔍 Exploratory Data Analysis

The following analyses were performed:

- Missing value analysis
- Distribution analysis of numerical variables
- Correlation analysis
- Outlier inspection
- Country characteristic comparisons
- Visualization of feature relationships

EDA provides valuable insights into the similarities and differences among countries before clustering.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Feature selection
- Feature scaling
- Preparing data for clustering analysis

Since K-Means is distance-based, feature scaling plays an important role in improving clustering performance.

---

# 🤖 Model

## K-Means Clustering

The K-Means algorithm was used to group countries with similar characteristics.

The following steps were performed:

- Training multiple K-Means models
- Comparing SSD values
- Applying the Elbow Method
- Selecting the optimal number of clusters
- Interpreting country groups

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

The optimal number of clusters was selected by analyzing the point where the SSD reduction begins to stabilize.

---

# 📊 Results

### Optimal Number of Clusters

- Number of Clusters: **15**

The clustering model successfully identified multiple groups of countries sharing similar socioeconomic characteristics.

---

# 📌 Key Findings

The clustering results suggest that:

- Countries can be grouped according to demographic and economic similarities.
- Clustering reveals hidden relationships among socioeconomic indicators.
- Countries within the same cluster tend to share common development characteristics.
- The analysis may provide useful insights for comparative studies and policy analysis.

---

# 📂 Project Structure

```text
World_Factbook_Clustering/

│
├── World_Factbook.csv
├── World_Factbook.ipynb
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

- K-Means Clustering
- Feature Scaling
- Exploratory Data Analysis
- Elbow Method
- Cluster Analysis

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/zziibbaa/ML_Portfolio.git
```

Move to the project directory

```bash
cd Machine_Learning/World_Factbook_Clustering
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

---

# 👩‍💻 Author

### Ziba Hatamian

M.Sc. in Biotechnology transitioning into Machine Learning and AI Engineering.

### Areas of Interest

- Machine Learning
- Deep Learning
- Data Science
- MLOps
- AI for Healthcare

GitHub:

```text
https://github.com/zziibbaa
```

---
```

این نسخه از نظر ساختار کاملاً با READMEهای قبلی هماهنگ است، اما عمداً کوتاه‌تر از Heart Disease نوشته شده است. به نظرم از این به بعد باید یک قانون داشته باشیم:

> **هرچه پروژه به Production نزدیک‌تر باشد، README آن مفصل‌تر باشد.**

بنابراین:
- Heart Disease → مفصل‌ترین README.
- Advertising Deployment → مفصل.
- Ames Housing → نسبتاً مفصل.
- NLP و Clustering → متوسط.
- EDA Projects → کوتاه و مختصر.

این باعث می‌شود کل پورتفولیو هم یکدست بماند و هم تناسب مناسبی بین حجم README و پیچیدگی پروژه حفظ شود.

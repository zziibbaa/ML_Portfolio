# 🎬 IMDB Movie Review Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project for sentiment analysis of movie reviews using traditional Machine Learning algorithms.

The project focuses on building a complete text classification pipeline including text preprocessing, exploratory data analysis, feature extraction using TF-IDF, model comparison, and performance evaluation using multiple classification metrics.

---

# 🚀 Project Highlights

- Natural Language Processing (NLP)
- Exploratory Data Analysis (EDA)
- Text Cleaning and Data Validation
- Word Frequency Analysis
- TF-IDF Feature Extraction
- Multinomial Naive Bayes Classification
- Linear Support Vector Classification (LinearSVC)
- Model Comparison
- Classification Report Analysis
- Binary Sentiment Classification

---

# 🔄 Project Workflow

```text
                  IMDB Movie Reviews Dataset
                               │
                               ▼
                         Data Cleaning
                               │
                               ▼
                      Missing Value Handling
                               │
                               ▼
                    Empty Review Detection
                               │
                               ▼
                    Exploratory Data Analysis
                               │
                               ▼
                      Word Frequency Analysis
                               │
                               ▼
                         Train/Test Split
                               │
                               ▼
                      TF-IDF Vectorization
                               │
                               ▼
                       Model Training
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             MultinomialNB            LinearSVC
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Model Evaluation
                               │
                               ▼
                       Performance Comparison
```

---

# 📊 Dataset

The project uses the IMDB Movie Review Dataset for binary sentiment classification.

### Dataset Characteristics

- Original Dataset Size: 2,000 movie reviews
- Valid Reviews after Cleaning: 1,938
- Number of Classes: 2
- Balanced Dataset:
    - Positive Reviews: 969
    - Negative Reviews: 969

### Target Variable

```text
label

- positive (pos)
- negative (neg)
```

Dataset Source:

```text
http://ai.stanford.edu/~amaas/data/sentiment
```

---

# 🔍 Exploratory Data Analysis

Several analyses were performed before model training, including:

- Missing value detection
- Empty review identification
- Class distribution analysis
- Word frequency analysis
- Most common words in positive reviews
- Most common words in negative reviews

The project uses CountVectorizer to identify the most frequent words appearing within each sentiment class, providing additional insights into language patterns associated with positive and negative reviews.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Removing missing values
- Removing empty reviews containing only white spaces
- Splitting the dataset into training and testing subsets
- Converting textual data into numerical representations using TF-IDF

The TF-IDF representation enables the models to capture the relative importance of words while reducing the impact of extremely common terms.

---

# 🤖 Models

Two Machine Learning algorithms were implemented and compared.

## Multinomial Naive Bayes

The model was trained using:

```text
TF-IDF Vectorizer
        │
        ▼
Multinomial Naive Bayes
        │
        ▼
Binary Sentiment Prediction
```

MultinomialNB provides a simple and computationally efficient baseline model for text classification problems.

---

## Linear Support Vector Classification (LinearSVC)

The model pipeline includes:

```text
TF-IDF Vectorizer
        │
        ▼
     LinearSVC
        │
        ▼
Binary Sentiment Prediction
```

LinearSVC is particularly well-suited for high-dimensional sparse feature spaces commonly encountered in NLP tasks.

---

# 📈 Model Performance

| Metric | MultinomialNB | LinearSVC |
|--------|--------------|-----------|
| Accuracy | 81% | 83% |
| Precision (Negative) | 75% | 81% |
| Recall (Negative) | 92% | 86% |
| Precision (Positive) | 90% | 85% |
| Recall (Positive) | 70% | 81% |

---

## Multinomial Naive Bayes

```text
              precision    recall    f1-score

negative          0.75      0.92       0.83
positive          0.90      0.70       0.79

accuracy                                0.81
```

### Observations

- Performs exceptionally well at identifying negative reviews.
- Achieves high recall for the negative class.
- Produces fewer false negatives for negative sentiments.
- Shows comparatively lower recall for positive reviews.

---

## Linear Support Vector Classification

```text
              precision    recall    f1-score

negative          0.81      0.86       0.83
positive          0.85      0.81       0.83

accuracy                                0.83
```

### Observations

- Provides more balanced performance across both classes.
- Improves positive review classification substantially.
- Achieves higher overall accuracy.
- Produces similar F1-scores for both sentiment classes.

---

# 📌 Key Findings

- Both models successfully classify movie reviews with relatively high accuracy.
- LinearSVC achieves the highest overall performance.
- MultinomialNB demonstrates superior recall for negative reviews.
- LinearSVC provides a more balanced trade-off between precision and recall.
- TF-IDF feature extraction proves highly effective for sentiment classification tasks.
- Word frequency analysis provides additional interpretability regarding language patterns in movie reviews.

---

# 📂 Project Structure

```text
IMDB_Movie_Review_Sentiment/

│
├── imdb_reviews.csv
├── IMDB_Sentiment_Analysis.ipynb
└── README.md
```

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF Vectorizer
- CountVectorizer
- Multinomial Naive Bayes
- Linear Support Vector Classification (LinearSVC)
- Matplotlib
- Seaborn

### NLP Techniques

- Text Preprocessing
- Word Frequency Analysis
- TF-IDF Feature Extraction
- Binary Text Classification
- Model Comparison

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

### Areas of Interest

- Machine Learning
- Deep Learning
- Natural Language Processing (NLP)
- MLOps
- Data Science
- AI for Healthcare

GitHub:

```text
https://github.com/zziibbaa
```

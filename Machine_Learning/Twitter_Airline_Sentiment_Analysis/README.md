# ✈️ Twitter Airline Sentiment Analysis

A Natural Language Processing (NLP) project for multi-class sentiment classification of customer tweets about U.S. airlines. The goal of this project is to analyze customer opinions and automatically classify tweets into three sentiment categories: **positive**, **neutral**, and **negative**.

The project demonstrates an end-to-end Machine Learning workflow including text preprocessing, TF-IDF feature extraction, model comparison, and sentiment prediction on unseen tweets.

---

# 🚀 Project Highlights

- Multi-class sentiment classification (Positive, Neutral, Negative)
- Text preprocessing and cleaning pipeline
- TF-IDF feature extraction
- Comparison of three Machine Learning models
- Cross-validation based evaluation
- Precision, Recall and F1-score analysis
- Confusion Matrix evaluation
- Prediction on unseen tweets
- Analysis of model trade-offs for different business objectives

---

# 📊 Dataset

**Twitter US Airline Sentiment Dataset**

- Source: https://www.kaggle.com/crowdflower/twitter-airline-sentiment
- Format: CSV
- Target Classes:
    - Positive
    - Neutral
    - Negative

The dataset contains customer tweets discussing their experiences with major U.S. airlines.

---

# 🔄 Project Workflow

```text
                 Raw Tweets
                      │
                      ▼
                Data Cleaning
                      │
                      ▼
               Text Preprocessing
                      │
                      ▼
                TF-IDF Encoding
                      │
                      ▼
               Train/Test Split
                      │
                      ▼
        --------------------------------------
        │                  │                 │
        ▼                  ▼                 ▼
   Naive Bayes      Logistic Regression     LinearSVC
        │                  │                 │
        --------------------------------------
                           │
                           ▼
                    Model Evaluation
          (Accuracy • Precision • Recall • F1)
                           │
                           ▼
                     Prediction Pipeline
                           │
                           ▼
                   New Tweet Prediction
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Removing URLs
- Removing punctuation marks
- Removing user mentions
- Removing stopwords
- Converting raw text into numerical representations using TF-IDF

Feature extraction is performed using:

```python
TfidfVectorizer()
```

---

# 🤖 Models

Three Machine Learning models were implemented and compared.

### Multinomial Naive Bayes

```python
MultinomialNB()
```

### Logistic Regression

```python
LogisticRegression()
```

### Linear Support Vector Classification

```python
LinearSVC()
```

---

# 📈 Model Performance

## Multinomial Naive Bayes

| Metric | Score |
|--------|------|
| Accuracy | 0.67 |
| Precision (Negative) | 0.66 |
| Recall (Negative) | **0.99** |
| F1-score (Negative) | 0.79 |
| Macro Average F1-score | 0.43 |

```text
              precision    recall    f1-score

negative          0.66      0.99       0.79
neutral           0.79      0.15       0.26
positive          0.89      0.14       0.24

accuracy                               0.67
```

---

## Logistic Regression

| Metric | Score |
|--------|------|
| Accuracy | 0.77 |
| Precision (Negative) | 0.80 |
| Recall (Negative) | 0.93 |
| F1-score (Negative) | 0.86 |
| Macro Average F1-score | 0.69 |

```text
              precision    recall    f1-score

negative          0.80      0.93       0.86
neutral           0.63      0.47       0.54
positive          0.81      0.58       0.68

accuracy                               0.77
```

---

## LinearSVC

| Metric | Score |
|--------|------|
| Accuracy | 0.77 |
| Precision (Negative) | 0.82 |
| Recall (Negative) | 0.89 |
| F1-score (Negative) | 0.86 |
| Macro Average F1-score | **0.70** |

```text
              precision    recall    f1-score

negative          0.82      0.89       0.86
neutral           0.59      0.52       0.55
positive          0.76      0.64       0.69

accuracy                               0.77
```

---

# 📌 Key Findings

The results indicate that different models exhibit different strengths.

- Multinomial Naive Bayes achieved an exceptionally high Recall (99%) for negative tweets, making it particularly useful for customer complaint detection.

- Logistic Regression demonstrated balanced performance across all three sentiment classes while maintaining high overall accuracy.

- LinearSVC achieved the highest Macro F1-score among the evaluated models and provided competitive performance across all sentiment categories.

- The choice of the best model depends on the business objective:
    - If detecting dissatisfied customers is the primary goal, Multinomial Naive Bayes is highly effective.
    - If balanced multi-class sentiment classification is required, Logistic Regression and LinearSVC provide superior performance.

---

# 🔮 Prediction Example

A prediction pipeline was created using TF-IDF and LinearSVC.

```python
pipe = Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('svc_model',LinearSVC(dual='auto'))
])

pipe.fit(df['text'],df['airline_sentiment'])
```

Predicting the sentiment of a new tweet:

```python
new_tweet = ['good flight']

pipe.predict(new_tweet)
```

Example output:

```python
['positive']
```

---

# 📂 Project Structure

```text
Twitter_Airline_Sentiment_Analysis
│
├── Twitter_Airline_Sentiment.ipynb
├── images/
├── dataset/
├── README.md
│
└── Prediction Pipeline
        │
        ├── TF-IDF Vectorizer
        └── LinearSVC Model
```

> The exact project structure may vary depending on future improvements.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- TfidfVectorizer
- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC
- Matplotlib
- Seaborn

---

# 🚀 Future Improvements

Possible extensions for this project include:

- Hyperparameter tuning using GridSearchCV.
- Transformer-based sentiment classification models.
- Deployment using FastAPI.
- Real-time Twitter sentiment prediction APIs.
- Visualization dashboards for airline sentiment monitoring.

---

# 👩‍💻 Author

### Ziba Hatamian

M.Sc. in Biotechnology transitioning into Machine Learning and AI Engineering.

#### Areas of Interest

- Machine Learning
- Deep Learning
- Natural Language Processing
- MLOps
- Model Deployment

GitHub:

```text
https://github.com/zziibbaa
```

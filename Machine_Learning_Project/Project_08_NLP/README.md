# 💬 Sentiment Analysis Projects

این پوشه شامل دو پروژه مجزا در زمینه تحلیل احساسات متنی با استفاده از الگوریتم‌های یادگیری ماشین است:

1. ✈️ تحلیل احساسات توییتر خطوط هوایی  
2. 🎬 تحلیل احساسات نظرات فیلم‌ها (IMDB)

---

## ✈️ Twitter Airline Sentiment Analysis

### 🎯 هدف پروژه  
دسته‌بندی توییت‌های کاربران درباره خطوط هوایی به سه کلاس احساسات: **مثبت**، **منفی** و **خنثی**.

### 📊 دیتاست  
- **منبع:** [Crowdflower Twitter Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)  
- **فرمت:** CSV  
- **کلاس‌ها:** `positive`, `neutral`, `negative`

### 🧹 پیش‌پردازش داده‌ها  
- حذف URL، منشن‌ها، علائم نگارشی  
- تبدیل به حروف کوچک  
- حذف stopwords  
- تبدیل متن به ویژگی عددی با `CountVectorizer` یا `TfidfVectorizer`

### 🤖 مدل‌های استفاده‌شده  
- ✅ **Multinomial Naive Bayes**  
- ✅ **Logistic Regression**  
- ✅ **Linear Support Vector Classification (LinearSVC)**

### 📈 ارزیابی عملکرد  
- 📊 **Accuracy**  
- 📊 **Precision / Recall / F1-score**  
- 📊 **Confusion Matrix**  
- 📊 **Cross-validation**

---

## 🎬 IMDB Movie Review Sentiment Analysis

### 🎯 هدف پروژه  
تشخیص احساسات مثبت یا منفی در نظرات کاربران درباره فیلم‌ها.

### 📊 دیتاست  
- **منبع:** [IMDB Sentiment Dataset – Stanford](http://ai.stanford.edu/~amaas/data/sentiment)  
- **فرمت:** فایل‌های متنی در پوشه‌های `pos` و `neg`  
- **کلاس‌ها:** `positive`, `negative`

### 🧹 پیش‌پردازش داده‌ها  
- حذف HTML و علائم نگارشی  
- تبدیل به حروف کوچک  
- حذف stopwords با:
  ```python
  CountVectorizer(stop_words='english')

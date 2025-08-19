# 🧬 تحلیل سطح بیان ژن با مدل K-Nearest Neighbors

این پروژه با هدف پیش‌بینی سطح بیان ژن‌ها بر اساس داده‌های بیوانفورماتیکی انجام شده است. با استفاده از داده‌های مرتبط با ویژگی‌های ژنتیکی، یک مدل K-Nearest Neighbors (KNN) طراحی و ارزیابی شده است. مراحل شامل ساخت پایپ‌لاین، تنظیم هایپرپارامترها با Grid Search و ارزیابی عملکرد مدل هستند.

---

## 📁 منبع داده

- **موضوع:** Gene Expression Level  
- **منبع علمی:** [ScienceDirect – Gene Expression Level](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/gene-expression-level)  
- داده‌ها شامل ویژگی‌های عددی و طبقه‌بندی‌شده مرتبط با بیان ژن در نمونه‌های مختلف هستند

---

## 📌 مراحل انجام‌شده در پروژه

### 1. ⚙️ ساخت پایپ‌لاین مدل‌سازی

- طراحی پایپ‌لاین شامل مراحل پیش‌پردازش داده‌ها (استانداردسازی، رمزگذاری ویژگی‌ها)  
- اتصال مدل KNN به پایپ‌لاین برای اجرای یکپارچه

### 2. 🔍 تنظیم هایپرپارامترها با Grid Search

- اجرای **Full Cross Validation** برای یافتن بهترین مقدار K  
- استفاده از `GridSearchCV` برای بررسی عملکرد مدل در مقادیر مختلف K  
- انتخاب مقدار بهینه بر اساس معیارهای ارزیابی (مانند دقت یا F1)

### 3. 🧪 ارزیابی مدل

ارزیابی مدل با استفاده از روش‌های زیر انجام شده است:

- **ماتریس سردرگمی (Confusion Matrix):** بررسی صحت طبقه‌بندی  
- **دقت، Recall، F1-Score:** برای سنجش کیفیت پیش‌بینی  
- **نمودار ROC Curve:** بررسی توانایی مدل در تفکیک کلاس‌ها  
- **نمودار Precision-Recall Curve:** تحلیل عملکرد مدل در شرایط عدم‌تعادل کلاس‌ها

---

## 📈 نتایج کلیدی

- مقدار بهینه K با استفاده از Cross Validation انتخاب شد  
- مدل توانست با دقت قابل‌قبول سطح بیان ژن را پیش‌بینی کند  
- نمودارهای ROC و Precision-Recall نشان‌دهنده عملکرد مناسب مدل در تفکیک کلاس‌ها هستند  
- ویژگی‌های خاصی از داده‌ها تأثیر بیشتری در پیش‌بینی داشتند که در تحلیل ویژگی‌ها مشخص شد

---

## 🧰 ابزارها و کتابخانه‌های استفاده‌شده

- Python  
- Pandas, NumPy  
- Scikit-learn (Pipeline, KNeighborsClassifier, GridSearchCV, metrics)  
- Seaborn, Matplotlib  

---

## 📚 منابع

- [Gene Expression Level – ScienceDirect](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/gene-expression-level)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

---

## 📝 نتیجه‌گیری

مدل KNN با تنظیم دقیق مقدار K توانسته عملکرد مناسبی در پیش‌بینی سطح بیان ژن‌ها ارائه دهد. استفاده از پایپ‌لاین باعث ساده‌سازی فرآیند و افزایش قابلیت بازتولید مدل شده است. ارزیابی‌های انجام‌شده نشان می‌دهند که مدل برای کاربردهای بیوانفورماتیکی قابل اعتماد است.

---

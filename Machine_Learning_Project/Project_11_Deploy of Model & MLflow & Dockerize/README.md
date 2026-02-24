# 🚀 پروژه ۱۱ – استقرار مدل یادگیری ماشین با MLflow، FastAPI و Docker

این پروژه یک نمونه عملی از پیاده‌سازی چرخه کامل توسعه تا استقرار (Deployment) مدل یادگیری ماشین است.  
مدل رگرسیون آموزش داده شده، آزمایش‌ها با MLflow ثبت شده‌اند، مدل در قالب API با FastAPI ارائه شده و در نهایت در یک کانتینر Docker اجرا می‌شود.

این ساختار مشابه معماری واقعی پروژه‌های Production در حوزه ML Engineering و MLOps است.

---

## 🎯 اهداف پروژه

- آموزش مدل رگرسیون روی دیتاست Advertising
- مدیریت و ثبت آزمایش‌ها با MLflow
- ذخیره و نسخه‌بندی مدل
- ارائه مدل در قالب REST API با FastAPI
- کانتینرسازی پروژه برای اجرای ایزوله و قابل حمل

---

## 🧠 چرخه کامل پروژه (End-to-End Flow)

```
Data → Training Script → MLflow Tracking → Model Serialization → FastAPI → Docker Container
```

### توضیح مراحل:

1. داده‌ها بارگذاری و پیش‌پردازش می‌شوند.
2. مدل آموزش داده می‌شود.
3. پارامترها و متریک‌ها در MLflow ثبت می‌شوند.
4. مدل نهایی ذخیره می‌شود (pickle).
5. مدل از طریق FastAPI در قالب API ارائه می‌شود.
6. کل سرویس داخل Docker اجرا می‌شود.

---

## 🏗 معماری پروژه

```
                ┌───────────────┐
                │ Advertising   │
                │ Dataset       │
                └──────┬────────┘
                       │
                ┌──────▼────────┐
                │ Training      │
                │ Script        │
                └──────┬────────┘
                       │
                ┌──────▼────────┐
                │ MLflow        │
                │ Tracking      │
                └──────┬────────┘
                       │
                ┌──────▼────────┐
                │ Saved Model   │
                └──────┬────────┘
                       │
                ┌──────▼────────┐
                │ FastAPI       │
                │ Prediction API│
                └──────┬────────┘
                       │
                ┌──────▼────────┐
                │ Docker        │
                │ Container     │
                └───────────────┘
```

---

## 📁 ساختار پروژه

```
Project_11_Deploy
│
├── Advertising.csv
├── Deploy_Model.ipynb
├── Deploy_Model.py
├── final_model.pkl
├── column_name.pkl
├── fast_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 مدیریت آزمایش‌ها با MLflow

اطلاعات ثبت‌شده در MLflow:

- پارامترهای مدل
- متریک‌های ارزیابی
- مدل ذخیره‌شده به عنوان Artifact

اجرای MLflow UI:

```bash
mlflow ui
```

سپس در مرورگر:

```
http://127.0.0.1:5000
```

---

## 🌐 ارائه مدل با FastAPI

اجرای محلی:

```bash
uvicorn fast_api:app --reload
```

مستندات Swagger:

```
http://127.0.0.1:8000/docs
```

---

## 🔎 نمونه درخواست (Request Example)

```json
{
  "TV": 150,
  "radio": 25,
  "newspaper": 10
}
```

---

## 🔮 نمونه پاسخ (Response Example)

```json
{
  "predictions": [18.42]
}
```

> می‌توان چند ورودی هم به صورت لیست ارسال کرد:

```json
[
  {"TV": 150, "radio": 25, "newspaper": 10},
  {"TV": 200, "radio": 30, "newspaper": 5}
]
```

---

## 🐳 اجرای پروژه با Docker

ساخت Docker Image:

```bash
docker build -t ml-deploy-app .
```

اجرای کانتینر:

```bash
docker run -p 8000:8000 ml-deploy-app
```

پس از اجرا، API روی پورت 8000 در دسترس خواهد بود.

---

## 📦 توضیح Dockerfile

- انتخاب base image پایتون
- کپی فایل‌های پروژه
- نصب dependencies از requirements.txt
- اجرای FastAPI با Uvicorn

این ساختار باعث می‌شود پروژه در هر محیطی بدون وابستگی به سیستم میزبان اجرا شود.

---

## 🛠 تکنولوژی‌های استفاده‌شده

- Python
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Uvicorn

---

## 🚀 نکات Production-Ready

- جدا کردن فایل آموزش از فایل API
- ذخیره مدل به صورت مستقل
- استفاده از requirements.txt برای reproducibility
- اجرای سرویس داخل کانتینر ایزوله
- ثبت آزمایش‌ها برای رهگیری مدل‌ها
- Validation ورودی‌ها با Pydantic (مقادیر غیرمنفی)

---

## 🎓 مهارت‌های تقویت‌شده در این پروژه

- طراحی پایپ‌لاین کامل ML
- مدیریت نسخه مدل
- پیاده‌سازی REST API برای مدل
- آشنایی با مفاهیم استقرار
- تفکر مهندسی در ساخت پروژه‌های ML

---

## 👩‍💻 توسعه‌دهنده

زیبا  
Machine Learning & AI Enthusiast  
مسیر تخصصی: Model Deployment و MLOps

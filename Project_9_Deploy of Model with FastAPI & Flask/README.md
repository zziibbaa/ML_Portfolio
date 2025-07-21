# 🧠 Machine Learning API - Flask & FastAPI Deployments

این پوشه شامل دو نسخه از پیاده‌سازی API برای مدل‌های یادگیری ماشین است:
- نسخه‌ای با **FastAPI**
- نسخه‌ای با **Flask**

هدف از این پروژه، تبدیل مدل‌های یادگیری ماشین به یک سرویس تحت وب است که بتواند از طریق درخواست‌های HTTP پیش‌بینی انجام دهد.

---

## 🗂️ ساختار فایل‌ها

```bash
API/
├── model.pkl               # مدل آموزش‌دیده ذخیره‌شده (با joblib یا pickle)
├── app.py                  # اسکریپت اصلی API (ممکن است برای Flask یا FastAPI باشد)
├── [flaask.py]             # Flask جداگانه
├── [fastapi.py]        # FastAPI جداگانه
└── README.md               # این فایل

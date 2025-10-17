# 🏢 NorthWind SQL Project

این پروژه شامل **۳۰ تمرین SQL** بر اساس دیتابیس **Northwind** است و هدف آن تمرین مفاهیم **SQL پایه و پیشرفته** و تحلیل داده‌ها با استفاده از جداول واقعی کسب‌وکار می‌باشد.

---

## 🗂️ ساختار پروژه
```
SQL_Project/
├── NorthWind_Project/
│   ├── Exercise.sql        # تمام کوئری‌های تمرینی
│   └── README.md           # توضیحات پروژه
└── Northwind/              # Submodule دیتابیس Northwind
```

---

## 🧩 تمرین‌ها و اهداف آن‌ها

| شماره | تمرین | هدف |
|-------|-------|-----|
| 1 | Display all columns from Customers | نمایش تمام ستون‌ها |
| 2 | Show CompanyName and Country of customers | نمایش نام شرکت و کشور |
| 3 | List orders with customer name | ارتباط Orders و Customers |
| 4 | Display employees info | اطلاعات کامل پرسنل |
| 5 | Count total orders | محاسبه تعداد کل سفارش‌ها |
| 6 | Count orders per customer | تعداد سفارش هر مشتری |
| 7 | Products with stock < 10 | نمایش محصولات کم موجودی |
| 8 | Customers from Germany | فیلتر مشتریان بر اساس کشور |
| 9 | Orders handled by "Davolio" | فیلتر سفارشات توسط کارمند مشخص |
| 10 | Customers with no orders | نمایش مشتریان بدون سفارش |
| 11 | Number of products per category | شمارش محصولات هر دسته |
| 12 | Most expensive product per category | محصول گران هر دسته |
| 13 | Average unit price per category | میانگین قیمت محصولات |
| 14 | Employees with >100 orders | پرسنل با سفارشات زیاد |
| 15 | Products never ordered | نمایش محصولات بدون سفارش |
| 16 | Customers with >5 orders | مشتریان فعال |
| 17 | Average order amount per customer | میانگین مبلغ سفارش |
| 18 | Top 5 best-selling products | ۵ محصول پرفروش |
| 19 | Countries with highest customer count | کشورها با بیشترین مشتری |
| 20 | Orders with >10 items | سفارشات با بیش از ۱۰ کالا |
| 21 | Earliest and latest order dates | قدیمی‌ترین و جدیدترین سفارش |
| 22 | Products with supplier names | ارتباط Products و Suppliers |
| 23 | Orders with customer and employee name | نمایش سفارش با نام مشتری و پرسنل |
| 24 | Count orders per month (1997) | تعداد سفارشات ماهانه |
| 25 | Total sales per customer | مجموع فروش مشتری |
| 26 | Employees handling US orders | پرسنل پردازش سفارشات آمریکا |
| 27 | Average number of products per order | میانگین تعداد محصولات هر سفارش |
| 28 | Products above average price | محصولات با قیمت بالاتر از میانگین |
| 29 | Total revenue per category | درآمد هر دسته محصول |
| 30 | Countries with no recorded orders | کشورهایی بدون سفارش |

---

## 🛠️ تکنولوژی‌ها و ابزارها
- **SQL Server** یا هر DBMS سازگار با T-SQL  
- Git و GitHub برای مدیریت نسخه  
- Submodule برای مدیریت دیتابیس Northwind  

---

## 🚀 نحوه اجرای پروژه
1. کلون کردن repository همراه با submodule:
```bash
git clone --recurse-submodules https://github.com/zziibbaa/SQL_Project.git
```
2. یا اگر قبلاً پروژه را کلون کرده‌اید:
```bash
git submodule update --init --recursive
```
3. باز کردن فایل `Exercise.sql` در محیط SQL Server Management Studio یا هر DBMS دیگر و اجرای کوئری‌ها.  

---

## 📚 منابع
- [Northwind Database on GitHub](https://github.com/zziibbaa/sql-server-samples/tree/main/samples/databases/northwind-pubs)  
- مستندات SQL Server برای T-SQL

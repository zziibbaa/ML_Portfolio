# 🏢 NorthWind SQL Project

این پروژه شامل تمرین‌ها و کوئری‌های SQL بر اساس دیتابیس **Northwind** است. هدف، تمرین مفاهیم **SQL پایه و پیشرفته** و تحلیل داده‌ها با استفاده از جداول واقعی کسب‌وکار می‌باشد.

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

## 🧩 تمرین‌های انجام شده

### 1️⃣ میانگین تعداد محصولات در هر سفارش
- استفاده از `SUM(Quantity)` و `AVG()`  
- نمایش میانگین واقعی محصولات سفارش داده شده  

```sql
SELECT AVG(TotalProducts) AS AvgProductsPerOrder
FROM (
    SELECT OrderID, SUM(Quantity) AS TotalProducts
    FROM [Order Details]
    GROUP BY OrderID
) AS OrderProductTotals;
```

### 2️⃣ محصولات بالاتر از میانگین قیمت
- استفاده از زیرکوئری برای مقایسه با میانگین کل محصولات  
- تمرین فیلتر داده‌ها با `WHERE` و `AVG()`  

```sql
SELECT ProductName, UnitPrice
FROM Products
WHERE UnitPrice > (SELECT AVG(UnitPrice) FROM Products);
```

### 3️⃣ درآمد کل هر دسته محصول
- استفاده از `JOIN` بین `Categories`, `Products` و `[Order Details]`  
- لحاظ کردن `Discount` در محاسبه درآمد  

```sql
SELECT c.CategoryName, 
       SUM(od.Quantity * od.UnitPrice * (1 - od.Discount)) AS revenueCategory
FROM Categories AS c
JOIN Products AS p ON p.CategoryID = c.CategoryID
JOIN [Order Details] AS od ON od.ProductID = p.ProductID
GROUP BY c.CategoryName
ORDER BY revenueCategory DESC;
```

### 4️⃣ کشورهایی بدون سفارش
- استفاده از `NOT EXISTS` یا `LEFT JOIN … IS NULL`  
- تمرین ارتباط بین جداول و شرط‌های فیلتر  

```sql
SELECT DISTINCT c.Country
FROM Customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM Orders AS o
    WHERE o.CustomerID = c.CustomerID
);
```

### 5️⃣ تمرین‌های دیگر
- انواع `JOIN`ها (INNER, LEFT, RIGHT)  
- GROUP BY و HAVING  
- توابع تحلیلی (`COUNT() OVER`, `SUM()`)  

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

/* =========================================
   SQL Practice Questions for Northwind Database
   ========================================= */

/* 1️⃣  Display all columns from the Customers table. */

SELECT * 
FROM Customers

/* 2️⃣  Show the CompanyName and Country of all customers. */

SELECT CompanyName,Country
FROM Customers

/* 3️⃣  List all orders along with the customer name who placed each order. */

SELECT o.CustomerID , c.CompanyName
FROM Orders AS o
JOIN Customers AS c
ON c.CustomerID=o.CustomerID

/* 4️⃣ Display each employee's first name, last name, and job title. */

SELECT CONCAT ( FirstName , '_', LastName , '---JOB_TITLE: ' , Title) AS FULL_INFO
FROM Employees


/* 5️⃣ Find the total number of orders in the system. */

SELECT COUNT(*) AS Count_ALL_OF_Orders
FROM Orders

/* 6️⃣ Count the number of orders placed by each customer. */

SELECT 
			o.CustomerID , 
			c.CompanyName AS customer_name,
			COUNT(o.CustomerID) AS total_orders
FROM Orders AS o
JOIN Customers AS c
ON c.CustomerID=o.CustomerID
GROUP BY o.CustomerID , c.CompanyName
ORDER BY COUNT(o.CustomerID) DESC

/* 7️⃣ Show all products with a stock quantity less than 10. */

SELECT 
			ProductID , 
			ProductName, 
			UnitsInStock
FROM Products
WHERE UnitsInStock <10
ORDER BY UnitsInStock 

/* 8️⃣ Find all customers from Germany. */


SELECT 
			CompanyName , ContactName , City, Country
FROM Customers
WHERE Country='Germany'

/* 9️⃣ List all orders handled by the employee named "Davolio". */

SELECT *
FROM Orders
WHERE EmployeeID IN (SELECT EmployeeID
								FROM Employees
								WHERE LastName= 'Davolio')

/* 🔟 Display all customers who have never placed an order. */

SELECT *
FROM Customers AS c
WHERE NOT EXISTS(SELECT *
							   FROM Orders AS o
							   WHERE o.CustomerID=c.CustomerID)

/*11️⃣  Show the number of products in each category. */

SELECT 
			c.CategoryName ,
			COUNT(p.CategoryID) AS product_count
FROM Products AS p
JOIN Categories AS c
ON c.CategoryID=p.CategoryID
GROUP BY c.CategoryName

/* 12️⃣ Find the most expensive product in each category. */
	
WITH ProductStats AS (
								    SELECT 
									CategoryID,
									ProductName,
									UnitPrice,
        MAX(UnitPrice) OVER(PARTITION BY CategoryID) AS MaxPrice,
        COUNT(*) OVER(PARTITION BY CategoryID) AS ProductCount
    FROM Products
)
SELECT *
FROM ProductStats
WHERE UnitPrice = MaxPrice;


/* 13️⃣ Calculate the average unit price of products in each category. */

SELECT c.CategoryName,
			AVG(UnitPrice) AS AVG_PRICE , 
			COUNT(*) AS productCount
FROM Products AS p
JOIN Categories AS c
ON p.CategoryID=c.CategoryID
GROUP BY c.CategoryName


/* 14️⃣ Find employees who have managed more than 100 orders. */

SELECT 
			CONCAT(e.FirstName, ' /' , e.LastName) AS employ_name ,
			COUNT(*) AS employ_order
FROM Orders AS o
JOIN Employees AS e
ON e.EmployeeID=o.EmployeeID
GROUP BY e.FirstName ,  e.LastName
HAVING COUNT(*)>100

/* 15️⃣ List all products that have never been ordered. */

SELECT	p.ProductID , p.ProductName
FROM Products AS p
WHERE NOT EXISTS(	SELECT 1
								FROM [Order Details] AS o_d
								WHERE o_d.ProductID=P.ProductID)

/* 16️⃣ Display customers who have placed more than 5 orders. */

SELECT CompanyName , COUNT(o.OrderID) AS total_orders
FROM Customers AS c
JOIN Orders AS o 
ON o.CustomerID=c.CustomerID
GROUP BY CompanyName
HAVING COUNT(o.OrderID)>5
ORDER BY total_orders DESC

/* 17️⃣ Calculate the average order amount for each customer. */

WITH order_totals AS(SELECT
											OrderID, 
											SUM(UnitPrice * Quantity * (1-Discount)) AS order_total
								FROM [Order Details]
								GROUP BY OrderID)
SELECT c.CompanyName , AVG(ot.order_total) AS avg_orderAmount
FROM Customers AS c
JOIN Orders AS o
ON c.CustomerID=o.CustomerID
JOIN order_totals AS ot
ON o.OrderID=ot.OrderID
GROUP BY c.CompanyName
ORDER BY avg_orderAmount DESC

/* 18️⃣ Find the top 5 best-selling products. */

SELECT TOP 5 p.ProductName ,  COUNT(OrderID) AS total_order
FROM [Order Details] AS od
JOIN Products AS p
ON od.ProductID=p.ProductID
GROUP BY ProductName
ORDER BY total_order DESC

/* 19️⃣ Show the countries with the highest number of customers. */

WITH total_order AS (SELECT CustomerID , 
											COUNT(OrderID) AS total_orders
								FROM Orders
								GROUP BY CustomerID)
SELECT c.Country , 
			SUM(t_o.total_orders) AS total_orders_country
FROM Customers AS c
JOIN total_order AS t_o
ON c.CustomerID=t_o.CustomerID
GROUP BY c.Country
ORDER BY total_orders_country DESC
/* 20️⃣ Find orders that contain more than 10 items. */

WITH order_stats AS (SELECT c.CompanyName ,
											o.OrderID ,
											SUM(od.Quantity) OVER (PARTITION BY o.OrderID) AS item_count
								FROM [Order Details] AS od
								JOIN Orders AS o
								ON o.OrderID=od.OrderID
								JOIN Customers AS c
								ON c.CustomerID=o.CustomerID)
SELECT DISTINCT
							CompanyName ,
							OrderID , 
							item_count
FROM order_stats
WHERE item_count>10
ORDER BY item_count DESC


/* 21️⃣ Display the earliest and latest order dates in the system. */

SELECT c.CompanyName ,  o.OrderDate 
FROM Orders AS o
JOIN Customers AS c
ON o.CustomerID=c.CustomerID
WHERE o.OrderDate IN
									(( SELECT MIN(OrderDate) FROM Orders),
									  (SELECT MAX(OrderDate) FROM Orders))

/* 22️⃣ List all products along with their supplier names. */

SELECT p.ProductName , s.SupplierID , s.CompanyName
FROM Products AS p
JOIN Suppliers AS s
ON p.SupplierID=s.SupplierID

/* 23️⃣ Show orders with both customer name and employee name. */

SELECT 
			o.OrderID , 
			c.CompanyName AS customer_company ,
			CONCAT(e.FirstName , '/' , e.LastName) AS EmployName
FROM Orders AS o
JOIN Customers AS c
ON c.CustomerID=o.CustomerID
JOIN Employees AS e
ON e.EmployeeID=o.EmployeeID

/* 24️⃣ Count the number of orders per month for the year 1997. */

SELECT MONTH(OrderDate) AS OrderMonth,
			COUNT(OrderDate) AS TotalOrders
FROM Orders
WHERE YEAR(OrderDate) = 1997
GROUP BY MONTH(OrderDate)
ORDER BY MONTH(OrderDate)

/* 25️⃣ Calculate the total sales for each customer and sort them in descending order. */

SELECT c.CompanyName AS customerCompanyk , 
			ROUND(SUM(od.UnitPrice * od.Quantity * (1- od.Discount)) , 2) AS totalSales
FROM Orders AS o
JOIN Customers AS c
ON c.CustomerID=o.CustomerID
JOIN [Order Details] AS od
ON o.OrderID=od.OrderID
GROUP BY c.CompanyName
ORDER BY totalSales DESC


/* 26️⃣ Find employees who have handled orders from customers in the USA. */

SELECT DISTINCT c.CompanyName AS customerCompany,
		    c.Country AS customerCountry , 
			CONCAT(e.FirstName , ' ' , e.LastName) AS employ_name
FROM Customers AS c
JOIN Orders AS o
ON o.CustomerID= c.CustomerID
JOIN Employees AS e
ON o.EmployeeID = e.EmployeeID
WHERE c.Country = 'USA'
ORDER BY employ_name


/* 27️⃣ Calculate the average number of products per order. */

SELECT AVG(totalProducts) AS AVGProducts
FROM (SELECT SUM(Quantity) AS totalProducts,
					  OrderID
		   FROM [Order Details]
		   GROUP BY OrderID)
AS OrderProductTotal
/* 28️⃣ Show products whose price is above the average price of all products. */

SELECT ProductName ,
			UnitPrice
FROM Products
WHERE UnitPrice>(SELECT AVG(UnitPrice)
							FROM Products)
ORDER BY UnitPrice DESC

/* 29️⃣ Calculate the total revenue generated by each product category. */

SELECT c.CategoryName , 
			SUM(od.Quantity * od.UnitPrice * (1-od.Discount)) AS revenueCategory
FROM Categories AS c
JOIN Products AS p
		ON p.CategoryID=c.CategoryID
JOIN [Order Details] AS od
		ON od.ProductID=p.ProductID
GROUP BY c.CategoryName
ORDER BY revenueCategory DESC


/* 30️⃣ Display countries that have no recorded orders. */

SELECT Country 
FROM Customers AS c
WHERE NOT EXISTS (SELECT 1
								FROM Orders AS o
								WHERE o.CustomerID=c.CustomerID)
/* =========================================
   End of SQL Practice Questions
   ========================================= */

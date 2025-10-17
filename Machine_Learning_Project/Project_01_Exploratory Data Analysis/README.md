# 🎬 Fandango Movie Ratings Analysis

این پروژه بر اساس مقاله [Be Suspicious Of Online Movie Ratings, Especially Fandango’s](https://fivethirtyeight.com/features/fandango-movies-ratings) از FiveThirtyEight و داده‌های منتشرشده در [مخزن GitHub](https://github.com/fivethirtyeight/data/tree/master/fandango) طراحی شده است.

## 📌 هدف پروژه

تحلیل و بررسی سیستم رتبه‌بندی فیلم‌ها در وب‌سایت Fandango و مقایسه آن با سایر پلتفرم‌های معتبر مانند IMDb، Metacritic و Rotten Tomatoes. هدف اصلی، بررسی صحت و دقت رتبه‌بندی‌ها و کشف هرگونه سوگیری یا خطای سیستمی در نمایش امتیازات به کاربران است.

## 📁 داده‌ها

پروژه از دو فایل داده استفاده می‌کند:

- `fandango_score_comparison.csv`: شامل امتیازات فیلم‌ها از منابع مختلف و حداقل ۳۰ رأی کاربر در Fandango.
- `fandango_scrape.csv`: شامل اطلاعات استخراج‌شده از صفحات HTML Fandango شامل تعداد ستاره‌ها، امتیاز واقعی و تعداد رأی‌دهندگان.

### ستون‌های مهم:

| ستون | توضیح |
|------|------|
| FILM | نام فیلم |
| Fandango_Stars | تعداد ستاره‌های نمایش‌داده‌شده در Fandango |
| Fandango_Ratingvalue | امتیاز واقعی فیلم از نظر کاربران |
| Fandango_Difference | اختلاف بین امتیاز واقعی و ستاره‌های نمایش‌داده‌شده |
| IMDB, Metacritic, RottenTomatoes | امتیازات از منابع دیگر، نرمال‌شده به مقیاس ۵ ستاره |

## 🔍 سوالات تحقیق

- آیا Fandango امتیازات فیلم‌ها را به‌صورت سیستماتیک بالا نشان می‌دهد؟
- تفاوت میان امتیاز واقعی و امتیاز نمایش‌داده‌شده چقدر است؟
- آیا این تفاوت در مقایسه با سایر پلتفرم‌ها قابل توجه است؟

## 🛠 ابزارها و تکنولوژی‌ها

- Python (Pandas, Matplotlib, Seaborn)
- Jupyter Notebook
- Git & GitHub

## 📊 نتایج اولیه

بر اساس مقاله FiveThirtyEight، مشخص شد که:
- تحلیل‌های آماری، نمودارهای مقایسه‌ای و مدل‌های پیش‌بینی امتیاز
- بیش از 98٪ فیلم‌ها در Fandango امتیازی بالاتر از 3 ستاره دارند.
- سیستم گرد کردن امتیازات در Fandango به‌جای گرد کردن به نزدیک‌ترین نیم‌ستاره، همیشه به بالا گرد می‌شود.
- این موضوع باعث افزایش مصنوعی امتیازات فیلم‌ها می‌شود.

## 📚 منابع

- مقاله اصلی: [FiveThirtyEight – Be Suspicious Of Online Movie Ratings](https://fivethirtyeight.com/features/fandango-movies-ratings)
- داده‌ها: [GitHub – Fandango Dataset](https://github.com/fivethirtyeight/data/tree/master/fandango)

---


# 🎬 Fandango Movie Ratings Analysis

An Exploratory Data Analysis (EDA) project that investigates potential biases in movie ratings displayed by Fandango and compares them with ratings from other popular movie platforms.

The project is inspired by FiveThirtyEight's article *"Be Suspicious Of Online Movie Ratings, Especially Fandango's"* and explores whether Fandango systematically inflates movie ratings presented to users.

---

# 🚀 Project Highlights

- Exploratory Data Analysis (EDA)
- Comparative Analysis of Movie Rating Systems
- Distribution Analysis of Movie Ratings
- Correlation Analysis
- Rating Difference Analysis
- Data Visualization using Seaborn and Matplotlib
- Identification of Potential Rating Biases

---

# 🔄 Project Workflow

```text
                 Fandango Dataset
                         │
                         ▼
                    Data Exploration
                         │
                         ▼
                    Data Cleaning
                         │
                         ▼
               Distribution Analysis
                         │
                         ▼
                 Rating Comparison
                         │
                         ▼
                Correlation Analysis
                         │
                         ▼
                 Difference Analysis
                         │
                         ▼
                 Visualization Results
                         │
                         ▼
                    Key Findings
```

---

# 📊 Dataset

The project uses two datasets published by FiveThirtyEight.

### Dataset Files

```text
fandango_score_comparison.csv

fandango_scrape.csv
```

The datasets contain movie ratings collected from several popular platforms, including:

- Fandango
- IMDb
- Rotten Tomatoes
- Metacritic

### Important Features

| Feature | Description |
|--------|------------|
| FILM | Movie title |
| Fandango_Stars | Displayed movie rating |
| Fandango_Ratingvalue | Actual user rating |
| Fandango_Difference | Difference between displayed and actual rating |
| IMDB | IMDb rating |
| Rotten Tomatoes | Rotten Tomatoes rating |
| Metacritic | Metacritic rating |

---

# 🔍 Research Questions

This project investigates the following questions:

- Does Fandango systematically display higher movie ratings?
- How large is the difference between actual and displayed ratings?
- Are Fandango ratings consistent with ratings from other movie platforms?
- Can rating distributions reveal potential biases in movie recommendations?

---

# 📈 Exploratory Data Analysis

The following analyses were performed:

- Distribution analysis of movie ratings
- Comparison between displayed and actual ratings
- Correlation analysis across different rating platforms
- Difference analysis of Fandango ratings
- Visualization of rating distributions

Several statistical and graphical methods were used to better understand rating behaviors across platforms.

---

# 📌 Key Findings

The analysis reveals that:

- Most movies receive ratings above three stars on Fandango.
- Displayed ratings tend to be higher than the corresponding rating values.
- Fandango ratings appear systematically higher when compared with other movie rating platforms.
- The rounding mechanism used by Fandango contributes to inflated displayed ratings.
- Movie ratings across platforms exhibit strong correlations, although noticeable differences remain in their displayed values.

These findings are consistent with concerns regarding potential rating inflation presented by FiveThirtyEight.

---

# 📉 Visualizations

The project includes various visualizations such as:

- Rating Distribution Plots
- Scatter Plots
- KDE Plots
- Correlation Analysis
- Comparative Rating Visualizations

These visualizations help identify differences between rating systems and provide insights into potential biases.

---

# 📂 Project Structure

```text
Fandango_Ratings_Analysis/

│
├── data/
├── notebooks/
├── images/
└── README.md
```

---

# 🛠 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

### Data Analysis Techniques

- Exploratory Data Analysis
- Distribution Analysis
- Correlation Analysis
- Comparative Analysis
- Data Visualization

---

# 📚 References

FiveThirtyEight Article:

> https://fivethirtyeight.com/features/be-suspicious-of-online-movie-ratings-especially-fandangos/

Dataset:

> https://github.com/fivethirtyeight/data/tree/master/fandango

---

# 👩‍💻 Author

### Ziba Hatamian

Junior Machine Learning Engineer

### Areas of Interest

- Machine Learning
- Deep Learning
- Data Science
- MLOps
- AI for Healthcare

GitHub:

```text
https://github.com/zziibbaa
```

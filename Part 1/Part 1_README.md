# Life Expectancy EDA — README

## Dataset Choice

**Dataset:** Life Expectancy (WHO) Dataset

**Source:** Kaggle (WHO Life Expectancy Dataset)

### Justification

This project uses the Life Expectancy (WHO) Dataset, which contains demographic, healthcare, economic, and social indicators collected for countries worldwide over multiple years.

This dataset was selected because it satisfies all assignment requirements:

- 2,938 records (well above the required minimum of 500 rows).
- More than 5 numerical features, including Adult Mortality, GDP, Schooling, BMI, Alcohol consumption, Population, HIV/AIDS, and others.
- At least 2 categorical features, including Country and Status.
- A continuous target variable (Life expectancy) suitable for regression.
- The same target can be converted into a binary classification problem by splitting it into Above Median and Below Median life expectancy classes.
- Contains realistic missing values, making it suitable for data cleaning, preprocessing, exploratory data analysis, feature engineering, traditional machine learning, ensemble methods, model evaluation, and LLM-powered prediction explanations.

For these reasons, this dataset provides a realistic end-to-end machine learning workflow while meeting every requirement of the assignment.


## Dataset Description

**Source:** World Health Organization (WHO) Global Health Observatory  
**File:** `Life_Expectancy_Data.csv`  
**Rows × Columns:** 2 938 × 22  
**Coverage:** 193 countries, years 2000–2015  

### Columns

| Column | Type | Description |
|--------|------|-------------|
| Country | category | Country name |
| Year | Int64 | Observation year (2000–2015) |
| Status | category | "Developed" or "Developing" |
| Life expectancy | float64 | Life expectancy at birth (years) — **target variable** |
| Adult Mortality | float64 | Adult mortality rate per 1 000 (ages 15–60) |
| infant deaths | int64 | Infant deaths per 1 000 population |
| Alcohol | float64 | Alcohol consumption per capita (litres, pure alcohol) |
| percentage expenditure | float64 | Health expenditure as % of GDP per capita |
| Hepatitis B | float64 | HepB immunisation coverage among 1-year-olds (%) |
| Measles | int64 | Reported measles cases per 1 000 |
| BMI | float64 | Average BMI of entire population |
| under-five deaths | int64 | Deaths under age 5 per 1 000 population |
| Polio | float64 | Polio (Pol3) immunisation among 1-year-olds (%) |
| Total expenditure | float64 | Government health expenditure as % of total expenditure |
| Diphtheria | float64 | DTP3 immunisation coverage among 1-year-olds (%) |
| HIV/AIDS | float64 | Deaths per 1 000 live births (HIV/AIDS, 0–4 years) |
| GDP | float64 | GDP per capita (USD) |
| Population | float64 | Country population |
| thinness 1-19 years | float64 | Prevalence of thinness in children 10–19 years (%) |
| thinness 5-9 years | float64 | Prevalence of thinness in children 5–9 years (%) |
| Income composition of resources | float64 | Human Development Index (0–1) |
| Schooling | float64 | Mean years of schooling |

---

## Task 1 – Load & Inspect

The dataset was loaded with `pd.read_csv()`. Column names were stripped of leading/trailing whitespace (several had hidden spaces). Shape confirmed: **2 938 rows × 22 columns**. All numeric columns loaded correctly as `float64` or `int64`; `Country` and `Status` were initially `object`.

---

## Task 2 – Null Value Analysis

### Null percentage table (all columns)

| Column | Null count | Null % |
|--------|-----------|--------|
| Country | 0 | 0.0 % |
| Year | 0 | 0.0 % |
| Status | 0 | 0.0 % |
| Life expectancy | 10 | 0.34 % |
| Adult Mortality | 10 | 0.34 % |
| infant deaths | 0 | 0.0 % |
| Alcohol | 194 | 6.60 % |
| percentage expenditure | 0 | 0.0 % |
| Hepatitis B | 553 | 18.82 % |
| Measles | 0 | 0.0 % |
| BMI | 34 | 1.16 % |
| under-five deaths | 0 | 0.0 % |
| Polio | 19 | 0.65 % |
| Total expenditure | 226 | 7.69 % |
| Diphtheria | 19 | 0.65 % |
| HIV/AIDS | 0 | 0.0 % |
| GDP | 448 | 15.25 % |
| **Population** | **652** | **22.19 %** |
| thinness 1-19 years | 34 | 1.16 % |
| thinness 5-9 years | 34 | 1.16 % |
| Income composition of resources | 167 | 5.68 % |
| Schooling | 163 | 5.55 % |

**Column exceeding 20 % null rate:** `Population` (22.19 %).  
`Population` was **not** imputed at this stage (handled separately in Task 9-a).

### Why the median rather than the mean for imputation

The median is the value that splits a distribution in half and is unaffected by extreme values. The mean, by contrast, is pulled toward outliers. In this dataset many columns (GDP, Population, infant deaths, Measles) are highly right-skewed, meaning a small number of very large values inflate the mean far above the typical country's value. Replacing a missing value with the mean would inject an artificially high number, distorting downstream statistics and model inputs. The median represents the "typical" country and keeps the imputed value within the realistic range of observed data.

---

## Task 3 – Duplicate Detection & Removal

`df.duplicated().sum()` returned **0 duplicate rows**. No rows were removed, so null percentages were unchanged.

---

## Task 4 – Data Type Correction

Country and Status were inferred as object, but they are repetitive categorical labels, so I converted them to category dtype. Year was converted to nullable Int64 for consistency.

No truly incorrect numeric-as-object column was found in this dataset after inspection.

**Memory impact:**  
- Before: **817 001 bytes**  
- After: **497 776 bytes**  
- Saved: **319 225 bytes (39.1 %)**  

Converting repetitive string columns to `category` reduces memory proportional to uniqueness; the saving is large because country names (~40 chars each) appeared 2 938 times before conversion.

---

## Task 5 – Descriptive Statistics & Skewness

### Skewness (sorted by absolute value)

| Column | Skewness |
|--------|----------|
| **Population** | **15.92** |
| infant deaths | 9.79 |
| under-five deaths | 9.50 |
| Measles | 9.44 |
| HIV/AIDS | 5.40 |
| percentage expenditure | 4.65 |
| GDP | 3.54 |
| Hepatitis B | −2.28 |
| Polio | −2.11 |
| Diphtheria | −2.08 |
| … | … |

**Most skewed column: `Population` (skewness = 15.92)**

### Interpretation of skewness for `Population`

A skewness of **+15.92** indicates an extreme positive (right) skew: the vast majority of countries have small populations (the distribution is concentrated near zero), but a handful of countries — India (~1.3 billion), China (~1.4 billion), the United States (~330 million) — create a very long right tail. On a histogram, the peak is located far to the left, with the tail stretching far to the right.

**Consequence for mean imputation:** The mean of `Population` is approximately **12.75 million**, while the median is only **1.39 million** — a ratio of nearly 9×. If missing population values were filled with the mean, every country receiving that imputed value would appear to be nearly 9× larger than the typical country. This would introduce systematic upward bias for all model features derived from population. The median (1.39 million) is far more representative of a "typical" country and should be used for imputation.

---

## Task 6 – Outlier Detection with IQR

### GDP

| Statistic | Value |
|-----------|-------|
| Q1 | 580.49 |
| Q3 | 4 779.41 |
| IQR | 4 198.92 |
| Lower bound | −5 717.89 |
| Upper bound | 11 077.78 |
| **Outlier rows** | **445** |

The lower bound is negative (impossible for GDP), so all outliers are on the upper end — high-income countries like Luxembourg and Norway. These values are real and economically meaningful, not data errors.

**Decision:** Outliers will be **retained** in Part 2. GDP is a key feature for life expectancy prediction; capping it would destroy the signal that wealthier nations achieve higher life expectancies. We may apply a log transformation in Part 2 to compress the range.

### Population

| Statistic | Value |
|-----------|-------|
| Q1 | 195 793 |
| Q3 | 7 420 359 |
| IQR | 7 224 566 |
| Lower bound | −10 641 055 (impossible) |
| Upper bound | 18 257 208 |
| **Outlier rows** | **294** |

All outliers are large countries such as China, India, and the United States. These are real observations.

**Decision:** Outliers will be **retained** and the column will receive a log transformation in Part 2. Population size is correlated with healthcare resource availability, so capping it would distort the relationship.

---

## Task 7 – Visualizations

### 7a – Line Plot: Global Average Life Expectancy by Year

<img width="984" height="484" alt="image" src="https://github.com/user-attachments/assets/febedb40-6461-4af1-9771-892fa54c2a34" />


The line plot shows global average life expectancy rising from approximately **66.8 years in 2000** to **71.6 years in 2015** — a steady upward trend of roughly 0.3 years per year. The improvement is broadly consistent, with no visible reversals, suggesting sustained global progress in healthcare, sanitation, and disease control over the period.

### 7b – Bar Chart: Mean Life Expectancy by Development Status

<img width="683" height="484" alt="image" src="https://github.com/user-attachments/assets/c129864a-1cb3-454e-843f-30674f796c39" />


Developed countries average approximately **79.2 years**, compared to **67.1 years** for developing countries — a gap of more than 12 years. The chart makes this disparity immediately visible and confirms that development status will be an important predictor in Part 2.

### 7c – Histogram: Population (Most Skewed Column)

<img width="884" height="484" alt="image" src="https://github.com/user-attachments/assets/d19ee944-ebb6-442a-ab33-9ff9dba67573" />

The distribution of `Population` is extremely right-skewed (skewness = 15.92). The histogram shows nearly all observations clustered in the lowest bin (small and medium-sized countries), with the bars dropping steeply and barely visible bars at higher values. The KDE curve has a sharp spike near zero with a very long, flat right tail extending into the billions. This shape confirms that the mean (12.75 million) is completely unrepresentative of the typical observation and that the median (1.39 million) must be used for imputation.

### 7d – Scatter Plot: GDP vs Life Expectancy

<img width="884" height="584" alt="image" src="https://github.com/user-attachments/assets/5f2f4df4-3843-413d-aabb-55fb97a694cb" />

The scatter plot shows a **positive, non-linear** relationship: countries with higher GDP per capita tend to have higher life expectancy. However, the relationship is steep at low GDP values (increasing GDP from near-zero to ~$5 000 yields large life-expectancy gains) and flattens at higher GDP levels (going from $30 000 to $100 000 produces only modest additional gains). Developing countries (orange) cluster in the lower-left, while developed countries (blue) cluster in the upper-right. This concave shape suggests a **logarithmic** transformation of GDP would linearise the relationship and improve Part 2 model fit.

### 7e – Box Plot: Life Expectancy by Development Status

<img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/17dedc7d-d67f-4d9f-911c-8fbb5c4a172c" />

The box plot reveals:  
- **Developed** countries: median ≈ 81 years, narrow interquartile range (IQR ≈ 4 years), very few outliers.  
- **Developing** countries: median ≈ 70 years, wide IQR (≈ 14 years), and many low outliers representing countries with severe disease burdens or conflict.

The large spread within the "Developing" group reflects the heterogeneity of that category — it includes both high-middle-income countries approaching developed-world outcomes and low-income countries still facing high child mortality.

---

## Task 8 – Correlation Heat Map

The Pearson correlation matrix was computed for all numeric columns and visualised as a heat map.

<img width="1478" height="1283" alt="image" src="https://github.com/user-attachments/assets/7266c6ad-2254-4d5e-a34b-10fd349a97eb" />

**Highest absolute correlation: `infant deaths` ↔ `under-five deaths` (r = 0.997)**

This near-perfect correlation makes sense: under-five deaths is a broader measure that includes infant deaths (deaths in the first year of life) as a subset. They are not independent measurements — they share a large portion of their counts.

**Is this a causal relationship?** Not in the usual sense. Both variables measure overlapping portions of child mortality. The correlation is definitional/compositional rather than causal. In Part 2, including both columns would introduce near-perfect multicollinearity; one should be dropped or the pair replaced by a single child-mortality composite.

**Alternative explanations / confounders:**  
- A third variable — country development status — drives both. Developing countries with poor healthcare systems have both high infant mortality and high under-five mortality; developed countries have low values of both. The correlation would remain high even if infant and under-five deaths were measured completely independently.

---

## Task 9-a – Imputation Strategy Comparison

The two highest-skewness columns are **`Population`** (skew = 15.92) and **`infant deaths`** (skew = 9.79). Both are strongly positively skewed.

| Column | Skewness | Mean | Median |
|--------|----------|------|--------|
| Population | 15.92 | 12 753 375 | 1 386 542 |
| infant deaths | 9.79 | 30.30 | 3.00 |

**Chosen statistic: Median for both columns.**

**Justification:**  
- **Positively skewed distributions** have a mean that is pulled upward by a small number of extreme high values. The mean is therefore higher than what a "typical" country looks like. Imputing missing values with the mean would insert an artificially inflated value, biasing the distribution toward the outlier region.  
- The **median** is the 50th percentile and is unaffected by extreme values. It represents the middle of the distribution — the value a randomly chosen country is closest to.  
- For `Population`: the mean (12.75 M) is nearly 9× the median (1.39 M). Using the mean would make every imputed country appear to be one of the world's larger nations.  
- For `infant deaths`: the mean (30.3) is 10× the median (3.0), again driven by a handful of countries with catastrophically high infant mortality.

After imputation, `df[["Population", "infant deaths"]].isnull().sum()` confirmed **0 remaining nulls** in both columns.

---

## Task 9-b – Spearman vs Pearson Correlation

### Three pairs with largest |Spearman − Pearson| difference

| Pair | Spearman | Pearson | |Difference| |
|------|----------|---------|------------|
| under-five deaths ↔ HIV/AIDS | higher | lower | **0.474** |
| infant deaths ↔ HIV/AIDS | higher | lower | **0.461** |
| infant deaths ↔ Income composition | higher | lower | **0.419** |

### Interpretations

**Pair 1: `under-five deaths` ↔ `HIV/AIDS`**  
|Spearman| > |Pearson|, meaning the relationship is **monotonic but non-linear**. Countries with higher HIV/AIDS death rates consistently have higher under-five mortality, but the relationship is not proportional — at high HIV/AIDS values, under-five deaths increase steeply (driven by sub-Saharan African countries), while at low values the relationship is flat. Pearson misses this because it assumes proportionality. **For Part 2 feature selection, Spearman will guide this pair** — it better captures the true monotonic association.

**Pair 2: `infant deaths` ↔ `HIV/AIDS`**  
Same pattern. The HIV/AIDS epidemic disproportionately affects mother-to-child transmission, driving infant mortality up in severely affected countries. The non-proportional, rank-consistent relationship makes Spearman the more informative measure here. **Spearman will guide Part 2 selection** for this pair.

**Pair 3: `infant deaths` ↔ `Income composition of resources`**  
|Spearman| > |Pearson|, indicating a **monotonic but non-linear** negative relationship: countries with higher HDI scores consistently have lower infant deaths, but the marginal reduction in infant deaths per unit of HDI gain is not constant. **Spearman will guide Part 2** for this pair as well.

In all three cases the rank-based Spearman measure is preferred because the raw values contain extreme outliers that suppress the Pearson coefficient, understating the true consistency of the relationship.

---

## Task 9-c – Grouped Aggregation

**Categorical column:** `Status`  
**Numeric column:** `Life expectancy`

| Status | Mean | Std | Count |
|--------|------|-----|-------|
| Developed | 79.20 | 4.07 | 512 |
| Developing | 67.13 | 8.99 | 2 426 |

**Highest-mean group:** `Developed` (mean life expectancy = 79.20 years)  
**Highest-std group:** `Developing` (std = 8.99 years)

**Ratio of highest to lowest mean:** 79.20 / 67.13 = **1.180** (~18 % higher for Developed).

### Implications for modelling

- The 18 % difference in group means is large in absolute terms (12 years of life expectancy). This confirms that `Status` carries meaningful predictive signal on its own.
- However, the **standard deviation within the Developing group (8.99 years)** is more than twice that of the Developed group (4.07 years). High within-group variance means that knowing a country is "Developing" alone is insufficient to reliably predict its life expectancy — a developing country could have life expectancy anywhere from the low 40s (conflict-affected or high-HIV sub-Saharan nations) to the high 70s (upper-middle-income Latin American or East Asian nations). The feature carries signal at the group-mean level but must be combined with other features (GDP, HIV/AIDS rate, schooling) to discriminate within the Developing category.

---

## Cleaning Summary

| Step | Action | Rows affected |
|------|--------|---------------|
| Column strip | Removed whitespace from headers | — |
| Null imputation (< 20 %) | Filled 18 numeric columns with column median | ~1 952 cells |
| Duplicate removal | None found | 0 |
| Type conversion | `Year` → Int64; `Status`, `Country` → category | Memory −39.1 % |
| Population imputation (Task 9-a) | Filled with median (1 386 542) | 652 cells |
| infant deaths imputation (Task 9-a) | Already 0 nulls after Task 2; confirmed | 0 cells |

Final cleaned file: **`cleaned_data.csv`** (2 938 rows × 22 columns, 0 nulls except none).

---

## Files in this Repository

| File | Description |
|------|-------------|
| `Life_Expectancy_Data.csv` | Original raw dataset (WHO) |
| `Part 1.ipynb` | Executed Jupyter notebook — all tasks, outputs, and plots |
| `cleaned_data.csv` | Cleaned dataset for use in Parts 2 & 3 |
| `Part 1_README.md` | This file |

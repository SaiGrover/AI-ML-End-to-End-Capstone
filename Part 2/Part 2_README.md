# Part 2 —  README

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


## Label Definitions

- **`y_reg`** (regression target): `Life expectancy` — the continuous numeric target from the cleaned dataset.
- **`y_clf`** (classification target): a binary label derived by binarizing `y_reg` at its median —
  `y_clf = (y_reg > y_reg.median()).astype(int)`, where the median life expectancy is **72.10 years**.
  `1` = above-median life expectancy, `0` = at-or-below-median.

**Feature matrix `X`:** all columns from `cleaned_data.csv` except the target (`Life expectancy`) and `Country`.
`Country` was dropped because it has 193 unique values — far too high-cardinality to one-hot encode without exploding the feature space and overfitting, and label-encoding it would impose a meaningless numeric ordering on country names.
Although one-hot encoding is generally used for unordered categorical variables, Country was intentionally excluded because its 193 unique categories would create an extremely sparse feature matrix with little generalization ability and a high risk of overfitting.

---

## Categorical Encoding

`Status` is the only categorical column in `X` (`Country` was already dropped). It has exactly two levels: **"Developing"** and **"Developed"**.

**Encoding chosen: Label encoding** — `Developing = 0`, `Developed = 1`.

**Justification:** "Developed" represents a genuinely higher level of economic and public-health development than "Developing" — there is a real, meaningful order between the two categories (this is essentially the WHO's own ordinal classification of national development status). Because there are only two levels, label encoding and one-hot encoding produce mathematically equivalent information here (a single 0/1 column either way), so we use the simpler label-encoded form.
Since Status contains only two naturally ordered categories, label encoding and one-hot encoding would carry the same information, making label encoding the simpler choice.

No other categorical columns required one-hot encoding in this dataset. Had a column existed with unordered categories (e.g., a "Region" column with no natural ranking), one-hot encoding (`pd.get_dummies(..., drop_first=True)`) would be the correct choice, because label encoding such a column would force an artificial numeric order onto it — a model would incorrectly treat "Region 3" as twice "Region 1.5" or otherwise interpret the encoded values as having a meaningful magnitude/distance relationship that does not exist between unordered categories. One-hot encoding avoids this by representing each category as an independent binary indicator with no implied ranking.

---

## Leak-Free Train-Test Split & Scaling

`X`, `y_reg`, and `y_clf` were split together using `train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)`, producing 2 350 training rows and 588 test rows.

A `StandardScaler` was fit **only** on `X_train` (`scaler.fit(X_train)`), then used to transform both `X_train` and `X_test` (`scaler.transform(...)`).

**Why fitting on the full dataset would be data leakage:** `StandardScaler` computes the mean and standard deviation of each feature from whatever data it is fit on. If it were fit on the combined train+test set, the resulting mean/standard-deviation parameters would be influenced by the test rows. Every subsequent transformation — including of the training set — would then implicitly encode information about the test set's distribution. The model would be trained on features that have "seen" statistics derived in part from data it is later evaluated on, producing an overly optimistic, non-generalizable estimate of performance. Fitting the scaler on the training set only and reusing those exact parameters on the test set mimics the real-world deployment scenario, where a model encounters genuinely new data whose distribution was never used to calibrate the model.

The StandardScaler transformation is `z = (x - μ) / σ`, where `μ` and `σ` must be learned from the training set only.

---

## Linear Regression

| Metric | Value |
|--------|-------|
| MSE | 15.2923 |
| R² | 0.8236 |

The model explains **82.4%** of the variance in life expectancy on the held-out test set.

### Top-3 |coefficient| features

| Feature | Coefficient |
|---------|-------------|
| under-five deaths | −11.04 |
| infant deaths | +10.85 |
| Adult Mortality | −2.60 |

### Interpreting coefficients (scaled features)

**Because features were standardized, coefficients are measured per one standard deviation increase rather than one raw unit.** Each coefficient represents the change in predicted life expectancy (in years) associated with a **one standard-deviation increase** in that feature, holding all other features constant.

- **A large positive coefficient** (e.g., `infant deaths`, +10.85) means that as this scaled feature increases by one standard deviation, the model's predicted life expectancy increases by about 10.85 years, all else equal. This sign is counter-intuitive on its face (more infant deaths should *lower* life expectancy) and is explained below.
- **A large negative coefficient** (e.g., `under-five deaths`, −11.04) means a one-standard-deviation increase in this feature is associated with an 11.04-year *decrease* in predicted life expectancy, all else equal — the expected, intuitive direction for a mortality-related variable.

**Important caveat:** `infant deaths` and `under-five deaths` have a Pearson correlation of **0.997** (identified in Part 1, Task 8) — under-five deaths definitionally includes infant deaths as a subset. This near-perfect multicollinearity makes the individual OLS coefficients unstable and difficult to interpret in isolation: the model can trade off magnitude between the two correlated features almost arbitrarily while keeping their *combined* effect roughly constant and consistent with intuition (more child deaths of any kind → lower life expectancy). The large opposite-signed coefficients are a symptom of multicollinearity, not a meaningful claim that infant deaths individually *raise* life expectancy. This instability is one of the motivations for the Ridge regression in the next section.

---

## Ridge Regression (alpha = 1.0)

| Model | MSE | R² |
|-------|-----|----|
| Linear Regression (OLS) | 15.2923 | 0.8236 |
| Ridge (alpha = 1.0) | 15.3267 | 0.8232 |

### Why Ridge produces a different coefficient profile, and what `alpha` controls

Ridge regression adds an L2 penalty term (`alpha * sum(coefficient^2)`) to the OLS loss function. This penalty discourages any single coefficient from growing very large, shrinking all coefficients toward zero in proportion to their magnitude. `alpha` controls the strength of this penalty: `alpha = 0` recovers plain OLS, while larger `alpha` shrinks coefficients more aggressively.

In this dataset, Ridge visibly moderates the extreme, multicollinearity-driven coefficients of `infant deaths` (10.85 → 9.53) and `under-five deaths` (−11.04 → −9.72) — pulling both toward more conservative, stable values without changing their sign. This is exactly Ridge's intended behavior when predictors are highly correlated: rather than letting two collinear features fight for credit with large, oppositely-signed coefficients, Ridge spreads the "explanatory burden" more evenly between them, producing coefficients that are more stable and trustworthy for interpretation. With `alpha = 1.0` here the regularization is mild, so MSE and R² are nearly identical to OLS (15.33 vs 15.29, 0.8232 vs 0.8236) — the penalty mainly reshaped the coefficient *profile*, not the predictive accuracy.

---

## Logistic Regression — Class Imbalance Check

| Class | Count |
|-------|-------|
| 1 (above median) | 1 178 |
| 0 (at/below median) | 1 172 |

Minority class proportion: **49.87%** — well above the 35% imbalance threshold. **No resampling (SMOTE) or `class_weight='balanced'` was needed**, because binarizing a continuous variable at its own median by construction produces an almost perfectly balanced split (50/50 up to rounding from ties at the median value). This is the expected and confirmed before/after picture: the "after" class counts are identical to "before" since no resampling step was triggered.

---

## Logistic Regression (C = 1.0, baseline)

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|--|---|---|
| **Actual 0** | 271 (TN) | 42 (FP) |
| **Actual 1** | 25 (FN) | 250 (TP) |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.92 | 0.87 | 0.89 | 313 |
| 1 | 0.86 | 0.91 | 0.88 | 275 |
| **Accuracy** | | | **0.89** | 588 |

### Precision / Recall Formulas

Using TP (true positives), FP (false positives), FN (false negatives), for the positive class (1 = above-median life expectancy):

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

For class 1: Precision = 250 / (250 + 42) = **0.856**, Recall = 250 / (250 + 25) = **0.909** (matching the classification report above).

### Which metric matters more for this task?

In this context, the positive class (1) represents "above-median life expectancy" — a country-level health outcome classification, likely used to flag countries for policy attention, resource allocation, or further investigation. A **false negative** here means a country whose life expectancy is genuinely above median gets incorrectly flagged as below-median (or vice-versa, depending on framing), while a **false positive** means a below-median country is misclassified as above-median.

For a task framed around identifying at-risk or under-performing countries (i.e., flagging the *below-median* group for intervention), **recall on the "below-median" class matters most** — missing a country that genuinely needs attention (a false negative for the at-risk class) is more costly than incorrectly flagging a country that is actually fine (a false positive), since the consequence of a missed at-risk country is a continued lack of intervention. In this symmetric binary setup, both classes are reasonably well-served (precision and recall for both classes sit in the high 0.8s–low 0.9s), but if the downstream use case is specifically "identify struggling countries," recall on class 0 (correctly catching all below-median countries) should be prioritized, even at some cost to precision.

### AUC

**AUC = 0.957**

<img width="784" height="684" alt="image" src="https://github.com/user-attachments/assets/bcdc463f-b684-4e72-bfb4-fabdda7a840f" />

Figure shows the ROC curve.

An AUC of 0.957 means that if we randomly pick one country with above-median life expectancy and one with below-median life expectancy, the model assigns a higher predicted probability to the above-median country about **95.7%** of the time. This is a very strong separation between the two classes — the model's probability scores rank countries by their health outcome status with high reliability, well above the 0.5 AUC of a random classifier.

---

## Decision-Threshold Sensitivity (0.30 → 0.70)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.30 | 0.7557 | 0.9673 | 0.8485 |
| 0.40 | 0.8081 | 0.9491 | 0.8729 |
| **0.50** | **0.8562** | **0.9091** | **0.8818** |
| 0.60 | 0.8947 | 0.8655 | 0.8799 |
| 0.70 | 0.9267 | 0.7818 | 0.8481 |

### Precision / Recall Formulas (repeated for this section)

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**(b) F1-maximizing threshold:** **0.50**, with F1 = 0.8818. This is also the default scikit-learn threshold, so the model's default behavior happens to already be near-optimal for balanced F1 on this test set.

**(c) Precision vs Recall — which matters more here:** As discussed above, if the downstream goal is to identify countries needing health-policy attention (i.e., correctly catching as many below-median, at-risk countries as possible), **recall is more important** than precision — a missed at-risk country (false negative) represents a continued lack of intervention, which is more costly than incorrectly flagging a country that turns out to be fine (false positive, which simply triggers extra (low-cost) review).

**(d) Threshold direction and cost:** To prioritize recall for the positive ("above-median") class, we would **lower** the threshold below 0.50 — e.g., to 0.30, which raises recall from 0.909 to 0.967. The cost of doing so is a drop in precision from 0.856 to 0.756: more below-median countries get incorrectly classified as above-median (more false positives), meaning some genuinely at-risk countries would be missed under a "flag low scorers" framing, or — under the alternate framing of flagging the positive class for resource allocation — more resources would be allocated to countries that don't strictly need them. The threshold choice should ultimately be set based on the relative real-world cost of a missed at-risk country versus a wasted resource allocation, which is a policy decision outside the scope of the model itself.

---

## Regularization Experiment: C = 0.01 vs C = 1.0

| Model | Precision | Recall | AUC |
|-------|-----------|--------|-----|
| C = 1.0 (baseline) | 0.8562 | 0.9091 | 0.9569 |
| C = 0.01 (strong regularization) | 0.8350 | 0.9018 | 0.9476 |

### What `C` controls, and the effect of reducing it

In scikit-learn's `LogisticRegression`, `C` is the **inverse** of the regularization strength: `C = 1 / lambda`, where `lambda` scales the L2 penalty on the coefficients. A **smaller `C`** means a **larger** penalty, forcing coefficients to shrink more aggressively toward zero (stronger regularization); a **larger `C`** means a weaker penalty, allowing the model to fit the training data more closely (approaching unregularized logistic regression as `C → ∞`).

Reducing `C` from 1.0 to 0.01 **worsened** performance on this dataset across every metric reported: precision dropped from 0.856 to 0.835, recall dropped from 0.909 to 0.902, and AUC dropped from 0.957 to 0.948. This indicates that the baseline model (`C=1.0`) was not meaningfully overfitting — the moderate amount of regularization implicit in `C=1.0` was already appropriate for this feature set and sample size, and forcing much stronger regularization (`C=0.01`) caused the model to underfit slightly, shrinking useful coefficients (including the genuinely informative mortality-related features) more than necessary and degrading its ability to separate the two classes.

---

## Bootstrap Confidence Interval for AUC Difference

500 bootstrap samples were drawn from the test set with replacement (`np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)`, `random_state` seeded for reproducibility). For each sample, the AUC of the `C=1.0` model and the `C=0.01` model were computed on the resampled rows, and the difference (`C=1.0` AUC − `C=0.01` AUC) was recorded.

| Statistic | Value |
|-----------|-------|
| Mean AUC difference | **+0.0093** |
| 95% CI lower bound (2.5th percentile) | **+0.0038** |
| 95% CI upper bound (97.5th percentile) | **+0.0150** |

**The 95% confidence interval [0.0038, 0.0150] excludes zero.** Since the 95% confidence interval does not cross zero, we have statistical evidence that the baseline model consistently outperforms the stronger regularized model. While the magnitude of the advantage is modest (roughly 0.4 to 1.5 AUC points), it is consistent: in every one of the 500 bootstrap resamples evaluated, the direction of the difference supports `C=1.0` outperforming `C=0.01`, confirming that the weaker-regularization model is the better choice for this dataset and feature set.

---

## Files in this Repository

| File | Description |
|------|-------------|
| `cleaned_data.csv` | Cleaned dataset from Part 1 (input to this analysis) |
| `Part 2.ipynb` | Executed Jupyter notebook — all preprocessing, model training, evaluation, and plots |
| `Part 2_README.md` | This file |

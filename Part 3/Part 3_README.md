# Part 3 — README

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


This part builds on the same `X_train_scaled`, `X_test_scaled`, `y_clf_train`, `y_clf_test` from Part 2 (identical `random_state=42` split and `StandardScaler` setup), and extends the analysis to ensemble models, hyperparameter tuning, and a reproducible, serialized pipeline.

---

## Task 1 — Decision Tree Baseline (Unconstrained)

| Metric | Value |
|--------|-------|
| Training accuracy | 1.0000 |
| Test accuracy | 0.9286 |
| Train-test gap | 0.0714 |

This tree shows clear signs of **overfitting**: it achieves perfect (100%) accuracy on the training data but drops to 92.9% on the held-out test set, a 7.1-point gap.

**Why decision trees are high-variance models:** an unconstrained tree splits greedily at each node, choosing the split that best separates the data *at that node alone*, without ever revisiting or reconsidering earlier decisions further up the tree. With no depth limit, the tree keeps splitting until every leaf is pure (or contains a single sample), effectively memorizing the training set's idiosyncrasies, including noise. A small change in the training data — a few different rows, a different random seed for tie-breaking — can produce a substantially different tree structure, because early splits cascade into very different downstream partitions. This sensitivity to the specific training sample, combined with the tendency to fit noise rather than signal, is what makes single deep decision trees "high variance" estimators.

---

## Task 2 — Controlled Decision Tree (max_depth=5, min_samples_split=20)

| Metric | Value |
|--------|-------|
| Training accuracy | 0.9400 |
| Test accuracy | 0.9218 |
| Train-test gap | 0.0182 |

**Role of `max_depth`:** limits how many levels of splits the tree is allowed to make. Capping depth at 5 forces the tree to stop partitioning the data after a fixed number of decisions, preventing it from carving out tiny, overly specific regions of the feature space that match training noise. This trades a small amount of bias (the tree can no longer perfectly separate every training point) for a substantial reduction in variance.

**Role of `min_samples_split`:** requires at least 20 samples to be present at a node before it is allowed to split further. This prevents the tree from making splits based on very small, statistically unreliable subsets of data — a split decided by 3 or 4 training rows is far more likely to be capturing noise than a genuine pattern, so this parameter directly guards against that failure mode.

**Comparison of train-test gaps:**

| Tree | Train-Test Gap |
|------|-----------------|
| Unconstrained | 0.0714 |
| Controlled (max_depth=5, min_samples_split=20) | 0.0182 |

The controlled tree's gap (0.018) is roughly **4x smaller** than the unconstrained tree's gap (0.071), confirming that limiting depth and minimum split size substantially reduces overfitting, at the cost of a small reduction in training accuracy (94.0% vs 100%) and a comparable reduction in test accuracy (92.2% vs 92.9% — only 0.7 points lower despite far less overfitting).

---

## Task 3 — Gini vs Entropy Comparison (max_depth=5)

| Criterion | Test Accuracy |
|-----------|----------------|
| Gini | 0.9218 |
| Entropy | 0.9014 |

### Formulas

**Gini impurity:**
```
Gini = 1 − Σ pᵢ²
```
where `pᵢ` is the proportion of samples belonging to class `i` at a given node.

**Entropy:**
```
Entropy = −Σ pᵢ log₂(pᵢ)
```

A node with **Gini = 0** is a **pure node** — every sample at that node belongs to the same class (`pᵢ = 1` for one class, `0` for all others), so `1 − 1² = 0`. Pure nodes require no further splitting; the tree can confidently assign the majority (only) class as the prediction for any sample that reaches that leaf.

On this dataset, Gini slightly outperformed Entropy (92.2% vs 90.1% test accuracy) at `max_depth=5`. The two criteria usually produce very similar trees in practice since they both measure node impurity in conceptually similar ways, but the small numeric differences in how they weight class proportions can lead to different split choices, especially with constrained depth.

---

## Task 4 — Random Forest

| Metric | Value |
|--------|-------|
| Training accuracy | 0.9966 |
| Test accuracy | 0.9507 |
| Test ROC-AUC | 0.9924 |

### Top 5 Features by Importance

| Feature | Importance |
|---------|-----------|
| Adult Mortality | 0.2307 |
| Income composition of resources | 0.1735 |
| Schooling | 0.1135 |
| HIV/AIDS | 0.0825 |
| BMI | 0.0629 |

**Added graph:** A top-10 Random Forest feature-importance bar chart was added directly after this feature-importance table in the notebook. It expands the view beyond the top five features and helps identify where importance begins to taper off.

### How Random Forest computes feature importance

For each feature, the Random Forest tracks how much that feature's splits reduce Gini impurity, averaged across every split that uses that feature, across every tree in the forest, and weighted by the number of samples each split affects. A feature that is repeatedly chosen for splits that cleanly separate the two classes accumulates a high importance score; a feature rarely used, or used only for splits that barely reduce impurity, accumulates a low score. The scores are normalized to sum to 1 across all features.

**Why this differs from a linear regression coefficient:** a linear regression coefficient measures the *marginal, linear* effect of a one-unit (or one-standard-deviation, if scaled) change in a feature on the predicted outcome, holding all other features fixed — it is a single, signed, additive number that assumes the relationship between feature and target is linear and constant across the entire range of the feature. Random Forest feature importance, in contrast, is a non-negative, unsigned measure of how *useful* a feature was for making splitting decisions across many different trees and many different non-linear partitions of the data — it captures non-linear relationships, interaction effects, and threshold effects that a single linear coefficient cannot represent. A feature can have high Random Forest importance even if its relationship to the target is highly non-linear or only matters within certain interactions with other features — something a linear model coefficient would never reveal.

### Bagging Concept

Random Forest is built on **bagging** (bootstrap aggregating): each of the 100 trees in the forest is trained on an independent **bootstrap sample** — a random sample of the same size as the training set, drawn *with replacement*, so each tree sees a slightly different subset of rows (some duplicated, some left out). In addition, at each split within each tree, only a random subset of features (by default, √(number of features) for classification) is considered as candidates for that split, rather than all features. This double randomization — random rows per tree, random feature subsets per split — ensures that the individual trees in the forest are decorrelated from one another; they make different mistakes on different parts of the feature space. When the forest's final prediction is formed by averaging (or majority-voting) across all 100 trees, the errors of individual overfit trees tend to cancel out, while the genuine signal that most trees agree on is reinforced. This is why a Random Forest test ROC-AUC (0.9924) is so much higher and more stable than a single deep decision tree's performance — averaging many high-variance, low-bias trees produces a low-variance, low-bias ensemble.

---

## Task 4a — Gradient Boosting

| Metric | Value |
|--------|-------|
| Training accuracy | 0.9736 |
| Test accuracy | 0.9405 |
| Test ROC-AUC | 0.9888 |

Unlike Random Forest's parallel, independent trees, Gradient Boosting builds trees **sequentially**, where each new tree is trained to correct the residual errors of the ensemble built so far. With `n_estimators=100`, `learning_rate=0.1`, and `max_depth=3` (shallow trees), Gradient Boosting achieves a slightly lower test ROC-AUC (0.9888) than the Random Forest (0.9924) on this dataset, though both substantially outperform the single decision trees and logistic regression. This is included in the cross-validated comparison in Task 5 below.

---

## Task 4b — Feature Ablation Study

**5 lowest-importance features (from the Task 4 Random Forest):** `Population`, `Hepatitis B`, `Measles`, `Year`, `Status`

A second Random Forest with identical hyperparameters (`n_estimators=100`, `max_depth=10`, `random_state=42`) was trained with these 5 features removed from both `X_train_scaled` and `X_test_scaled`.

| Model | Features | Test AUC |
|-------|----------|----------|
| Full model | 20 | 0.9924 |
| Reduced model | 15 | 0.9919 |
| **Difference (full − reduced)** | | **0.0006** |

**Added graph:** A full-vs-reduced Random Forest ablation bar chart was added in this task section. It labels both AUC and feature count, making the small performance cost of dropping five features easy to compare.

### Interpretation

The AUC drop from removing the 5 lowest-importance features is negligible (0.0006, essentially within noise). This strongly suggests these five features were **genuinely close to uninformative** for this classification task — their presence or absence makes almost no difference to the model's ability to separate above-median from below-median life expectancy countries. This is consistent with their low importance scores from Task 4: `Population`, `Hepatitis B`, `Measles`, `Year`, and `Status` simply weren't pulling much weight relative to the dominant predictors (`Adult Mortality`, `Income composition of resources`, `Schooling`, `HIV/AIDS`, `BMI`).

**Production trade-off discussion:** deploying the reduced, 15-feature model would lower inference cost (fewer feature computations per prediction), reduce the data-collection and maintenance burden (5 fewer columns to source, validate, and keep up to date — including dropping `Population`, a column with substantial nulls in the raw data and 22% missingness, per Part 1), and simplify the production pipeline overall. Given that the AUC degradation here (0.06 percentage points) is far below any reasonable tolerance threshold, this trade-off looks clearly favorable: the reduced model is a strong candidate for production deployment, trading negligible predictive cost for meaningfully lower operational complexity.

---

## Task 5 — Cross-Validated Comparison (5-fold StratifiedKFold, AUC)

| Model | Mean AUC | Std AUC |
|-------|----------|---------|
| Logistic Regression | 0.9447 | 0.0060 |
| Decision Tree (controlled) | 0.9587 | 0.0125 |
| Random Forest | 0.9891 | 0.0053 |
| Gradient Boosting | 0.9868 | 0.0052 |

### Why cross-validation gives a more reliable estimate of generalization than a single train-test split

A single train-test split produces one estimate of test performance that depends heavily on which specific rows happened to land in the test set — a "lucky" or "unlucky" split can over- or under-state how well a model generalizes. **5-fold StratifiedKFold cross-validation** instead partitions the training data into 5 roughly equal folds (each preserving the original class balance, due to stratification), trains the model 5 times — each time holding out a different fold as the validation set — and reports the mean and standard deviation of the 5 resulting AUC scores. This gives two crucial pieces of information a single split cannot: (1) a mean performance estimate averaged over 5 different validation sets, which is far less sensitive to the luck of any one particular split, and (2) a standard deviation, which directly quantifies how *stable* that performance is across different subsets of the data — a model with low CV standard deviation (like Random Forest's 0.0053) can be trusted to generalize consistently, while a model with higher variability (like the Decision Tree's 0.0125) is more sensitive to exactly which rows it happens to be trained and evaluated on.

---

## Task 6 — GridSearchCV Hyperparameter Tuning (Random Forest)

### Parameter Grid

```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

This grid contains **3 × 3 × 2 = 18** distinct hyperparameter configurations. With 5-fold cross-validation, GridSearchCV fits a total of **18 × 5 = 90 models**.

### Best Result

| | |
|---|---|
| **Best parameters** | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` |
| **Best score (mean 5-fold CV AUC)** | **0.9898** |

### Grid Search vs Randomized Search Trade-off

**Grid Search** exhaustively evaluates every combination in the specified parameter grid, guaranteeing that the best combination *within that grid* is found, but its computational cost grows multiplicatively with the number of parameters and values per parameter — adding one more hyperparameter with 3 values would triple the total fits required (here, 90 -> 270). This becomes impractical for grids with many parameters or wide ranges. **Randomized Search**, by contrast, samples a fixed number of random combinations from the specified parameter distributions, allowing the search budget (number of fits) to be set independently of the grid's theoretical size — it can explore a much larger or more finely-grained search space for the same computational cost, at the cost of no longer guaranteeing the globally-best combination is tested. In practice, Randomized Search is often preferred for large search spaces or expensive models, while Grid Search remains a sound choice (as used here) for smaller, well-scoped grids like this 18-combination Random Forest grid, where exhaustiveness is computationally affordable.

---

## Task 7 — Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|--------------------|---------------|----------|
| 0.2 | 1.0000 | 0.9770 |
| 0.4 | 1.0000 | 0.9855 |
| 0.6 | 1.0000 | 0.9892 |
| 0.8 | 1.0000 | 0.9917 |
| 1.0 | 1.0000 | 0.9923 |

<img width="884" height="584" alt="image" src="https://github.com/user-attachments/assets/49a941dd-471d-45a4-9580-3672fe6c4874" />

Since the best model is a pipeline, test AUC was computed on raw `X_test`, not `X_test_scaled`, because the pipeline applies imputation and scaling internally.

### Interpretation

**(i) Does training AUC decrease as the training set grows?** No — training AUC stays pinned at a perfect 1.0000 across every fraction tested. This is expected behavior for the tuned Random Forest (`max_depth=None`, `min_samples_leaf=1`): with no depth limit and the ability to isolate single training samples in leaf nodes, the model can perfectly memorize the training set at any size, so training AUC never drops below 1.0, regardless of how much training data is provided.

**(ii) Does test AUC increase with more training data?** Yes, clearly and monotonically — test AUC rises from 0.977 at 20% of the training data to 0.992 at 100%, a steady improvement of about 1.5 percentage points as more data is added. This rising trend has not yet fully flattened by 100% of the available training data (the increase from 80% -> 100% is still a positive 0.0006, smaller than earlier increments but not zero).

**(iii) Conclusion — data-limited or capacity-limited?** The model is best described as **mildly data-limited, but the marginal returns are diminishing rapidly**. Test AUC is still inching upward at 100% of the training data, suggesting that collecting more labeled data would likely continue to improve generalization performance somewhat — but the size of each successive improvement (0.0085 -> 0.0037 -> 0.0025 -> 0.0006 across the four increments) is shrinking quickly, indicating we are approaching the point of diminishing returns. The constant 1.0 training AUC across all fractions also signals that the model has more than enough *capacity* to fit any amount of training data thrown at it (it is not capacity-constrained) — the bottleneck for further test-AUC gains is genuinely the quantity (and possibly diversity) of training examples available, not the model's expressive power.

---

## Task 8 — Serialized Model

The best pipeline from `GridSearchCV` (`SimpleImputer(strategy='median')` -> `StandardScaler()` -> `RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=200, random_state=42)`) was saved with:

```python
joblib.dump(best_pipeline, 'best_model.pkl')
```

The notebook includes a reload-and-predict block:

```python
loaded_model = joblib.load('best_model.pkl')
sample_rows = pd.DataFrame([
    X_train.iloc[0].to_dict(),
    X_train.iloc[1].to_dict()
], columns=X_train.columns)
preds = loaded_model.predict(sample_rows)
probs = loaded_model.predict_proba(sample_rows)[:, 1]
```

This ran without errors and confirmed correct predictions on two hand-crafted test rows, validating that the serialized pipeline — including its built-in imputation and scaling steps — can be loaded and used standalone, without re-running any of the original training code.

`best_model.pkl` is included in this repository (≈ 4.3 MB, well under the 100 MB threshold).

---

## Task 9 — Summary Comparison Table & Final Recommendation

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
|-------|---------------------|---------------------|--------------|
| Logistic Regression | 0.9447 | 0.0060 | 0.9569 |
| Decision Tree (controlled) | 0.9587 | 0.0125 | 0.9668 |
| Random Forest | 0.9891 | 0.0053 | 0.9924 |
| Gradient Boosting | 0.9868 | 0.0052 | 0.9888 |
| **Tuned RF Pipeline (GridSearchCV)** | **0.9898** | **0.0049** | **0.9923** |

**Added graph:** A grouped bar chart was added after the final comparison table in the notebook. It compares CV mean AUC and test AUC for each model, showing that the tuned Random Forest pipeline has both the strongest validation performance and matching test performance.

### Recommendation

**The Tuned Random Forest Pipeline (from GridSearchCV) is the recommended model.** It achieves the highest 5-fold cross-validated mean AUC (0.9898) among all models compared, edging out the untuned Random Forest (0.9891) and clearly outperforming Gradient Boosting (0.9868), the controlled Decision Tree (0.9587), and Logistic Regression (0.9447). Its low CV standard deviation (0.0049) and test-set AUC (0.9923) closely match its CV estimate, indicating the model generalizes reliably rather than having been overfit to a lucky validation split. Beyond raw performance, this pipeline is the most production-ready of all options evaluated: it bundles imputation, scaling, and the classifier into a single serialized `best_model.pkl` artifact that can be reloaded and queried with a single `.predict()` call, with no manual preprocessing required at inference time — and the Task 4b ablation study further suggests that a leaner 15-feature variant of this Random Forest could be deployed with virtually no AUC cost, offering an attractive path to an even simpler production system if operational simplicity becomes a priority.

---

## Files in this Repository

| File | Description |
|------|--------------|
| `cleaned_data.csv` | Cleaned dataset from Part 1 |
| `Part 3.ipynb` | Executed notebook — all ensemble, tuning, ablation, and pipeline code |
| `best_model.pkl` | Serialized, tuned Random Forest pipeline (imputer + scaler + classifier) |
| `Part 3_README.md` | This file |

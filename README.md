# Life Expectancy Machine Learning Capstone

## Project Description

This project builds an end-to-end machine learning workflow on the Life Expectancy (WHO) Dataset. It starts with data cleaning and exploratory analysis, then trains regression and classification models, compares ensemble methods, saves the best production-ready model, and finally adds an LLM-powered explanation layer for model predictions.

## Dataset Overview

The project uses the Life Expectancy (WHO) Dataset from Kaggle. It contains 2,938 country-year records with demographic, healthcare, economic, and social indicators such as Adult Mortality, GDP, Schooling, BMI, Alcohol consumption, Population, HIV/AIDS, Country, Status, and Life expectancy. The continuous `Life expectancy` target is used for regression and is also converted into an above-median vs below-median target for classification.

## Technologies Used

- Python 3.9+
- Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn
- joblib
- python-dotenv
- requests
- jsonschema
- OpenRouter / OpenAI-compatible chat completion API

## Repository Structure

```text
Masai Capstone Project/
|-- README.md
|-- Part 1/
|   |-- Life_Expectancy_Data.csv
|   |-- cleaned_data.csv
|   |-- Part 1.ipynb
|   `-- Part 1_README.md
|-- Part 2/
|   |-- cleaned_data.csv
|   |-- Part 2.ipynb
|   `-- Part 2_README.md
|-- Part 3/
|   |-- cleaned_data.csv
|   |-- best_model.pkl
|   |-- Part 3.ipynb
|   `-- Part 3_README.md
`-- Part 4/
    |-- .env.example
    |-- .gitignore
    |-- best_model.pkl
    |-- Part 4.ipynb
    `-- Part 4_README.md
```

## Project Parts

**Part 1 - Data Preprocessing & EDA**  
Loads the dataset, analyzes null values, removes duplicates, corrects data types, studies skewness and outliers, creates required visualizations, compares Pearson and Spearman correlations, and saves `cleaned_data.csv`.

**Part 2 - Regression & Classification**  
Builds supervised learning baselines using linear regression, Ridge regression, and logistic regression. It includes train-test splitting, scaling, classification metrics, ROC-AUC, threshold analysis, regularization comparison, and bootstrap confidence intervals.

**Part 3 - Ensemble Models, Tuning & Deployment**  
Trains decision trees, Random Forest, and Gradient Boosting models, evaluates feature importance, performs feature-removal analysis, runs 5-fold cross-validation, tunes a Random Forest pipeline with GridSearchCV, and saves the best model as `best_model.pkl`.

**Part 4 - LLM-Powered Prediction Explanations**  
Loads the saved model, predicts on three hand-crafted feature vectors, applies PII guardrails, builds structured prompts, validates JSON explanations, and uses an OpenRouter-compatible LLM interface with a mock fallback for reproducible execution.

## How to Run

1. Open the project folder in Jupyter, VS Code, or another notebook environment.
2. Run the notebooks in order:
   - `Part 1/Part 1.ipynb`
   - `Part 2/Part 2.ipynb`
   - `Part 3/Part 3.ipynb`
   - `Part 4/Part 4.ipynb`
3. For Part 4 live LLM calls, copy `Part 4/.env.example` to `Part 4/.env`, add an OpenRouter API key, and set `FORCE_MOCK_MODE=false`. The notebook defaults to mock mode so it can run without API credits or network access.

## Results Summary

The best model is the tuned Random Forest pipeline from Part 3. It achieved a 5-fold cross-validated mean AUC of approximately `0.9898`, a CV standard deviation of approximately `0.0049`, and a test-set AUC of approximately `0.9923`. The final pipeline includes median imputation, scaling, and the Random Forest classifier, making it suitable for reuse through the serialized `best_model.pkl` artifact.

## License

This project is for educational use as part of a capstone assignment.

# Email Spam Detection - Part 1: Data Acquisition, Cleaning and Exploratory Data Analysis

## Project Overview

This project focuses on preparing an Email Spam Detection dataset for machine learning. The raw dataset was inspected, cleaned, analyzed, and visualized before any predictive modeling.

---

# Dataset Description

The dataset contains information about emails and includes the following features:

* email_id
* subject
* email_length
* num_links
* num_exclamations
* capital_letter_percent
* has_offer
* sender_domain
* spam_score
* label (Spam / Not Spam)

The objective is to clean the dataset and understand its characteristics before using it for machine learning.

---

# Data Loading

The dataset was loaded using the Pandas library.

```python
pd.read_csv()
```

The first five rows, data types, and dataset shape were displayed successfully.

---

# Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

and their percentages were calculated.

Numeric columns with less than 20% missing values were imputed using the **median**.

### Why Median?

The median is less sensitive to outliers than the mean. Since several numeric columns are positively skewed, the median provides a better representation of the center of the data and avoids distortion caused by extreme values.

---

# Duplicate Removal

Duplicate rows were detected using:

```python
df.duplicated().sum()
```

A total of **15 duplicate rows** were removed.

After duplicate removal, missing value percentages were checked again.

---

# Data Type Conversion

The following columns were converted to the **category** data type:

* has_offer
* sender_domain
* label

This reduced memory usage and improved efficiency when handling categorical variables.

---

# Descriptive Statistics

Descriptive statistics were generated using:

```python
df.describe()
```

The dataset's numeric columns were summarized using count, mean, standard deviation, minimum, maximum, and quartiles.

---

# Skewness Analysis

Skewness was calculated for every numeric column.

The most skewed column was:

**num_exclamations**

Skewness ≈ **1.119**

This is a **positive (right) skew**, indicating that most emails contain relatively few exclamation marks, while a small number contain many, creating a long right tail.

Because positive skew pulls the mean upward, the **median** was selected for missing value imputation.

---

# Outlier Detection (IQR Method)

The IQR method was applied to two numeric columns.

## num_exclamations

* Q1 = 1
* Q3 = 5
* IQR = 4
* Outliers = 0

No outliers were detected.

## capital_letter_percent

* Q1 = 12
* Q3 = 40
* IQR = 28
* Outliers = 15

These outliers were retained because they may represent genuine spam emails that use excessive capital letters.

Outliers will be handled during model building if required rather than being removed immediately.

---

# Visualizations

The following visualizations were created:

### Line Plot

A line plot of Spam Score across the dataset showed how spam scores vary between emails.

### Bar Chart

The bar chart compared the average spam score for each email label.

### Histogram

The histogram of **num_exclamations** showed a positively skewed distribution.

### Scatter Plot

The scatter plot between **spam_score** and **capital_letter_percent** suggested a positive relationship, where emails with higher spam scores generally contained a higher percentage of capital letters.

### Box Plot

The box plot compared spam scores across email labels and highlighted differences in median values and variability.

---

# Correlation Analysis

Pearson correlation was computed for all numeric variables and visualized using a heatmap.

The strongest correlated variable pair was identified from the correlation matrix.

Although a strong correlation exists, correlation alone does not imply causation. A third variable such as email content characteristics or promotional language may influence both variables simultaneously.

---

# Mean vs Median Comparison

The two most skewed numeric columns were analyzed by comparing their mean and median values.

Median was selected because skewed distributions make the mean less representative of the true center.

After imputation, all remaining missing values were removed successfully.

---

# Spearman Correlation

Spearman rank correlation was computed and compared with Pearson correlation.

The three variable pairs with the greatest absolute difference between Spearman and Pearson correlations were identified.

For feature selection in future modeling, Spearman correlation will be preferred whenever variables exhibit monotonic but non-linear relationships. Pearson correlation remains useful for identifying approximately linear relationships.

---

# Grouped Aggregation

Grouped aggregation was performed using:

* Categorical Feature: label
* Numeric Feature: spam_score

The following statistics were calculated:

* Mean
* Standard Deviation
* Count

The group with the highest mean spam score and the group with the highest variability were identified.

The ratio between the highest and lowest group means was calculated to evaluate whether the categorical feature carries predictive information.

---

# Clean Dataset

The cleaned dataset was saved as:


# Part 2 – Supervised Machine Learning Model

## Dataset

The cleaned dataset (`cleaned_data.csv`) generated in Part 1 was used for building regression and classification models.

## Target Variables

### Regression Target

* **Target:** `spam_score`
* This is a continuous numerical variable used for predicting the spam score.

### Classification Target

* **Target:** `label`
* Encoded as:

  * Ham = 0
  * Spam = 1

## Feature Encoding

Two categorical columns (`subject` and `sender_domain`) had no natural ordering. Therefore, one-hot encoding was applied with `drop_first=True` to prevent multicollinearity. One-hot encoding avoids introducing a false ordinal relationship that label encoding would create.

## Train-Test Split and Scaling

The dataset was split into:

* Training Set: 80%
* Testing Set: 20%

A `StandardScaler` was fitted **only on the training data** and then used to transform both training and testing datasets. Fitting the scaler on the complete dataset would cause **data leakage**, because information from the test set would influence the training process.

---

# Linear Regression

### Results

* Mean Squared Error (MSE): **54.34**
* R² Score: **0.9457**

### Most Important Features

1. num_links
2. subject_Click to verify
3. subject_Win cash now

Large positive coefficients indicate that increasing the corresponding standardized feature increases the predicted spam score. Large negative coefficients would indicate that increasing the feature decreases the predicted spam score.

---

# Ridge Regression

### Results

* Mean Squared Error (MSE): **54.27**
* R² Score: **0.9457**

### Comparison

| Model             | MSE   | R² Score |
| ----------------- | ----- | -------- |
| Linear Regression | 54.34 | 0.9457   |
| Ridge Regression  | 54.27 | 0.9457   |

Ridge Regression applies L2 regularization, shrinking large coefficients and reducing overfitting. The `alpha` parameter controls the strength of regularization. Higher alpha values produce stronger coefficient shrinkage.

---

# Logistic Regression

The dataset contained:

* Ham = 367 samples
* Spam = 233 samples

Since both classes exceeded 35% of the dataset, additional imbalance handling was not required.

### Results

* Accuracy = **1.00**
* Precision = **1.00**
* Recall = **1.00**
* F1 Score = **1.00**
* AUC = **1.00**

### Confusion Matrix

| Actual | Predicted Ham | Predicted Spam |
| ------ | ------------- | -------------- |
| Ham    | 66            | 0              |
| Spam   | 0             | 54             |

---

# ROC Curve

The ROC curve produced an Area Under the Curve (AUC) of **1.00**, indicating perfect separation between spam and ham emails.

---

# Precision and Recall

Precision Formula:

**Precision = TP / (TP + FP)**

Recall Formula:

**Recall = TP / (TP + FN)**

For spam email detection, Recall is particularly important because missing a spam email (False Negative) can reduce the effectiveness of spam filtering. However, this dataset achieved perfect Precision and Recall.

---

# Decision Threshold Analysis

Thresholds tested:

* 0.30
* 0.40
* 0.50
* 0.60
* 0.70

All thresholds produced identical performance:

* Precision = 1.00
* Recall = 1.00
* F1 Score = 1.00

Therefore, the default threshold of **0.50** was retained.

---

# Logistic Regression Regularization

Two Logistic Regression models were compared.

| Model    | Precision | Recall | AUC  |
| -------- | --------- | ------ | ---- |
| C = 1.0  | 1.00      | 1.00   | 1.00 |
| C = 0.01 | 1.00      | 1.00   | 1.00 |

The parameter **C** controls the inverse strength of regularization. Smaller values of C apply stronger L2 regularization. In this dataset, reducing C from 1.0 to 0.01 did not change model performance.

---

# Bootstrap Confidence Interval

500 bootstrap samples were generated.

Results:

* Mean AUC Difference = **-6.66 × 10⁻¹⁹**
* Lower 95% CI = **-1.11 × 10⁻¹⁶**
* Upper 95% CI = **1.11 × 10⁻¹⁶**

Since the confidence interval includes zero, there is no statistically reliable evidence that one Logistic Regression model consistently outperforms the other.

---

# Conclusion

Both the regression and classification models performed exceptionally well on the cleaned dataset. Linear Regression and Ridge Regression produced very similar predictive performance, while Logistic Regression achieved perfect classification accuracy with an AUC of 1.00. The bootstrap analysis further confirmed that stronger regularization did not significantly change model performance on this dataset.


```text
cleaned_data.csv
```

This file will be used for Part 2 and Part 3 of the project.

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


# Part 3 – Advanced Modeling, Hyperparameter Tuning and Pipeline

## Decision Tree Baseline

A Decision Tree Classifier with default parameters was trained on the scaled training data.

### Results

* Training Accuracy: **1.00**
* Testing Accuracy: **1.00**

The baseline Decision Tree achieved identical training and testing accuracy, indicating no evidence of overfitting on this dataset. In general, decision trees are considered high-variance models because they greedily fit the training data at each split and can easily memorize noise in more complex datasets.

---

## Controlled Decision Tree

A second Decision Tree was trained with:

* max_depth = 5
* min_samples_split = 20

### Results

* Training Accuracy: **1.00**
* Testing Accuracy: **1.00**

The controlled tree achieved the same performance as the unconstrained tree. The **max_depth** parameter limits tree growth, reducing variance and helping prevent overfitting. The **min_samples_split** parameter prevents splitting nodes containing very few samples, reducing sensitivity to noise.

---

## Gini vs Entropy

Two Decision Tree models were compared using different splitting criteria.

### Results

| Criterion | Test Accuracy |
| --------- | ------------- |
| Gini      | 1.00          |
| Entropy   | 1.00          |

### Formulae

**Gini Impurity**

Gini = 1 − Σ(pi²)

**Entropy**

Entropy = −Σ(pi log₂(pi))

A Gini impurity value of **0** indicates that all samples in the node belong to the same class, meaning the node is perfectly pure.

---

## Random Forest

A Random Forest classifier was trained with:

* n_estimators = 100
* max_depth = 10
* random_state = 42

### Results

* Training Accuracy: **1.00**
* Testing Accuracy: **1.00**
* ROC-AUC: **1.00**

### Top 5 Important Features

| Feature                | Importance |
| ---------------------- | ---------: |
| num_exclamations       |   0.285543 |
| num_links              |   0.272256 |
| capital_letter_percent |   0.174283 |
| has_offer              |   0.104605 |
| email_length           |   0.066764 |

Random Forest feature importance represents the average reduction in Gini impurity contributed by each feature across all trees. Unlike Linear Regression coefficients, these values indicate feature usefulness rather than the direction of the relationship.

### Bagging

Random Forest uses bootstrap sampling, where each tree is trained on a random sample drawn with replacement from the training data. At every split, only a random subset of features is considered. Combining predictions from many trees reduces variance and generally improves generalization compared with a single deep decision tree.

---

## Gradient Boosting

Gradient Boosting Classifier parameters:

* n_estimators = 100
* learning_rate = 0.1
* max_depth = 3

### Results

* Training Accuracy: **1.00**
* Testing Accuracy: **1.00**
* ROC-AUC: **1.00**

Gradient Boosting also achieved excellent performance, producing nearly perfect classification.

---

## Feature Ablation Study

The five least important features identified by the Random Forest were:

* sender_domain_gmail.com
* sender_domain_offers.org
* sender_domain_outlook.com
* sender_domain_promo.net
* sender_domain_yahoo.com

### Results

| Model         | ROC-AUC  |
| ------------- | -------- |
| Full Model    | **1.00** |
| Reduced Model | **1.00** |

Removing the five least important features did not reduce the ROC-AUC score. This suggests that these features contributed little predictive information and can be removed to simplify the model without affecting performance.

---

## Cross-Validation Comparison

Five-fold Stratified Cross Validation was used with ROC-AUC scoring.

| Model               | Mean AUC | Std AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   1.0000 |  0.0000 |
| Decision Tree       |   0.9936 |  0.0085 |
| Random Forest       |   1.0000 |  0.0000 |
| Gradient Boosting   |   0.9978 |  0.0043 |

Cross-validation provides a more reliable estimate of model performance because it evaluates the model on multiple train-test splits rather than relying on a single split.

---

## Hyperparameter Tuning using GridSearchCV

The following parameter grid was evaluated:

* n_estimators = [50, 100, 200]
* max_depth = [5, 10, None]
* min_samples_leaf = [1, 5]

Total parameter combinations:

18

Using 5-fold cross-validation:

Total model fits = **90**

### Best Parameters

* max_depth = 5
* min_samples_leaf = 1
* n_estimators = 50

### Best Cross-Validation ROC-AUC

**1.00**

Grid Search evaluates every possible parameter combination to identify the best-performing model. Although computationally expensive, it guarantees that the optimal combination within the search space is found.

---

## Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
| ----------------: | -----------: | -------: |
|               20% |         1.00 |     1.00 |
|               40% |         1.00 |     1.00 |
|               60% |         1.00 |     1.00 |
|               80% |         1.00 |     1.00 |
|              100% |         1.00 |     1.00 |

The training and testing AUC remained constant as the amount of training data increased. This indicates that the dataset is easily separable and that additional training data is unlikely to improve performance significantly.

---

## Model Serialization

The best-performing pipeline was saved using:

`joblib.dump(best_pipeline, "best_model.pkl")`

The saved model was successfully reloaded using `joblib.load()` and used to predict two sample records without errors.

Predictions:

```
[1 1]
```

---

## Final Model Recommendation

Among all evaluated models, **Random Forest** is recommended for deployment. It achieved a perfect ROC-AUC of **1.00**, demonstrated excellent stability during five-fold cross-validation, and remained robust even after removing low-importance features. The tuned pipeline can be easily reused through the saved `best_model.pkl` file, making deployment straightforward and reproducible.

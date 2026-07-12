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

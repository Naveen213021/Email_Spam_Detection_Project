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

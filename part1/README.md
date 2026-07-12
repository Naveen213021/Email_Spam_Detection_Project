
Part 1: Data Acquisition, Cleaning and Exploratory Data Analysis
Project Overview
This project focuses on preparing an Email Spam Detection dataset for machine learning. The raw dataset was inspected, cleaned, analyzed, and visualized before any predictive modeling.

Dataset Description
The dataset contains information about emails and includes the following features:

email_id
subject
email_length
num_links
num_exclamations
capital_letter_percent
has_offer
sender_domain
spam_score
label (Spam / Not Spam)
The objective is to clean the dataset and understand its characteristics before using it for machine learning.

Data Loading
The dataset was loaded using the Pandas library.

pd.read_csv()
The first five rows, data types, and dataset shape were displayed successfully.

Missing Value Analysis
Missing values were identified using:

df.isnull().sum()
and their percentages were calculated.

Numeric columns with less than 20% missing values were imputed using the median.

Why Median?
The median is less sensitive to outliers than the mean. Since several numeric columns are positively skewed, the median provides a better representation of the center of the data and avoids distortion caused by extreme values.

Duplicate Removal
Duplicate rows were detected using:

df.duplicated().sum()
A total of 15 duplicate rows were removed.

After duplicate removal, missing value percentages were checked again.

Data Type Conversion
The following columns were converted to the category data type:

has_offer
sender_domain
label
This reduced memory usage and improved efficiency when handling categorical variables.

Descriptive Statistics
Descriptive statistics were generated using:

df.describe()
The dataset's numeric columns were summarized using count, mean, standard deviation, minimum, maximum, and quartiles.

Skewness Analysis
Skewness was calculated for every numeric column.

The most skewed column was:

num_exclamations

Skewness ≈ 1.119

This is a positive (right) skew, indicating that most emails contain relatively few exclamation marks, while a small number contain many, creating a long right tail.

Because positive skew pulls the mean upward, the median was selected for missing value imputation.

Outlier Detection (IQR Method)
The IQR method was applied to two numeric columns.

num_exclamations
Q1 = 1
Q3 = 5
IQR = 4
Outliers = 0
No outliers were detected.

capital_letter_percent
Q1 = 12
Q3 = 40
IQR = 28
Outliers = 15
These outliers were retained because they may represent genuine spam emails that use excessive capital letters.

Outliers will be handled during model building if required rather than being removed immediately.

Visualizations
The following visualizations were created:

Line Plot
A line plot of Spam Score across the dataset showed how spam scores vary between emails.

Bar Chart
The bar chart compared the average spam score for each email label.

Histogram
The histogram of num_exclamations showed a positively skewed distribution.

Scatter Plot
The scatter plot between spam_score and capital_letter_percent suggested a positive relationship, where emails with higher spam scores generally contained a higher percentage of capital letters.

Box Plot
The box plot compared spam scores across email labels and highlighted differences in median values and variability.

Correlation Analysis
Pearson correlation was computed for all numeric variables and visualized using a heatmap.

The strongest correlated variable pair was identified from the correlation matrix.

Although a strong correlation exists, correlation alone does not imply causation. A third variable such as email content characteristics or promotional language may influence both variables simultaneously.

Mean vs Median Comparison
The two most skewed numeric columns were analyzed by comparing their mean and median values.

Median was selected because skewed distributions make the mean less representative of the true center.

After imputation, all remaining missing values were removed successfully.

Spearman Correlation
Spearman rank correlation was computed and compared with Pearson correlation.

The three variable pairs with the greatest absolute difference between Spearman and Pearson correlations were identified.

For feature selection in future modeling, Spearman correlation will be preferred whenever variables exhibit monotonic but non-linear relationships. Pearson correlation remains useful for identifying approximately linear relationships.

Grouped Aggregation
Grouped aggregation was performed using:

Categorical Feature: label
Numeric Feature: spam_score
The following statistics were calculated:

Mean
Standard Deviation
Count
The group with the highest mean spam score and the group with the highest variability were identified.

The ratio between the highest and lowest group means was calculated to evaluate whether the categorical feature carries predictive information.

Clean Dataset
The cleaned dataset was saved as:

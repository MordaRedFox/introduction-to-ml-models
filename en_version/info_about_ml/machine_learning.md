# General Information about Machine Learning

## Table of Contents
1. [What is Machine Learning](#what-is-machine-learning)
2. [What Problems Does Machine Learning Solve](#what-problems-does-machine-learning-solve)
3. [Exploratory Factor Analysis](#exploratory-factor-analysis)
4. [Factor Preparation](#factor-preparation)
5. [Model Selection and Creation](#model-selection-and-creation)
6. [Model Hyperparameter Tuning](#model-hyperparameter-tuning)
7. [Model Quality Assessment](#model-quality-assessment)
8. [Conclusion](#conclusion)

---

## What is Machine Learning

### Core Definition
Machine Learning is a branch of artificial intelligence that studies methods for building algorithms capable of learning from data and making predictions or decisions without explicit programming.

### How It Works
Instead of writing strict rules (as in conventional programming), we "show" the algorithm many examples, and it finds patterns in the data by itself.

### Types of Machine Learning
- **Supervised Learning**: labeled data with correct answers (labels) is available.
- **Unsupervised Learning**: finding structure in data without labels.
- **Reinforcement Learning**: the algorithm learns from its own actions and their consequences via a reward function.

---

## What Problems Does Machine Learning Solve

### Binary Classification
Assigning objects to two specific categories. For example:
- Cats/dogs in a photo
- Good customer/bad customer

### Multiclass Classification
Assigning objects to an arbitrary number of specific categories. For example:
- Identifying a number in an image
- Determining the type of rose from a photo

### Regression
Predicting numerical values. For example:
- Apartment price
- Tomorrow's air temperature
- Number of store sales

### Clustering
Grouping similar objects without predefined labels. For example:
- Customer segmentation
- Grouping documents by topic

### Other Tasks
- **Anomaly Detection**: finding unusual objects
- **Dimensionality Reduction**: simplifying data without losing important information
- **Recommendation Systems**: suggesting products/movies to a user

---

## Exploratory Factor Analysis

### Why It's Needed
Before building a model, it's necessary to understand the data you'll be working with. This helps:
- Find errors in the data
- Detect outliers in the data
- Identify missing values in the data
- Observe patterns in the data
- Choose the correct data processing methods

### Main Analysis Methods

#### Descriptive Statistics
To start working, it is advisable to conduct a primary data analysis, including calculating basic statistical indicators.

Example code:

```python
import pandas as pd

data = pd.read_csv('data.csv')
print(data.describe())  # basic statistics
print(data.info())      # information about data types
print(data.head())      # first rows of data
```

#### Visualization
- **Histograms**: distribution of numerical features
- **Scatter Plots**: relationship between two features
- **Box Plots**: detecting outliers
- **Heatmaps**: correlations between features

#### Correlation Analysis
Searching for features related to each other and to the target variable.

Example code:

```python
correlation_matrix = data.corr()
```

---

## Factor Preparation

### Handling Missing Values
- **Deletion**: removing data with missing values if there are few of them
- **Imputation**: replacing missing values, e.g., with the median value
- **Prediction**: predicting missing values using other models

### Encoding Categorical Features
- **One-Hot Encoding**: splitting a category into binary factors
- **Label Encoding**: assigning numbers to categories
- **Target Encoding**: replacing a category with the mean value of the target variable

### Feature Scaling
For the correct operation of many machine learning algorithms, feature scaling is required. This ensures model convergence by equalizing the influence of each feature on the weight update process.

Example code:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
```

### Handling Outliers
**Outliers** are observations in the data that significantly differ from other values and can be caused by measurement errors, rare events, or natural data variability. Outliers can severely distort analysis results and degrade the quality of machine learning models.

Methods for handling outliers:
- **Deletion**: removing clearly erroneous outliers
- **Capping**: replacing outliers with boundary values
- **Transformation**: applying logarithms to reduce the influence of outliers

---

## Model Selection and Creation

### How to Choose a Model
Model selection depends on the task at hand and the available data:

#### Key Model Selection Criteria:
- **Data Size**: small/large dataset
- **Interpretability**: whether predictions need to be explained
- **Training Speed**: how fast the model should train
- **Task Type**: classification, regression, clustering
- **Data Linearity**: linear or complex nonlinear dependencies

#### For Classification
- **Logistic Regression**: simple and interpretable method, works well when classes are linearly separable
- **Decision Trees**: easy to understand but prone to overfitting, good for nonlinear dependencies
- **Random Forest**: more robust to overfitting but harder to interpret, good for most tasks
- **Support Vector Machines (SVM)**: good for complex boundaries but slow on large data

#### For Regression
- **Linear Regression**: basic method, good when the relationship is linear
- **Regression Trees**: nonlinear dependencies but prone to overfitting
- **Random Forest for Regression**: ensemble method, robust to noise
- **Ridge/Lasso Regression**: with regularization, when there are many features

### Model Creation Process
Example code:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Model creation and training
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Model prediction
y_pred = model.predict(x_test)
```

---

## Model Hyperparameter Tuning

### What are Hyperparameters
These are global parameters (model settings) that are not learned from data but are set in advance. For example:
- Tree depth
- Model learning rate
- Number of trees in a forest

### Hyperparameter Tuning Methods

#### Grid Search
Iterates over all combinations from a given set of hyperparameters.

Example code:

```python
from sklearn.model_selection import GridSearchCV

parameters = {
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(model, parameters, cv=5)
grid_search.fit(X_train, y_train)
```

#### Random Search
Randomly selects combinations from given distributions.

Example code:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 200),           # random integer from 50 to 200
    'max_depth': randint(3, 15),                # random integer from 3 to 15
    'min_samples_split': randint(2, 20),        # random integer from 2 to 20
    'min_samples_leaf': randint(1, 10),         # random integer from 1 to 10
    'max_features': ['sqrt', 'log2', None],     # categorical parameter
    'bootstrap': [True, False]                  # boolean parameter
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,                                  # number of random combinations
    cv=5,                                       # number of cross-validation folds
    random_state=52,                            # for reproducibility
    n_jobs=-1                                   # use all processors
)

random_search.fit(x_train, y_train)

# Best parameters and score
print('Best parameters:', random_search.best_params_)
print('Best score:', random_search.best_score_)
```

#### Cross-Validation
Splits the data into several parts and alternately uses them for training and validating the model.

Example code:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, x_train, y_train, cv=5)
print(f'Average accuracy: {scores.mean()}')
```

---

## Model Quality Assessment

### For Classification

#### Confusion Matrix
Shows how many objects of each class are correctly classified:

<div align="center">

<table>
    <tr align="center">
        <th>Metric</th>
        <th>Formula</th>
        <th>Interpretation</th>
    </tr>
    <tr align="center">
        <td><strong>Accuracy</strong></td>
        <td><code>(TP + TN) / (TP + TN + FP + FN)</code></td>
        <td>Overall proportion of correct predictions</td>
    </tr>
    <tr align="center">
        <td><strong>Precision</strong></td>
        <td><code>TP / (TP + FP)</code></td>
        <td>Accuracy of positive predictions</td>
    </tr>
    <tr align="center">
        <td><strong>Recall</strong></td>
        <td><code>TP / (TP + FN)</code></td>
        <td>Completeness of positive predictions</td>
    </tr>
    <tr align="center">
        <td><strong>F1-score</strong></td>
        <td><code>2 * (Precision * Recall) / (Precision + Recall)</code></td> <td>Harmonic mean of precision and recall</td>
    </tr>
</table>

</div>

#### Key Metrics
- **Accuracy**: overall correctness of the model's predictions
- **Precision**: accuracy of the model's positive predictions
- **Recall**: completeness of detecting positive classes by the model
- **F1-score**: balance between precision and recall
- **ROC-AUC**: area under the ROC curve

### For Regression
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

### Bootstrap and Confidence Intervals
Using bootstrap, one can assess the reliability of quality metrics by constructing their confidence intervals.

Example code (`decision_tree/model.py`):

```python
# Combining test data
x_y_test = x_test.copy(deep=True)
x_y_test['label'] = y_test

boot_accuracies = []
boot_precisions = []
boot_recalls = []
boot_f1_scores = []

n_bootstraps = 1000

print(f'\n\nRunning bootstrap ({n_bootstraps} iterations)...')
for i in range(n_bootstraps):
    if (i + 1) % 100 == 0:
        print(f'Completed iterations: {i + 1}/{n_bootstraps}')

    # Creating bootstrap sample
    x_y_test_boot = x_y_test.sample(len(x_y_test), replace=True)
    x_test_boot = x_y_test_boot.drop(columns='label')
    y_test_boot = x_y_test_boot['label']

    # Model predictions
    y_pred = best_model.predict(x_test_boot)

    # Calculating quality metrics
    boot_accuracies.append(accuracy_score(y_test_boot, y_pred))
    boot_precisions.append(precision_score(y_test_boot, y_pred,
                                         average='weighted', zero_division=0))
    boot_recalls.append(recall_score(y_test_boot, y_pred,
                                   average='weighted', zero_division=0))
    boot_f1_scores.append(f1_score(y_test_boot, y_pred,
                                 average='weighted', zero_division=0))

def calculate_confidence_interval(metric_values):
    """Calculates confidence intervals (95%)"""
    sorted_metrics = np.sort(metric_values)
    lower_bound = sorted_metrics[int(0.025 * len(sorted_metrics))]
    upper_bound = sorted_metrics[int(0.975 * len(sorted_metrics))]
    return lower_bound, upper_bound

accuracy_ci = calculate_confidence_interval(boot_accuracies)
precision_ci = calculate_confidence_interval(boot_precisions)
recall_ci = calculate_confidence_interval(boot_recalls)
f1_ci = calculate_confidence_interval(boot_f1_scores)

# Output confidence intervals
print('\n\nMetric confidence intervals (bootstrap):')
print('Accuracy:')
print(f'Interval mean value: {best_test_accuracy}')
print(f'Interval: [{accuracy_ci[0]}, {accuracy_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    accuracy_ci[1] - accuracy_ci[0]})')

print('\nPrecision:')
print(f'Interval mean value: {best_test_precision}')
print(f'Interval: [{precision_ci[0]}, {precision_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    precision_ci[1] - precision_ci[0]})')

print('\nRecall:')
print(f'Interval mean value: {best_test_recall}')
print(f'Interval: [{recall_ci[0]}, {recall_ci[1]}]')
print(f'Interval description: (95% CI, width: {recall_ci[1] - recall_ci[0]})')

print('\nF1:')
print(f'Interval mean value: {best_test_f1}')
print(f'Interval: [{f1_ci[0]}, {f1_ci[1]}]')
print(f'Interval description: (95% CI, width: {f1_ci[1] - f1_ci[0]})')
```

---

## Conclusion

Machine Learning is a powerful tool for solving complex problems where traditional programming fails. The main thing is to understand the task facing the developer, thoroughly prepare the data, choose a suitable model, and correctly assess its quality.

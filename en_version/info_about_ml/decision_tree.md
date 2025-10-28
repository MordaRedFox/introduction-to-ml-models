# Information on Decision Trees and Random Forests

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Multiclass Classification](#multiclass-classification)
3. [Tree Construction Algorithm](#tree-construction-algorithm)
4. [Splitting Criteria](#splitting-criteria)
5. [Combating Overfitting](#combating-overfitting)
6. [Random Forest](#random-forest)
7. [Practical Aspects](#practical-aspects)
8. [Comparison with Linear Models](#comparison-with-linear-models)
9. [Quality Metrics](#quality-metrics)
10. [Bootstrap and Confidence Intervals](#bootstrap-and-confidence-intervals)
11. [Implementation in a Project](#implementation-in-a-project)
12. [Conclusion](#conclusion)

---

## Basic Concepts

### What is a Decision Tree?
A decision tree is a machine learning algorithm that builds a tree-like structure of decisions for classification or regression. It is one of the most intuitive algorithms because it mimics the human decision-making process.

### Geometric Interpretation
Each question of the form "feature value **`xᵢ ≥ c`**" corresponds to a half-space. Paths from the root to a leaf correspond to regions of space bounded by hyperplanes. Essentially, a decision tree partitions the feature space into rectangular regions, each corresponding to a specific class or value.

### Advantages of Decision Trees
- **Interpretability**: Easy to visualize and understand the prediction logic
- **Non-linearity**: Can model complex non-linear dependencies
- **Versatility**: Works with numerical and categorical features
- **No scaling required**: Not sensitive to feature scale

### Limitations
- **Prone to overfitting**: Especially with deep trees
- **Instability**: Small changes in data can significantly alter the tree structure
- **Local optimization**: Greedy algorithm does not guarantee global optimum

---

## Multiclass Classification

### One-vs-Rest Approach
- Build k binary classifiers
- The i-th classifier determines belonging to class i or the rest
- The class with the highest value of **`⟨wᵢ,x⟩`** is selected

### One-vs-One Approach
- For each pair of classes, build a binary classifier
- Select the class with the highest number of "votes"
- Ambiguity can arise with an equal number of votes

### Accuracy Metric
**`accuracy = number of correctly classified objects / total number of objects`**

A basic metric, but can be uninformative with imbalanced classes.

### Multiclass Precision and Recall
**Micro-averaging:**
- Considers the total number of **`TP, FP, FN`** for all classes
- **`precision = ΣTPᵢ / (ΣTPᵢ + ΣFPᵢ)`**
- **`recall = ΣTPᵢ / (ΣTPᵢ + ΣFNᵢ)`**
- Good when the overall balance between precision and recall is important

**Macro-averaging:**
- Averages metrics across all classes
- **`precision = (1/k) * Σprecisionᵢ`**
- **`recall = (1/k) * Σrecallᵢ`**
- Better when all classes are equally important

---

## Tree Construction Algorithm

### Recursive Splitting
The tree construction algorithm uses a greedy "divide and conquer" approach. At each step, the best split of the data based on one of the features is selected.

Example:

```python
def build_decision_tree(x, current_depth):
    # Basic cases - recursion stopping conditions
    if all_samples_same_class(x):
        return LeafNode(majority_class(x))

    if current_depth >= max_depth:
        return LeafNode(most_frequent_class(x))

    # Finding the optimal data split
    feature, threshold = find_best_split(x)
    left_subset, right_subset = split_dataset(x, feature, threshold)

    # Recursive construction of subtrees
    left_subtree = build_decision_tree(left_subset, current_depth + 1)
    right_subtree = build_decision_tree(right_subset, current_depth + 1)

    # Returns an internal node of the tree
    return InternalNode(feature, threshold, left_subtree, right_subtree)
```

### Stopping Conditions
- All objects in the node belong to the same class
- Maximum tree depth is reached
- Number of objects in the node is less than the minimum threshold

---

## Splitting Criteria

### Split Quality Function
Split quality is assessed by the reduction in "chaos" of the data:

**`Q(x,x_l,x_r) = H(x) - (|x_l|/|x|)H(x_l) - (|x_r|/|x|)H(x_r)`**, where **`H(x)`** is a measure of data chaos

The greater the reduction in chaos, the better the split.

### Gini Criterion (Classification)
A measure of data impurity:

**`H(x) = Σp_c(1 - p_c)`**, where **`p_c`** is the proportion of objects of class **`c`**

Gini impurity is minimal (equals 0) when all objects belong to the same class.

### Entropy (Classification)
An information-theoretic measure of uncertainty:

**`H(x) = -Σp_c·log₂(p_c)`**

Also reaches a minimum when the data is pure.

### MSE (Regression)
For regression tasks, the mean squared error is used:

**`H(x) = (1/|x|)Σ(yᵢ - ȳ)²`**

---

## Combating Overfitting

Decision trees are prone to overfitting, especially when they are deep. Here are the main methods to combat this problem:

### Limiting Tree Depth
Limits the maximum depth of the tree, preventing excessive complexity.

Example:

```python
model = DecisionTreeClassifier(max_depth=10)
```

### Minimum Samples in a Leaf
Guarantees that each leaf has enough data for reliable statistics.

Example:

```python
model = DecisionTreeClassifier(min_samples_leaf=5)
```

### Minimum Samples for Splitting
Requires a minimum number of objects in a node for further splitting.

Example:

```python
model = DecisionTreeClassifier(min_samples_split=20)
```

### Tree Pruning
- Post-processing of the built tree
- Removing branches that do not improve validation quality
- A more complex approach than preliminary constraints

---

## Random Forest

### Main Idea
Random Forest is an ensemble method that solves the overfitting problem of individual trees through:
- **Bootstrap aggregating (Bagging)**: each tree is trained on a random subsample of the data
- **Random feature selection**: at each split, only a random subset of features is considered

### Construction Algorithm
Example:

```python
def build_random_forest(x, y, n_estimators, min_samples_leaf, max_features):
    forest = []

    for i in range(n_estimators):
        # Creating a bootstrap sample
        X_bootstrap, y_bootstrap = create_bootstrap_sample(x, y)

        # Building a tree with constraints
        tree = build_decision_tree(
            X_bootstrap, 
            y_bootstrap,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
        forest.append(tree)

    return forest
```

### Advantages of Random Forest
- **Resistance to overfitting**: averaging predictions of many trees
- **High accuracy**: often outperforms individual trees
- **Feature importance estimation**: can determine the most significant features
- **Handling missing values**: more robust than a single tree

### Hyperparameters
- **n_estimators**: number of trees
- **max_depth**: maximum depth of trees
- **min_samples_split**: minimum number of samples required to split a node
- **min_samples_leaf**: minimum number of samples required in a leaf node
- **max_features**: number of features to consider for the best split

### Recommendations for Hyperparameter Selection
- **For regression**: **`max_features = m/3`** (where **`m`** is the total number of features)
- **For classification**: **`max_features = √m`**
- **n_estimators**: the more the better (up to a saturation point)

---

## Practical Aspects

### Working with Categorical Features
1. **One-Hot Encoding**: creating binary features for each category
2. **Target Encoding**: replacing categories with the mean of the target variable
3. **Direct handling in the tree**: splitting by categories

### Feature Importance
Random Forest allows estimating the importance of each feature based on how much it improves the split quality.

Example:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
importances = model.feature_importances_
```

### Out-of-Bag Estimation (OOB)
- Each tree is built on ~63% of the data
- The remaining 37% are used for validation
- Does not require a separate validation set

---

## Comparison with Linear Models

<div align="center">

<table>
    <tr align="center">
        <th>Characteristic</th>
        <th>Decision Tree</th>
        <th>Linear Models</th>
    </tr>
    <tr align="center">
        <td>Flexibility</td>
        <td>High, non-linear dependencies</td>
        <td>Limited, linear dependencies</td>
    </tr>
    <tr align="center">
        <td>Overfitting</td>
        <td>Prone to overfitting</td>
        <td>Less prone</td>
    </tr>
    <tr align="center">
        <td>Training Speed</td>
        <td>Slower</td>
        <td>Faster (gradient methods)</td>
    </tr>
    <tr align="center">
        <td>Interpretability</td>
        <td>High (visualization)</td>
        <td>High (coefficients)</td>
    </tr>
    <tr align="center">
        <td>Data Requirements</td>
        <td>No scaling required</td>
        <td>Requires scaling</td>
    </tr>
</table>

</div>

Trees are better suited for complex non-linear dependencies, while linear models are better when a linear relationship is expected or speed is important.

---

## Quality Metrics

### For Classification
- **Accuracy**: overall accuracy
- **Precision**: accuracy of positive predictions
- **Recall**: completeness of detecting positive classes
- **F1-score**: harmonic mean of precision and recall
- **ROC-AUC**: area under the ROC curve

### For Regression
- **MSE**: mean squared error
- **MAE**: mean absolute error
- **R²**: coefficient of determination

---

## Bootstrap and Confidence Intervals

### Bootstrap Method
A statistical method for estimating metric uncertainty through multiple data resampling.

### Bootstrap Implementation
Code example:

```python
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
```

### Calculating Confidence Intervals
Code example:

```python
def calculate_confidence_interval(metric_values):
    """Calculates confidence intervals (95%)"""
    sorted_metrics = np.sort(metric_values)
    lower_bound = sorted_metrics[int(0.025 * len(sorted_metrics))]
    upper_bound = sorted_metrics[int(0.975 * len(sorted_metrics))]
    return lower_bound, upper_bound
```

---

## Implementation in a Project

### Decision Tree (Scikit-learn)
Example:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Creating and training the model
model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=5,
    random_state=52
)
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
```

### Random Forest (Scikit-learn)
Example:

```python
from sklearn.ensemble import RandomForestClassifier

# Creating and training the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=52
)
model.fit(x_train, y_train)

# Feature importance
importances = model.feature_importances_
```

### Tree Visualization
Example:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, filled=True)
plt.show()
```

---

## Conclusion

Decision trees and random forests are powerful machine learning tools that are especially useful when model interpretability and handling non-linear dependencies are important. Random Forest solves the overfitting problem of individual trees and often shows better results in practice.

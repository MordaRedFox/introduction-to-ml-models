# Logistic Regression Information

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Mathematical Model](#mathematical-model)
3. [Model Training](#model-training)
4. [Classification and Threshold Value](#classification-and-threshold-value)
5. [Quality Metrics](#quality-metrics)
6. [Practical Aspects](#practical-aspects)
7. [Bootstrap and Confidence Intervals](#bootstrap-and-confidence-intervals)
8. [Interpreting Logistic Regression Model Coefficients](#interpreting-logistic-regression-model-coefficients)
9. [Conclusion](#conclusion)

---

## Basic Concepts

### What is Logistic Regression?
Logistic regression is a statistical method for modeling the probability of an object belonging to one of two classes.

### Method Features
- Solves the **binary classification** problem
- Predicts the **probability** of belonging to the positive class
- Uses the **sigmoid function** to transform a linear combination into a probability

### Geometric Interpretation
The model finds a separating hyperplane in the feature space. The distance from a point to this hyperplane determines the model's "confidence" in the prediction.

### Advantages of Logistic Regression
- **Interpretability**: coefficients show the direction and strength of feature influence
- **Speed**: fast training and prediction
- **Probabilities**: predicts probabilities, not just classes
- **Robustness**: works well with noisy data

### Limitations
- **Linearity**: assumes linear separability of classes
- **Preprocessing**: requires careful data preparation
- **Outliers**: sensitive to outliers in the data
- **Complex dependencies**: not suitable for complex nonlinear dependencies

---

## Mathematical Model

### Basic Formula
Probability of belonging to the positive class:

**`p̂(x) = σ(⟨w,x⟩) = 1 / (1 + e^(-⟨w,x⟩))`**

where:
- **`⟨w,x⟩ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ`** — linear combination
- **`σ(z)`** — sigmoid function
- **`w`** — model weight vector
- **`x`** — object feature vector

### Logit Function
Transformation of probability into odds ratio:

**`logit(p) = ln(p / (1-p)) = ⟨w,x⟩`**

### Probability Interpretation
- **`p̂(x) > 0.5`** — the object is more likely to belong to the positive class
- **`p̂(x) < 0.5`** — the object is more likely to belong to the negative class
- **`p̂(x) = 0.5`** — classification uncertainty

---

## Model Training

### Likelihood Function
Maximizing the data likelihood:

**`L(w) = ∏ p̂(yᵢ|xᵢ)`**

where:
**`p̂(yᵢ|xᵢ) = p̂(xᵢ)^( (1+yᵢ)/2 ) * (1-p̂(xᵢ))^( (1-yᵢ)/2 )`**

### Log-Likelihood
Minimizing the loss function:

**`loss(w) = -∑ ln(p̂(yᵢ|xᵢ)) = ∑ ln(1 + e^(-yᵢ⟨w,xᵢ⟩))`**

### Gradient Descent
Partial derivative with respect to parameter wⱼ:

**`∂loss/∂wⱼ = -∑ yᵢxᵢⱼσ(-yᵢ⟨w,xᵢ⟩)`**

where:
- **`xᵢⱼ`** — value of the j-th feature of the i-th object
- **`σ(z)`** — sigmoid function

---

## Classification and Threshold Value

### Threshold Rule
**`ŷ = sign(p̂(x) - t)`**

where:
- **`t`** — threshold value (default 0.5)
- **`ŷ`** — predicted class (1 or -1)

### Threshold Selection
- **By default**: t = 0.5
- **With class imbalance**: selection on the validation set
- **Criterion**: maximizing accuracy or F1-score

### Threshold Selection Strategy
Example of searching for the optimal threshold value:

```python
best_threshold = None
best_f1 = -float('inf')

val_probabilities = model.predict_proba(x_val)[:, 1]

for t in range(0, 1001):
    threshold = 0.001 * t
    y_val_pred = val_probabilities > threshold
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
```

---

## Quality Metrics

### Confusion Matrix

<div align="center">

<table>
    <tr align="center">
        <th rowspan="2">Actual \ Predicted</th>
        <th colspan="2">Prediction</th>
    </tr>
    <tr align="center">
        <th>Positive (1)</th>
        <th>Negative (0)</th>
    </tr>
    <tr align="center">
        <td><strong>Positive (1)</strong></td>
        <td>TP (True Positive)</td>
        <td>FN (False Negative)</td>
    </tr>
    <tr align="center">
        <td><strong>Negative (0)</strong></td>
        <td>FP (False Positive)</td>
        <td>TN (True Negative)</td>
    </tr>
</table>

</div>

### Main Metrics

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

### Additional Metrics
- **ROC-AUC**: area under the ROC curve
- **Precision-Recall AUC**: area under the Precision-Recall curve
- **Log Loss**: logarithmic loss function

### Metric Features
- **Accuracy**: performs poorly with class imbalance
- **Precision**: important when the cost of FP is high (e.g., spam filter)
- **Recall**: important when the cost of FN is high (e.g., medical diagnosis)
- **F1-score**: balances precision and recall

---

## Practical Aspects

### Data Preprocessing
- **Standardization**: mandatory for gradient descent convergence
- **Handling imbalance**: class weighting, oversampling/undersampling
- **Feature selection**: excluding multicollinear features

### Regularization
- **L1-regularization (Lasso)**: feature selection, zeroing coefficients
- **L2-regularization (Ridge)**: reduces overfitting
- **Elastic Net**: combination of L1 and L2 regularization

### Working with Categorical Features
Code example:

```python
# Binary encoding of categorical features
from category_encoders import BinaryEncoder

binary_encoders = {}
for feature in final_binary_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])
    
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]
    
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)
    binary_encoders[feature] = encoder
```

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
    x_test_boot = x_y_test_boot.drop(columns='satisfaction')
    y_test_boot = x_y_test_boot['satisfaction']

    # Model predictions with optimal threshold
    predicted_probas = model.predict_proba(x_test_boot)
    y_pred = predicted_probas[:, 1] >= best_threshold

    # Calculating quality metrics
    boot_accuracies.append(accuracy_score(y_test_boot, y_pred))
    boot_precisions.append(precision_score(y_test_boot, y_pred,
                                           zero_division=0))
    boot_recalls.append(recall_score(y_test_boot, y_pred, zero_division=0))
    boot_f1_scores.append(f1_score(y_test_boot, y_pred, zero_division=0))
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

## Interpreting Logistic Regression Model Coefficients

### Odds Ratio
**`OR = e^w`** — how many times the odds change when the feature changes by 1 unit

### Interpreting Coefficients
- **Positive coefficient**: increasing the feature increases the probability of the positive class
- **Negative coefficient**: increasing the feature decreases the probability of the positive class
- **Coefficient magnitude**: strength of influence on the log odds

Code example:

```python
# Obtaining coefficients and their significance
coefficients = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': model.coef_[0],
    'Exp(Coefficient)': np.exp(model.coef_[0]),
    'Effect on Odds': [
        'Increases' if coef > 0 else 'Decreases' for coef in model.coef_[0]]
})

# Sorting by absolute coefficient value
coefficients_sorted = coefficients.reindex(
    coefficients['Coefficient'].abs().sort_values(ascending=False).index)

print('\n\nInterpreting model coefficients...')
print('\nModel coefficients (sorted by influence):')
print(coefficients_sorted.to_string(index=False))
```

---

## Conclusion

Logistic regression is a powerful and efficient method for binary classification tasks that combines predictive ability with high interpretability. Unlike decision trees, it provides probabilistic estimates, allowing for more flexible adjustment of classification thresholds depending on business requirements.

# Linear Regression Information

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Mathematical Model](#mathematical-model)
3. [Least Squares Method](#least-squares-method)
4. [Solution in Matrix Form](#solution-in-matrix-form)
5. [Gradient Descent](#gradient-descent)
6. [Quality Metrics](#quality-metrics)
7. [Regularization](#regularization)
8. [Practical Aspects](#practical-aspects)
9. [Assumption Checking](#assumption-checking)
10. [Analysis and Interpretation of Results](#analysis-and-interpretation-of-results)
11. [Bootstrap and Confidence Intervals](#bootstrap-and-confidence-intervals)
12. [Conclusion](#conclusion)

---

## Basic Concepts

### What is Linear Regression?
Linear regression is a statistical method for modeling the relationship between:
- **Explanatory variables** (features, factors) **`x_1, x_2, ..., x_n`**
- **Dependent variable** (target variable) **`y`**

### Types of Linear Regression
- **Simple Linear Regression**: one independent variable
- **Multiple Linear Regression**: several independent variables

### Geometric Interpretation
For the case with one feature, the model tries to find a straight line that best approximates the cloud of data points in the feature space.

---

## Mathematical Model

### General Formula
**`ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ`**

where:
- **`ŷ`** — predicted value of the target variable
- **`w₀`** — intercept
- **`w₁, w₂, ..., wₙ`** — feature coefficients
- **`x₁, x₂, ..., xₙ`** — feature values

### Vector Form
**`ŷ = wᵀx`**

where **`w = [w₀, w₁, ..., wₙ]ᵀ`** and **`x = [1, x₁, x₂, ..., xₙ]ᵀ`** (with added one)

---

## Least Squares Method

### Objective Function
Minimization of the sum of squared errors:

**`MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`**

### Solution for One Feature
For the model **`ŷ = w₀ + w₁x`**:

**`w₁ = [Σ(xᵢ - x̄)(yᵢ - ȳ)] / [Σ(xᵢ - x̄)²]`**

**`w₀ = ȳ - w₁x̄`**

where:
- **`x̄`** — sample mean of the feature
- **`ȳ`** — sample mean of the target variable

### Statistical Properties of Estimates
- **Unbiasedness**: **`E[w] = w_{true}`**
- **Efficiency**: minimum variance among unbiased estimates
- **Consistency**: as the data volume increases, the estimate converges to the true value

---

## Solution in Matrix Form

### Matrix Representation
**`y = Xw + ε`**

where:
- **`y`** — target value vector of dimension **`n × 1`**
- **`X`** — feature matrix of dimension **`n × (m+1)`**
- **`w`** — parameter vector of dimension **`(m+1) × 1`**
- **`ε`** — error vector of dimension **`n × 1`**

### Normal Equation
**`w = (XᵀX)⁻¹Xᵀy`**

### Conditions for Solution Existence
- Matrix **`XᵀX`** must be invertible
- Features must not be linearly dependent
- Number of observations must be greater than the number of features

---

## Gradient Descent

### Iterative Formula
**`w^{(k+1)} = w^{(k)} - η * ∇Q(w^{(k)})`**

where:
- **`η`** — learning rate
- **`∇Q(w)`** — gradient of the loss function

### Types of Gradient Descent
1. **Full Gradient Descent**: using all data to compute the gradient
2. **Stochastic Gradient Descent (SGD)**: using one random observation
3. **Mini-batch Gradient Descent**: using a data subsample

### MSE Gradient
**`∇MSE(w) = -(2/n) * Xᵀ(y - Xw)`**

---

## Quality Metrics

### Main Metrics

<div align="center">

<table>
    <tr align="center">
        <th>Metric</th>
        <th>Formula</th>
        <th>Features</th>
    </tr>
    <tr align="center">
        <td><strong>MSE</strong></td>
        <td><code>(1/n) * Σ(yᵢ - ŷᵢ)²</code></td>
        <td>Sensitive to outliers, differentiable</td>
    </tr>
    <tr align="center">
        <td><strong>RMSE</strong></td>
        <td><code>√MSE</code></td>
        <td>Interpretable in units of the target variable</td>
    </tr>
    <tr align="center">
        <td><strong>MAE</strong></td>
        <td><code>(1/n) * Σ|yᵢ - ŷᵢ|</code></td>
        <td>Less sensitive to outliers</td>
    </tr>
    <tr align="center">
        <td><strong>R²</strong></td>
        <td><code>1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]</code></td>
        <td>Proportion of explained variance</td>
    </tr>
</table>

</div>

### Additional Metrics
- **Adjusted R²**: **`1 - [(1-R²)(n-1)/(n-p-1)]`**
- **MAPE**: **`(100/n) * Σ\|(yᵢ - ŷᵢ)/yᵢ\|`**
- **MSLE**: **`(1/n) * Σ(log(yᵢ+1) - log(ŷᵢ+1))²`**

### R² Interpretation
- **0.0**: model is no better than the mean value
- **0.0-0.3**: weak explanatory power
- **0.3-0.7**: moderate explanatory power
- **0.7-1.0**: strong explanatory power
- **1.0**: perfect prediction (usually a sign of overfitting)

---

## Regularization

### Overfitting Problem
When the model is too complex and fits the noise in the data.

### Ridge Regression (L2 Regularization)
**`Q(w) = MSE + α * Σwᵢ²`**

Solution: **`w = (XᵀX + αI)⁻¹Xᵀy`**

### Lasso Regression (L1 Regularization)
**`Q(w) = MSE + α * Σ\|wᵢ\|`**

Leads to zeroing out some coefficients

### Elastic Net
Combination of L1 and L2 regularization:
**`Q(w) = MSE + α * ρ * Σ\|wᵢ\| + α * (1-ρ)/2 * Σwᵢ²`**

---

## Practical Aspects

### Data Preprocessing
1. **Standardization**: **`x' = (x - μ)/σ`**
2. **Normalization**: **`x' = (x - min)/(max - min)`**
3. **Handling missing values**
4. **Outlier processing**
5. **Encoding categorical variables**:
   - One-Hot Encoding
   - Target Encoding
   - Label Encoding

### Implementation Features
Example from code:

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)
```

### Data Analysis Before Modeling
1. Exploring the distribution of the target variable
2. Analyzing correlations between features and the target variable
3. Visualizing relationships (scatter plots, box plots)
4. Exploring categorical variables through analysis of means by groups

---

## Assumption Checking

### Linearity
The relationship between features and the target variable should be linear.
**Check**: residual analysis (residuals vs fitted plot)

### Homoscedasticity
Constant variance of errors.
**Check**: residual analysis (uniform distribution of points)

### Normality of Error Distribution
Errors should be normally distributed.
**Check**: Q-Q plot, Shapiro-Wilk test

### Absence of Multicollinearity
Features should not be highly correlated with each other.
**Check**: correlation matrix, VIF (Variance Inflation Factor)

### Independence of Errors
Errors should not be correlated with each other.
**Check**: Durbin-Watson test

---

## Analysis and Interpretation of Results

### Coefficient Analysis
- **Coefficient sign**: direction of the feature's influence on the target variable
- **Coefficient magnitude**: strength of influence (with standardized features)
- **Statistical significance**: p-value < 0.05

### Feature Importance
Sorting features by absolute coefficient values

Example from code:

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)
```

### Residual Analysis
- **Residual plot**: residuals vs predicted values
- **Residual distribution**: should be normal
- **Outliers**: points with large residuals

---

## Bootstrap and Confidence Intervals

### Bootstrap Method
A statistical method for estimating metric uncertainty through multiple data resampling.

### Bootstrap Implementation
Example from code:

```python
n_bootstraps = 1000
confidence_level = 0.95

r2_scores = []
mae_scores = []
rmse_scores = []

print(f'\n\nRunning bootstrap ({n_bootstraps} iterations)...')
for i in range(n_bootstraps):
    if (i + 1) % 100 == 0:
        print(f'Iterations completed: {i + 1}/{n_bootstraps}')

    x_boot, y_boot = resample(x_test, y_test, random_state=i)
    y_pred_boot = model.predict(x_boot)

    r2_scores.append(r2_score(y_boot, y_pred_boot))
    mae_scores.append(mean_absolute_error(y_boot, y_pred_boot))
    rmse_scores.append(np.sqrt(mean_squared_error(y_boot, y_pred_boot)))
```

### Confidence Interval Calculation
Example from code:

```python
def calculate_confidence_interval(scores, confidence=0.95):
    """Calculating the confidence interval for the scores array"""
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper
```

### Confidence Interval Interpretation
- **Narrow interval**: high confidence in the estimate
- **Wide interval**: high uncertainty
- **Includes zero**: effect may not be significant

### Practical Usefulness of the Model
Worst-case analysis based on the lower confidence interval bound:
- Minimum expected R²
- Maximum expected errors (MAE, RMSE)

---

## Conclusion

Linear regression is a fundamental and powerful machine learning tool that remains one of the most popular methods for regression tasks due to its simplicity, interpretability, and computational efficiency.

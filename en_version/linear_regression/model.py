from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample


# =============================================================================
# 1. Model creation and training
# =============================================================================
print('=' * 80)
print('1. Model creation and training')
print('=' * 80)
sleep(2)

# Reading data
x_train = pd.read_csv('data_linear_regression/x_train_data.csv')
y_train = pd.read_csv('data_linear_regression/y_train_data.csv')
x_test = pd.read_csv('data_linear_regression/x_test_data.csv')
y_test = pd.read_csv('data_linear_regression/y_test_data.csv')

print(f'\n\nTraining set size: {x_train.shape}')
print(f'Test set size: {x_test.shape}')
sleep(5)

# Model creation and training
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# =============================================================================
# 2. Model quality metrics
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Model quality metrics')
print('=' * 80)
sleep(2)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
retraining = r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print('\n\nModel quality metrics:')
print(f'R² on training set: {r2_train}')
print(f'R² on test set: {r2_test}')
print(f'Overfitting (R² difference): {retraining}')
print(f'MAE on test set: {mae}')
print(f'MSE on test set: {mse}')
print(f'RMSE on test set: {rmse}')
sleep(10)


# =============================================================================
# 3. Bootstrap for confidence intervals of quality metrics
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Bootstrap for confidence intervals of quality metrics')
print('=' * 80)
sleep(2)

# Bootstrap parameters
n_bootstraps = 1000
confidence_level = 0.95

# Arrays to store metrics on bootstrap samples
r2_scores = []
mae_scores = []
rmse_scores = []

print(f'\n\nRunning bootstrap ({n_bootstraps} iterations)...')
for i in range(n_bootstraps):
    if (i + 1) % 100 == 0:
        print(f'Completed iterations: {i + 1}/{n_bootstraps}')

    # Create bootstrap sample (with replacement)
    x_boot, y_boot = resample(x_test, y_test, random_state=i)

    # Prediction on bootstrap sample
    y_pred_boot = model.predict(x_boot)

    # Calculate metrics
    r2_scores.append(r2_score(y_boot, y_pred_boot))
    mae_scores.append(mean_absolute_error(y_boot, y_pred_boot))
    rmse_scores.append(np.sqrt(mean_squared_error(y_boot, y_pred_boot)))

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval for scores array"""
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper

# Confidence intervals for each metric
r2_ci = calculate_confidence_interval(r2_scores, confidence_level)
mae_ci = calculate_confidence_interval(mae_scores, confidence_level)
rmse_ci = calculate_confidence_interval(rmse_scores, confidence_level)

# Output results
print('\n\nConfidence intervals of metrics (bootstrap):')
print('R²:')
print(f'Interval mean value: {r2_test}')
print(f'Interval: [{r2_ci[0]}, {r2_ci[1]}]')
print(f'Interval description: (95% CI, width: {r2_ci[1] - r2_ci[0]})')
sleep(10)

print('\nMAE:')
print(f'Interval mean value: {mae}')
print(f'Interval: [{mae_ci[0]}, {mae_ci[1]}]')
print(f'Interval description: (95% CI, width: {mae_ci[1] - mae_ci[0]})')
sleep(10)

print('\nRMSE:')
print(f'Interval mean value: {rmse}')
print(f'Interval: [{rmse_ci[0]}, {rmse_ci[1]}]')
print(f'Interval description: (95% CI, width: {rmse_ci[1] - rmse_ci[0]})')
sleep(10)

# Analysis of model practical usefulness
print('\n\nAnalysis of model practical usefulness:')
print(f'In worst case (lower bound of CI):')
print(f'  R² could be only {r2_ci[0]}')
print(f'  MAE could reach {mae_ci[1]}')
print(f'  RMSE could reach {rmse_ci[1]}')
sleep(10)


# =============================================================================
# 4. Model coefficients interpretation
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Model coefficients interpretation')
print('=' * 80)
sleep(2)

print('\n\nCoefficients interpretation:')
feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
})

# Sorting by importance
feature_importance = feature_importance.sort_values(
    'abs_coefficient', ascending=False)
for _, row in feature_importance.iterrows():
    sign = '+' if row['coefficient'] > 0 else '-'
    print(f'{row['feature']}: {sign}{abs(row['coefficient'])}')
sleep(10)


# =============================================================================
# 5. Predictions vs Actual values visualization
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Predictions vs Actual values visualization')
print('=' * 80)
sleep(2)

print('\n\nRunning plot analysis...')

sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Predictions vs Actual values')

plt.subplot(1, 2, 2)
residuals = y_test.values.flatten() - y_test_pred.flatten()
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals analysis')

plt.tight_layout()
plt.show()

print('\n\nTraining and analysis completed!\n\n')

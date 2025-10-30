from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


# =============================================================================
# 1. Data loading and preparation
# =============================================================================
print('=' * 80)
print('1. Data loading and preparation')
print('=' * 80)
sleep(2)

# Reading data
x_train = pd.read_csv('data_logistic_regression/x_train_data.csv')
y_train = pd.read_csv('data_logistic_regression/y_train_data.csv')
x_test = pd.read_csv('data_logistic_regression/x_test_data.csv')
y_test = pd.read_csv('data_logistic_regression/y_test_data.csv')

# Splitting training data into training and validation sets (75%:25%)
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=52, stratify=y_train)

# Converting from 2D array to 1D for correctness
y_train_split = y_train_split.values.ravel()
y_val = y_val.values.ravel()

print(f'\n\nTraining set size: {x_train_split.shape}')
print(f'Validation set size: {x_val.shape}')
sleep(5)


# =============================================================================
# 2. Model creation and training
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Model creation and training')
print('=' * 80)
sleep(2)

# Creating and training the model
model = LogisticRegression(penalty=None)
model.fit(x_train_split, y_train_split)

# Predictions
train_predictions = model.predict(x_train_split)
train_probabilities = model.predict_proba(x_train_split)

# Intermediate checks
print('\n\nModel predictions:')

print('\nFirst 10 predicted classes:')
print(train_predictions[:10])
sleep(5)

print('\nFirst 10 predicted probabilities:')
print(train_probabilities[:10])
sleep(10)


# =============================================================================
# 3. Model evaluation on training and validation sets
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Model evaluation on training and validation sets')
print('=' * 80)
sleep(2)

# Quality metrics on training set
train_accuracy = accuracy_score(y_train_split, train_predictions)
train_precision = precision_score(y_train_split, train_predictions,
                                  zero_division=0)
train_recall = recall_score(y_train_split, train_predictions, zero_division=0)
train_f1 = f1_score(y_train_split, train_predictions, zero_division=0)

print('\n\nQuality metrics on training set:')
print(f'Accuracy: {train_accuracy}')
print(f'Precision: {train_precision}')
print(f'Recall: {train_recall}')
print(f'F1-score: {train_f1}')
sleep(10)

# Quality metrics on validation set
val_predictions = model.predict(x_val)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, zero_division=0)
val_recall = recall_score(y_val, val_predictions, zero_division=0)
val_f1 = f1_score(y_val, val_predictions, zero_division=0)

print('\nQuality metrics on validation set:')
print(f'Accuracy: {val_accuracy}')
print(f'Precision: {val_precision}')
print(f'Recall: {val_recall}')
print(f'F1-score: {val_f1}')
sleep(10)


# =============================================================================
# 4. Finding the optimal threshold value
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Finding the optimal threshold value')
print('=' * 80)
sleep(2)

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

print('\n\nFinding optimal threshold for classification...')
print(f'Best F1-score on validation set: {best_f1}')
print(f'Optimal threshold: {best_threshold}')
sleep(5)

# Visualization of F1-score dependence on threshold
thresholds = np.linspace(0, 1, 100)
f1_scores = []

for threshold in thresholds:
    y_val_pred = val_probabilities > threshold
    f1_scores.append(f1_score(y_val, y_val_pred, zero_division=0))

print('\n\nAnalyzing the plot...')

sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores)
plt.axvline(x=best_threshold, color='r', linestyle='--',
            label=f'Optimal threshold: {best_threshold}')
plt.xlabel('Threshold value')
plt.ylabel('F1-score')
plt.title('F1-score dependence on threshold value')
plt.legend()
plt.grid(True)
plt.show()

sleep(2)


# =============================================================================
# 5. Model evaluation on test set
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Model evaluation on test set')
print('=' * 80)
sleep(2)

# Predictions with optimal threshold
test_probabilities = model.predict_proba(x_test)[:, 1]
test_predictions_optimal = test_probabilities > best_threshold

# Quality metrics on test set
test_accuracy = accuracy_score(y_test, test_predictions_optimal)
test_precision = precision_score(y_test, test_predictions_optimal,
                                 zero_division=0)
test_recall = recall_score(y_test, test_predictions_optimal, zero_division=0)
test_f1 = f1_score(y_test, test_predictions_optimal, zero_division=0)

print('\n\nQuality metrics on test set (with optimal threshold):')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1-score: {test_f1}')
sleep(10)

print('\n\nAnalyzing the plot...')

# Confusion matrix
cm = confusion_matrix(y_test, test_predictions_optimal)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion matrix on test set')
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()
sleep(2)

# Full classification report
print('\n\nClassification report:')
print(classification_report(y_test, test_predictions_optimal, zero_division=0))
sleep(10)


# =============================================================================
# 6. Bootstrap for metric confidence intervals
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Bootstrap for metric confidence intervals')
print('=' * 80)
sleep(2)

# Combining test data
x_y_test = x_test.copy(deep=True)
x_y_test['satisfaction'] = y_test.values

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

# Displaying confidence intervals
print('\n\nMetric confidence intervals (bootstrap):')
print('Accuracy:')
print(f'Interval mean value: {test_accuracy}')
print(f'Interval: [{accuracy_ci[0]}, {accuracy_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    accuracy_ci[1] - accuracy_ci[0]})')
sleep(10)

print('\nPrecision:')
print(f'Interval mean value: {test_precision}')
print(f'Interval: [{precision_ci[0]}, {precision_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    precision_ci[1] - precision_ci[0]})')
sleep(10)

print('\nRecall:')
print(f'Interval mean value: {test_recall}')
print(f'Interval: [{recall_ci[0]}, {recall_ci[1]}]')
print(f'Interval description: (95% CI, width: {recall_ci[1] - recall_ci[0]})')
sleep(10)

print('\nF1:')
print(f'Interval mean value: {test_f1}')
print(f'Interval: [{f1_ci[0]}, {f1_ci[1]}]')
print(f'Interval description: (95% CI, width: {f1_ci[1] - f1_ci[0]})')
sleep(10)

print('\n\nAnalyzing the plot...')

# Visualization of metric distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics = [boot_accuracies, boot_precisions, boot_recalls, boot_f1_scores]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
colors = ['blue', 'green', 'red', 'purple']

for i, (ax, metric, name, color) in enumerate(zip(
        axes.flat, metrics, metric_names, colors)):
    sns.histplot(metric, ax=ax, color=color, kde=True)
    ax.set_title(f'Distribution of "{name}" metric')
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

sleep(2)


# =============================================================================
# 7. Model coefficients interpretation
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Model coefficients interpretation')
print('=' * 80)
sleep(2)

# Getting coefficients and their significance
coefficients = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': model.coef_[0],
    'Exp(Coefficient)': np.exp(model.coef_[0]),
    'Impact on Odds': [
        'Increases' if coef > 0 else 'Decreases' for coef in model.coef_[0]]
})

# Sorting by absolute coefficient value
coefficients_sorted = coefficients.reindex(
    coefficients['Coefficient'].abs().sort_values(ascending=False).index)

print('\n\nModel coefficients (sorted by impact):')
print(coefficients_sorted.to_string(index=False))
sleep(10)

print('\n\nAnalyzing the plot...')

# Visualization of feature importance
plt.figure(figsize=(12, 8))
colors = ['red' if coef < 0 else 'blue'
          for coef in coefficients_sorted['Coefficient']]
plt.barh(coefficients_sorted['Feature'],
         coefficients_sorted['Coefficient'], color=colors)
plt.xlabel('Coefficient value')
plt.title('Feature importance in logistic regression model')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

sleep(2)

print('\n\nTraining and analysis completed!\n\n')

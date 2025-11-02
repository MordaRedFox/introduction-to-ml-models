from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
x_train = pd.read_csv('data_decision_tree/x_train_data.csv')
y_train = pd.read_csv('data_decision_tree/y_train_data.csv')
x_test = pd.read_csv('data_decision_tree/x_test_data.csv')
y_test = pd.read_csv('data_decision_tree/y_test_data.csv')

# Converting from 2D array to 1D for correctness
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f'\n\nTraining set size: {x_train.shape}')
print(f'Test set size: {x_test.shape}')
sleep(5)

# Determining image size based on selected pixels
n_pixels = x_train.shape[1]
# Nearest square size
img_size = int(np.sqrt(n_pixels))
print(f'\n\nAssumed image size: {img_size}x{img_size} = '
      f'{img_size ** 2} pixels')
print(f'Actual number of pixels: {n_pixels} (out of 784 possible)')
sleep(5)

# Splitting data into training and validation sets (80%:20%)
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=52)

print(f'\n\nTraining set size: {x_train_split.shape}')
print(f'Validation set size: {x_val.shape}')
sleep(5)

def print_image(pixels, ax=None, title=None, original_size=28):
    """Visualizes an image from a pixel array considering reduced
    dimensionality"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    # Creating an array of zeros with size 28x28 = 784 pixels (for the image)
    full_image = np.zeros(original_size ** 2)

    # Filling only those pixels that are present in the final data (pixel name
    # format in data - "pixelX", where X is a number)
    pixel_indices = []
    for col in x_train.columns:
        if col.startswith('pixel'):
            try:
                # Extracting pixel number from the name
                idx = int(col[5:])
                pixel_indices.append(idx)
            except ValueError:
                continue

    # Sorting indices and filling values
    pixel_indices.sort()
    for i, pixel_idx in enumerate(pixel_indices[:len(pixels)]):
        if pixel_idx < len(full_image):
            full_image[pixel_idx] = pixels[i]

    # Converting linear array to 28x28 matrix
    image = full_image.reshape(original_size, original_size)
    # Setting appearance
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return ax


# =============================================================================
# 2. Model creation and training
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Model creation and training')
print('=' * 80)
sleep(2)

# Dictionary of all models and their results
models_results = {}

# Decision trees with different hyperparameters
min_samples_leaf_values = [1, 3, 5, 10]

print('\n\nTraining decision trees...')
for min_leaf in min_samples_leaf_values:
    print(f'\nTraining DecisionTree (min_samples_leaf={min_leaf})...')
    print(f'Model name: DT_leaf_{min_leaf}')

    # Creating and training the model
    model_dt = DecisionTreeClassifier(min_samples_leaf=min_leaf,
                                      criterion='gini', random_state=52)
    model_dt.fit(x_train_split, y_train_split)

    # Model predictions
    train_pred = model_dt.predict(x_train_split)
    val_pred = model_dt.predict(x_val)

    # Model accuracy
    train_accuracy = accuracy_score(y_train_split, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    # Saving data to the dictionary of all models
    models_results[f'DT_leaf_{min_leaf}'] = {
        'model': model_dt,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

    print(f'Model accuracy on training set: {train_accuracy}')
    print(f'Model accuracy on validation set: {val_accuracy}')
    print(f'Accuracy difference: {train_accuracy - val_accuracy}')
    sleep(10)

# Random forest with different hyperparameters
n_estimators_values = [100, 300, 500, 1000]

print('\n\nTraining random forest...')
for n_est in n_estimators_values:
    print(f'\nTraining RandomForest (n_estimators={n_est})...')
    print(f'Model name: RF_est_{n_est}')

    # Creating and training the model
    model_rf = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=3,
                                      max_features='sqrt', criterion='gini',
                                      random_state=52, n_jobs=-1)
    model_rf.fit(x_train_split, y_train_split)

    # Model predictions
    train_pred = model_rf.predict(x_train_split)
    val_pred = model_rf.predict(x_val)

    # Model accuracy
    train_accuracy = accuracy_score(y_train_split, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    # Saving data to the dictionary of all models
    models_results[f'RF_est_{n_est}'] = {
        'model': model_rf,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

    print(f'Model accuracy on training set: {train_accuracy}')
    print(f'Model accuracy on validation set: {val_accuracy}')
    print(f'Accuracy difference: {train_accuracy - val_accuracy}')
    sleep(10)


# =============================================================================
# 3. Selecting the best model for further analysis
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Selecting the best model for further analysis')
print('=' * 80)
sleep(2)

best_model_name = max(models_results.keys(),
                      key=lambda x: models_results[x]['val_accuracy'])
best_model = models_results[best_model_name]['model']

print(f'\n\nBest model: {best_model_name}')
print(f'Accuracy on validation set: {models_results[
    best_model_name]['val_accuracy']}')
sleep(5)

# Model comparison visualization
sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 6))
models_names = list(models_results.keys())
list_train_accuracy = [models_results[name]['train_accuracy']
                       for name in models_names]
list_val_accuracy = [models_results[name]['val_accuracy']
                     for name in models_names]

print('\n\nAnalyzing the plot...')
x_pos = np.arange(len(models_names))
width = 0.35
plt.bar(x_pos - width / 2, list_train_accuracy, width,
        label='Training set', alpha=0.7)
plt.bar(x_pos + width / 2, list_val_accuracy, width,
        label='Validation set', alpha=0.7)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model accuracy comparison')
plt.xticks(x_pos, models_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
sleep(5)


# =============================================================================
# 4. Evaluating the best model on the validation set
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Evaluating the best model on the validation set')
print('=' * 80)
sleep(2)

best_val_pred = best_model.predict(x_val)

# Model quality metrics
best_val_accuracy = accuracy_score(y_val, best_val_pred)
best_val_precision = precision_score(y_val, best_val_pred,
                                     average='weighted', zero_division=0)
best_val_recall = recall_score(y_val, best_val_pred,
                               average='weighted', zero_division=0)
best_val_f1 = f1_score(y_val, best_val_pred,
                       average='weighted', zero_division=0)

print(f'\n\nQuality metrics of model {best_model_name} on validation set:')
print(f'Accuracy: {best_val_accuracy}')
print(f'Precision: {best_val_precision}')
print(f'Recall: {best_val_recall}')
print(f'F1-score: {best_val_f1}')
sleep(10)

# Confusion matrix visualization
print('\n\nAnalyzing the plot...')
cm = confusion_matrix(y_val, best_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion matrix on validation set')
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.show()


# =============================================================================
# 5. Error analysis of the best model on the validation set
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Error analysis of the best model on the validation set')
print('=' * 80)
sleep(2)

error_mask = best_val_pred != y_val
x_errors =  x_val[error_mask]
y_errors_true = y_val[error_mask]
y_errors_pred = best_val_pred[error_mask]

print(f'\n\nNumber of errors: {len(x_errors)}')
print(f'Error rate: {len(x_errors) / len(y_val):.4f}')
sleep(10)

# Analysis of most frequent errors
error_counts = pd.DataFrame({
    'true': y_errors_true,
    'pred': y_errors_pred
}).value_counts().head(10).reset_index()
error_counts.columns = ['True Class', 'Predicted Class', 'Count']

print(f'\n\nMost frequent errors:')
print(error_counts.to_string(index=False))
sleep(10)

# Error examples visualization
print('\n\nVisualizing the most frequent errors:')
if len(error_counts) > 0:
    # Selecting 5 most frequent errors
    common_errors = error_counts.head(5)

    for i in range(len(common_errors)):
        true_class = common_errors.iloc[i]['True Class']
        pred_class = common_errors.iloc[i]['Predicted Class']
        count = common_errors.iloc[i]['Count']

        print(f'Visualizing errors {true_class} -> {pred_class} '
              f'(count: {count})')

        # Finding examples of this error
        error_indices = np.where(
            (y_errors_true == true_class) & (y_errors_pred == pred_class))[0]

        # Selecting no more than 6 error examples
        examples_to_show = min(6, len(error_indices))

        # Separate plot for each error
        n_cols = 3
        n_rows = (examples_to_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        # Converting axes to array for consistency if there's only one row
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        plot_idx = 0

        for j in range(examples_to_show):
            idx = error_indices[j]
            actual_idx = x_errors.index[idx]
            pixels = x_val.loc[actual_idx].values

            # Calculating subplot coordinates
            row = plot_idx // n_cols
            col = plot_idx % n_cols

            ax = print_image(pixels=pixels, ax=axes[row, col],
                             title=f'Example {j+1}')
            plot_idx += 1

        # Hiding unused subplots
        for k in range(plot_idx, n_rows * n_cols):
            row = k // n_cols
            col = k % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(f'Classification error: {true_class} → {pred_class}')
        plt.tight_layout()
        plt.show()
        sleep(2)
else:
    print('No errors to visualize!')
sleep(5)


# =============================================================================
# 6. Evaluating the best model on the test set
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Evaluating the best model on the test set')
print('=' * 80)
sleep(2)

best_test_pred = best_model.predict(x_test)

best_test_accuracy = accuracy_score(y_test, best_test_pred)
best_test_precision = precision_score(y_test, best_test_pred,
                                      average='weighted', zero_division=0)
best_test_recall = recall_score(y_test, best_test_pred,
                                average='weighted', zero_division=0)
best_test_f1 = f1_score(y_test, best_test_pred,
                        average='weighted', zero_division=0)

print(f'\n\nQuality metrics of model {best_model_name} on test set:')
print(f'Accuracy: {best_test_accuracy}')
print(f'Precision: {best_test_precision}')
print(f'Recall: {best_test_recall}')
print(f'F1-score: {best_test_f1}')
sleep(10)

print('\n\nClassification report:')
print(classification_report(y_test, best_test_pred, zero_division=0))
sleep(10)


# =============================================================================
# 7. Bootstrap for confidence intervals of quality metrics
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Bootstrap for confidence intervals of quality metrics')
print('=' * 80)
sleep(2)

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
print('\n\nConfidence intervals of metrics (bootstrap):')
print('Accuracy:')
print(f'Interval mean value: {best_test_accuracy}')
print(f'Interval: [{accuracy_ci[0]}, {accuracy_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    accuracy_ci[1] - accuracy_ci[0]})')
sleep(10)

print('\nPrecision:')
print(f'Interval mean value: {best_test_precision}')
print(f'Interval: [{precision_ci[0]}, {precision_ci[1]}]')
print(f'Interval description: (95% CI, width: {
    precision_ci[1] - precision_ci[0]})')
sleep(10)

print('\nRecall:')
print(f'Interval mean value: {best_test_recall}')
print(f'Interval: [{recall_ci[0]}, {recall_ci[1]}]')
print(f'Interval description: (95% CI, width: {recall_ci[1] - recall_ci[0]})')
sleep(10)

print('\nF1:')
print(f'Interval mean value: {best_test_f1}')
print(f'Interval: [{f1_ci[0]}, {f1_ci[1]}]')
print(f'Interval description: (95% CI, width: {f1_ci[1] - f1_ci[0]})')
sleep(10)

# Metrics distribution visualization
print('\n\nVisualizing metrics distribution...')
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
sleep(5)


# =============================================================================
# 8. Interpreting the best model
# =============================================================================
print('\n\n' + '=' * 80)
print('8. Interpreting the best model')
print('=' * 80)
sleep(2)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_

    importance_df = pd.DataFrame({
        'pixel': x_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print('\n\nTop 10 most important pixels:')
    print(importance_df.head(10))
    sleep(10)

    # Visualizing important pixels on 28x28 image
    plt.figure(figsize=(12, 10))

    # Creating pixel importance map
    importance_map = np.zeros(28 * 28)

    # Filling important pixels
    for i, (pixel_name, importance) in enumerate(zip(
                x_train.columns, feature_importance)):
        if pixel_name.startswith('pixel'):
            try:
                pixel_idx = int(pixel_name[5:])
                if pixel_idx < len(importance_map):
                    importance_map[pixel_idx] = importance
            except ValueError:
                continue

    print('\n\nAnalyzing the plot...')

    # Visualizing pixel importance map
    plt.imshow(importance_map.reshape(28, 28), cmap='hot',
               interpolation='nearest')
    plt.colorbar(label='Relative importance')
    plt.title('Pixel importance map for classification')
    plt.axis('off')
    plt.show()

print('\n\nTraining and analysis completed!\n\n')

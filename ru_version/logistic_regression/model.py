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
# 1. Загрузка и подготовка данных
# =============================================================================
print('=' * 80)
print('1. Загрузка и подготовка данных')
print('=' * 80)
sleep(2)

# Чтение данных
x_train = pd.read_csv('data_logistic_regression/x_train_data.csv')
y_train = pd.read_csv('data_logistic_regression/y_train_data.csv')
x_test = pd.read_csv('data_logistic_regression/x_test_data.csv')
y_test = pd.read_csv('data_logistic_regression/y_test_data.csv')

# Разделение обучающей выборки на обучающую и валидационную (75%:25%)
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=52, stratify=y_train)

# Преобразование из 2D массива в 1D для корректности
y_train_split = y_train_split.values.ravel()
y_val = y_val.values.ravel()

print(f'\n\nРазмер обучающей выборки: {x_train_split.shape}')
print(f'Размер валидационной выборки: {x_val.shape}')
sleep(5)


# =============================================================================
# 2. Создание и обучение модели
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Создание и обучение модели')
print('=' * 80)
sleep(2)

# Создание и обучение модели
model = LogisticRegression(penalty=None)
model.fit(x_train_split, y_train_split)

# Предсказания
train_predictions = model.predict(x_train_split)
train_probabilities = model.predict_proba(x_train_split)

# Промежуточные проверки
print('\n\nПредсказания модели:')

print('\nПервые 10 предсказанных классов:')
print(train_predictions[:10])
sleep(5)

print('\nПервые 10 предсказанных вероятностей:')
print(train_probabilities[:10])
sleep(10)


# =============================================================================
# 3. Оценка качества модели на обучающей и валидационной выборках
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Оценка качества модели на обучающей и валидационной выборках')
print('=' * 80)
sleep(2)

# Метрики качества на обучающей выборке
train_accuracy = accuracy_score(y_train_split, train_predictions)
train_precision = precision_score(y_train_split, train_predictions,
                                  zero_division=0)
train_recall = recall_score(y_train_split, train_predictions, zero_division=0)
train_f1 = f1_score(y_train_split, train_predictions, zero_division=0)

print('\n\nМетрики качества на обучающей выборке:')
print(f'Accuracy: {train_accuracy}')
print(f'Precision: {train_precision}')
print(f'Recall: {train_recall}')
print(f'F1-score: {train_f1}')
sleep(10)

# Метрики качества на валидационной выборке
val_predictions = model.predict(x_val)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, zero_division=0)
val_recall = recall_score(y_val, val_predictions, zero_division=0)
val_f1 = f1_score(y_val, val_predictions, zero_division=0)

print('\nМетрики качества на валидационной выборке:')
print(f'Accuracy: {val_accuracy}')
print(f'Precision: {val_precision}')
print(f'Recall: {val_recall}')
print(f'F1-score: {val_f1}')
sleep(10)


# =============================================================================
# 4. Подбор оптимального порогового значения
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Подбор оптимального порогового значения')
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

print('\n\nПодбор оптимального порогового значения для классификации...')
print(f'Лучшее значение F1-score на валидационной выборке: {best_f1}')
print(f'Оптимальный порог: {best_threshold}')
sleep(5)

# Визуализация зависимости F1-score от порога
thresholds = np.linspace(0, 1, 100)
f1_scores = []

for threshold in thresholds:
    y_val_pred = val_probabilities > threshold
    f1_scores.append(f1_score(y_val, y_val_pred, zero_division=0))

print('\n\nВыполняется анализ графика...')

sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores)
plt.axvline(x=best_threshold, color='r', linestyle='--',
            label=f'Оптимальный порог: {best_threshold}')
plt.xlabel('Пороговое значение')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от порогового значения')
plt.legend()
plt.grid(True)
plt.show()

sleep(2)


# =============================================================================
# 5. Оценка качества модели на тестовой выборке
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Оценка качества модели на тестовой выборке')
print('=' * 80)
sleep(2)

# Предсказания с оптимальным порогом
test_probabilities = model.predict_proba(x_test)[:, 1]
test_predictions_optimal = test_probabilities > best_threshold

# Метрики качества на тестовой выборке
test_accuracy = accuracy_score(y_test, test_predictions_optimal)
test_precision = precision_score(y_test, test_predictions_optimal,
                                 zero_division=0)
test_recall = recall_score(y_test, test_predictions_optimal, zero_division=0)
test_f1 = f1_score(y_test, test_predictions_optimal, zero_division=0)

print('\n\nМетрики качества на тестовой выборке (с оптимальным порогом):')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1-score: {test_f1}')
sleep(10)

print('\n\nВыполняется анализ графика...')

# Матрица ошибок
cm = confusion_matrix(y_test, test_predictions_optimal)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Предсказано 0', 'Предсказано 1'],
            yticklabels=['Фактически 0', 'Фактически 1'])
plt.title('Матрица ошибок на тестовой выборке')
plt.xlabel('Предсказанный класс')
plt.ylabel('Фактический класс')
plt.show()
sleep(2)

# Полный отчет о классификации
print('\nОтчёт о классификации:')
print(classification_report(y_test, test_predictions_optimal, zero_division=0))
sleep(10)


# =============================================================================
# 6. Бутстрап для доверительных интервалов метрик
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Бутстрап для доверительных интервалов метрик')
print('=' * 80)
sleep(2)

# Объединение тестовых данных
x_y_test = x_test.copy(deep=True)
x_y_test['satisfaction'] = y_test.values

boot_accuracies = []
boot_precisions = []
boot_recalls = []
boot_f1_scores = []

n_bootstraps = 1000

print(f'\n\nВыполняется бутстрап ({n_bootstraps} итераций)...')
for i in range(n_bootstraps):
    if (i + 1) % 100 == 0:
        print(f'Завершено итераций: {i + 1}/{n_bootstraps}')

    # Создание бутстрап-выборки
    x_y_test_boot = x_y_test.sample(len(x_y_test), replace=True)
    x_test_boot = x_y_test_boot.drop(columns='satisfaction')
    y_test_boot = x_y_test_boot['satisfaction']

    # Предсказания модели с оптимальным порогом
    predicted_probas = model.predict_proba(x_test_boot)
    y_pred = predicted_probas[:, 1] >= best_threshold

    # Вычисление метрик качества
    boot_accuracies.append(accuracy_score(y_test_boot, y_pred))
    boot_precisions.append(precision_score(y_test_boot, y_pred,
                                           zero_division=0))
    boot_recalls.append(recall_score(y_test_boot, y_pred, zero_division=0))
    boot_f1_scores.append(f1_score(y_test_boot, y_pred, zero_division=0))

def calculate_confidence_interval(metric_values):
    """Вычисляет доверительные интервалы (95%)"""
    sorted_metrics = np.sort(metric_values)
    lower_bound = sorted_metrics[int(0.025 * len(sorted_metrics))]
    upper_bound = sorted_metrics[int(0.975 * len(sorted_metrics))]
    return lower_bound, upper_bound

accuracy_ci = calculate_confidence_interval(boot_accuracies)
precision_ci = calculate_confidence_interval(boot_precisions)
recall_ci = calculate_confidence_interval(boot_recalls)
f1_ci = calculate_confidence_interval(boot_f1_scores)

# Вывод доверительных интервалов
print('\n\nДоверительные интервалы метрик (бутстрап):')
print('Accuracy:')
print(f'Среднее значение интервала: {test_accuracy}')
print(f'Интервал: [{accuracy_ci[0]}, {accuracy_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {
    accuracy_ci[1] - accuracy_ci[0]})')
sleep(10)

print('\nPrecision:')
print(f'Среднее значение интервала: {test_precision}')
print(f'Интервал: [{precision_ci[0]}, {precision_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {
    precision_ci[1] - precision_ci[0]})')
sleep(10)

print('\nRecall:')
print(f'Среднее значение интервала: {test_recall}')
print(f'Интервал: [{recall_ci[0]}, {recall_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {
    recall_ci[1] - recall_ci[0]})')
sleep(10)

print('\nF1:')
print(f'Среднее значение интервала: {test_f1}')
print(f'Интервал: [{f1_ci[0]}, {f1_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {f1_ci[1] - f1_ci[0]})')
sleep(10)

print('\n\nВыполняется анализ графика...')

# Визуализация распределения метрик
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics = [boot_accuracies, boot_precisions, boot_recalls, boot_f1_scores]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
colors = ['blue', 'green', 'red', 'purple']

for i, (ax, metric, name, color) in enumerate(zip(
        axes.flat, metrics, metric_names, colors)):
    sns.histplot(metric, ax=ax, color=color, kde=True)
    ax.set_title(f'Распределение метрики "{name}"')
    ax.set_xlabel(name)
    ax.set_ylabel('Частота')

plt.tight_layout()
plt.show()

sleep(2)


# =============================================================================
# 7. Интерпретация коэффициентов модели
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Интерпретация коэффициентов модели')
print('=' * 80)
sleep(2)

# Получение коэффициентов и их значимости
coefficients = pd.DataFrame({
    'Признак': x_train.columns,
    'Коэффициент': model.coef_[0],
    'Exp(Коэффициент)': np.exp(model.coef_[0]),
    'Влияние на шансы': [
        'Увеличивает' if coef > 0 else 'Уменьшает' for coef in model.coef_[0]]
})

# Сортировка по абсолютному значению коэффициента
coefficients_sorted = coefficients.reindex(
    coefficients['Коэффициент'].abs().sort_values(ascending=False).index)

print('\n\nКоэффициенты модели (отсортированы по влиянию):')
print(coefficients_sorted.to_string(index=False))
sleep(10)

print('\n\nВыполняется анализ графика...')

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
colors = ['red' if coef < 0 else 'blue'
          for coef in coefficients_sorted['Коэффициент']]
plt.barh(coefficients_sorted['Признак'],
         coefficients_sorted['Коэффициент'], color=colors)
plt.xlabel('Значение коэффициента')
plt.title('Важность признаков в модели логистической регрессии')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

sleep(2)

print('\n\nОбучение и анализ завершены!\n\n')

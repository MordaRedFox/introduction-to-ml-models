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
# 1. Загрузка и подготовка данных
# =============================================================================
print('=' * 80)
print('1. Загрузка и подготовка данных')
print('=' * 80)
sleep(2)

# Чтение данных
x_train = pd.read_csv('data_decision_tree/x_train_data.csv')
y_train = pd.read_csv('data_decision_tree/y_train_data.csv')
x_test = pd.read_csv('data_decision_tree/x_test_data.csv')
y_test = pd.read_csv('data_decision_tree/y_test_data.csv')

# Преобразование из 2D массива в 1D для корректности
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f'\n\nРазмер обучающей выборки: {x_train.shape}')
print(f'Размер тестовой выборки: {x_test.shape}')
sleep(5)

# Определение размера изображения на основе отобранных пикселей
n_pixels = x_train.shape[1]
# Ближайший квадратный размер
img_size = int(np.sqrt(n_pixels))
print(f'\n\nПредполагаемый размер изображения: {img_size}x{img_size} = '
      f'{img_size ** 2} пикселей')
print(f'Фактическое количество пикселей: {n_pixels} (из 784 возможных)')
sleep(5)

# Разделение данных на обучающую и тестовую выборки (80%:20%)
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=52)

print(f'\n\nРазмер обучающей выборки: {x_train_split.shape}')
print(f'Размер валидационной выборки: {x_val.shape}')
sleep(5)

def print_image(pixels, ax=None, title=None, original_size=28):
    """Визуализирует изображение из массива пикселей с учетом
    уменьшенной размерности"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    # Создание массива нулей размером 28x28 = 784 пикселя (для изображения)
    full_image = np.zeros(original_size ** 2)

    # Заполнение данными только тех пикселей, которые есть в итоговых данных
    # (формат имени пикселей в данных - "pixelX", где X - число)
    pixel_indices = []
    for col in x_train.columns:
        if col.startswith('pixel'):
            try:
                # Извлечение номера пикселя из названия
                idx = int(col[5:])
                pixel_indices.append(idx)
            except ValueError:
                continue

    # Сортировка индексов и заполнение значений
    pixel_indices.sort()
    for i, pixel_idx in enumerate(pixel_indices[:len(pixels)]):
        if pixel_idx < len(full_image):
            full_image[pixel_idx] = pixels[i]

    # Преобразование линейного массива в матрицу 28x28
    image = full_image.reshape(original_size, original_size)
    # Настройка внешнего вида
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return ax


# =============================================================================
# 2. Создание и обучение моделей
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Создание и обучение моделей')
print('=' * 80)
sleep(2)

# Словарь всех моделей и результатов их работы
models_results = {}

# Деревья принятия решений с разными гиперпараметрами
min_samples_leaf_values = [1, 3, 5, 10]

print('\n\nОбучение деревьев принятия решений...')
for min_leaf in min_samples_leaf_values:
    print(f'\nОбучение DecisionTree (min_samples_leaf={min_leaf})...')
    print(f'Название модели: DT_leaf_{min_leaf}')

    # Создание и обучение модели
    model_dt = DecisionTreeClassifier(min_samples_leaf=min_leaf,
                                      criterion='gini', random_state=52)
    model_dt.fit(x_train_split, y_train_split)

    # Предсказания модели
    train_pred = model_dt.predict(x_train_split)
    val_pred = model_dt.predict(x_val)

    # Точность модели
    train_accuracy = accuracy_score(y_train_split, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    # Сохранение данных в словаре всех моделей
    models_results[f'DT_leaf_{min_leaf}'] = {
        'model': model_dt,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

    print(f'Точность модели на обучающей выборке: {train_accuracy}')
    print(f'Точность модели на валидационной выборке: {val_accuracy}')
    print(f'Разница точностей: {train_accuracy - val_accuracy}')
    sleep(10)

# Случайный лес с разными гиперпараметрами
n_estimators_values = [100, 300, 500, 1000]

print('\n\nОбучение случайного леса...')
for n_est in n_estimators_values:
    print(f'\nОбучение RandomForest (n_estimators={n_est})...')
    print(f'Название модели: RF_est_{n_est}')

    # Создание и обучение модели
    model_rf = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=3,
                                      max_features='sqrt', criterion='gini',
                                      random_state=52, n_jobs=-1)
    model_rf.fit(x_train_split, y_train_split)

    # Предсказания модели
    train_pred = model_rf.predict(x_train_split)
    val_pred = model_rf.predict(x_val)

    # Точность модели
    train_accuracy = accuracy_score(y_train_split, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    # Сохранение данных в словаре всех моделей
    models_results[f'RF_est_{n_est}'] = {
        'model': model_rf,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

    print(f'Точность модели на обучающей выборке: {train_accuracy}')
    print(f'Точность модели на валидационной выборке: {val_accuracy}')
    print(f'Разница точностей: {train_accuracy - val_accuracy}')
    sleep(10)


# =============================================================================
# 3. Выбор лучшей модели для дальнейшего анализа
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Выбор лучшей модели для дальнейшего анализа')
print('=' * 80)
sleep(2)

best_model_name = max(models_results.keys(),
                      key=lambda x: models_results[x]['val_accuracy'])
best_model = models_results[best_model_name]['model']

print(f'\n\nЛучшая модель: {best_model_name}')
print(f'Точность на валидационной выборке: {models_results[
    best_model_name]['val_accuracy']}')
sleep(5)

# Визуализация сравнения моделей
sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 6))
models_names = list(models_results.keys())
list_train_accuracy = [models_results[name]['train_accuracy']
                       for name in models_names]
list_val_accuracy = [models_results[name]['val_accuracy']
                     for name in models_names]

print('\n\nВыполняется анализ графика...')
x_pos = np.arange(len(models_names))
width = 0.35
plt.bar(x_pos - width / 2, list_train_accuracy, width,
        label='Обучающая выборка', alpha=0.7)
plt.bar(x_pos + width / 2, list_val_accuracy, width,
        label='Валидационная выборка', alpha=0.7)

plt.xlabel('Модели')
plt.ylabel('Точности')
plt.title('Сравнение точности моделей')
plt.xticks(x_pos, models_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
sleep(5)


# =============================================================================
# 4. Оценка качества лучшей модели на валидационной выборке
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Оценка качества лучшей модели на валидационной выборке')
print('=' * 80)
sleep(2)

best_val_pred = best_model.predict(x_val)

# Метрики качества модели
best_val_accuracy = accuracy_score(y_val, best_val_pred)
best_val_precision = precision_score(y_val, best_val_pred,
                                     average='weighted', zero_division=0)
best_val_recall = recall_score(y_val, best_val_pred,
                               average='weighted', zero_division=0)
best_val_f1 = f1_score(y_val, best_val_pred,
                       average='weighted', zero_division=0)

print(f'\n\nМетрики качества модели {best_model_name} на '
      f'валидационной выборке:')
print(f'Accuracy: {best_val_accuracy}')
print(f'Precision: {best_val_precision}')
print(f'Recall: {best_val_recall}')
print(f'F1-score: {best_val_f1}')
sleep(10)

# Визуализация матрицы ошибок
print('\n\nВыполняется анализ графика...')
cm = confusion_matrix(y_val, best_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Матрица ошибок на валидационной выборке')
plt.xlabel('Предсказанный класс')
plt.ylabel('Фактический класс')
plt.show()


# =============================================================================
# 5. Анализ ошибок лучшей модели на валидационной выборке
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Анализ ошибок лучшей модели на валидационной выборке')
print('=' * 80)
sleep(2)

error_mask = best_val_pred != y_val
x_errors =  x_val[error_mask]
y_errors_true = y_val[error_mask]
y_errors_pred = best_val_pred[error_mask]

print(f'\n\nКоличество ошибок: {len(x_errors)}')
print(f'Доля ошибок: {len(x_errors) / len(y_val):.4f}')
sleep(10)

# Анализ самых частых ошибок
error_counts = pd.DataFrame({
    'true': y_errors_true,
    'pred': y_errors_pred
}).value_counts().head(10).reset_index()
error_counts.columns = ['Истинный класс', 'Предсказанный класс', 'Количество']

print(f'\n\nСамые частые ошибки:')
print(error_counts.to_string(index=False))
sleep(10)

# Визуализация примеров ошибок
print('\n\nВизуализация самых частых ошибок:')
if len(error_counts) > 0:
    # Выбираются 5 самых частых ошибок
    common_errors = error_counts.head(5)

    for i in range(len(common_errors)):
        true_class = common_errors.iloc[i]['Истинный класс']
        pred_class = common_errors.iloc[i]['Предсказанный класс']
        count = common_errors.iloc[i]['Количество']

        print(f'Визуализация ошибок {true_class} -> {pred_class} '
              f'(количество: {count})')

        # Нахождение примеров этой ошибки
        error_indices = np.where(
            (y_errors_true == true_class) & (y_errors_pred == pred_class))[0]

        # Выбор не более 6 примеров ошибки
        examples_to_show = min(6, len(error_indices))

        # Отдельный график для каждой ошибки
        n_cols = 3
        n_rows = (examples_to_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        # Преобразование axes в массив для единообразия,
        # если есть только одна строка
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        plot_idx = 0

        for j in range(examples_to_show):
            idx = error_indices[j]
            actual_idx = x_errors.index[idx]
            pixels = x_val.loc[actual_idx].values

            # Вычисление координат subplot
            row = plot_idx // n_cols
            col = plot_idx % n_cols

            ax = print_image(pixels=pixels, ax=axes[row, col],
                             title=f'Пример {j+1}')
            plot_idx += 1

        # Скрытие неиспользованных subplots
        for k in range(plot_idx, n_rows * n_cols):
            row = k // n_cols
            col = k % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(f'Ошибка классификации: {true_class} → {pred_class}')
        plt.tight_layout()
        plt.show()
        sleep(2)
else:
    print('Нет ошибок для визуализации!')
sleep(5)


# =============================================================================
# 6. Оценка качества лучшей модели на тестовой выборке
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Оценка качества лучшей модели на тестовой выборке')
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

print(f'\n\nМетрики качества модели {best_model_name} на тестовой выборке:')
print(f'Accuracy: {best_test_accuracy}')
print(f'Precision: {best_test_precision}')
print(f'Recall: {best_test_recall}')
print(f'F1-score: {best_test_f1}')
sleep(10)

print('\n\nОтчёт о классификации:')
print(classification_report(y_test, best_test_pred, zero_division=0))
sleep(10)


# =============================================================================
# 7. Бутстрап для доверительных интервалов метрик качества
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Бутстрап для доверительных интервалов метрик качества')
print('=' * 80)
sleep(2)

# Объединение тестовых данных
x_y_test = x_test.copy(deep=True)
x_y_test['label'] = y_test

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
    x_test_boot = x_y_test_boot.drop(columns='label')
    y_test_boot = x_y_test_boot['label']

    # Предсказания модели
    y_pred = best_model.predict(x_test_boot)

    # Вычисление метрик качества
    boot_accuracies.append(accuracy_score(y_test_boot, y_pred))
    boot_precisions.append(precision_score(y_test_boot, y_pred,
                                         average='weighted', zero_division=0))
    boot_recalls.append(recall_score(y_test_boot, y_pred,
                                   average='weighted', zero_division=0))
    boot_f1_scores.append(f1_score(y_test_boot, y_pred,
                                 average='weighted', zero_division=0))

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
print(f'Среднее значение интервала: {best_test_accuracy}')
print(f'Интервал: [{accuracy_ci[0]}, {accuracy_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {
    accuracy_ci[1] - accuracy_ci[0]})')
sleep(10)

print('\nPrecision:')
print(f'Среднее значение интервала: {best_test_precision}')
print(f'Интервал: [{precision_ci[0]}, {precision_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {
    precision_ci[1] - precision_ci[0]})')
sleep(10)

print('\nRecall:')
print(f'Среднее значение интервала: {best_test_recall}')
print(f'Интервал: [{recall_ci[0]}, {recall_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {recall_ci[1] - recall_ci[0]})')
sleep(10)

print('\nF1:')
print(f'Среднее значение интервала: {best_test_f1}')
print(f'Интервал: [{f1_ci[0]}, {f1_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {f1_ci[1] - f1_ci[0]})')
sleep(10)

# Визуализация распределения метрик
print('\n\nВизуализация распределения метрик...')
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
sleep(5)


# =============================================================================
# 8. Интерпретация лучшей модели
# =============================================================================
print('\n\n' + '=' * 80)
print('8. Интерпретация лучшей модели')
print('=' * 80)
sleep(2)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_

    importance_df = pd.DataFrame({
        'pixel': x_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print('\n\nТоп 10 самых важных пикселей:')
    print(importance_df.head(10))
    sleep(10)

    # Визуализация важных пикселей на изображении 28x28
    plt.figure(figsize=(12, 10))

    # Создание карты важности пикселей
    importance_map = np.zeros(28 * 28)

    # Заполнение важных пикселей
    for i, (pixel_name, importance) in enumerate(zip(
                x_train.columns, feature_importance)):
        if pixel_name.startswith('pixel'):
            try:
                pixel_idx = int(pixel_name[5:])
                if pixel_idx < len(importance_map):
                    importance_map[pixel_idx] = importance
            except ValueError:
                continue

    print('\n\nВыполняется анализ графика...')

    # Визуализация карты важности пикселей
    plt.imshow(importance_map.reshape(28, 28), cmap='hot',
               interpolation='nearest')
    plt.colorbar(label='Относительная важность')
    plt.title('Карта важности пикселей для классификации')
    plt.axis('off')
    plt.show()

print('\n\nОбучение и анализ завершены!\n\n')

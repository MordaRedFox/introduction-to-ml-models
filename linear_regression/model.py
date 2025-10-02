from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample


# =============================================================================
# 1. Создание и обучение модели
# =============================================================================
print('=' * 80)
print('1. Создание и обучение модели')
print('=' * 80)
sleep(2)

# Чтение данных
x_train = pd.read_csv('linear_regression_data/x_train_data.csv')
y_train = pd.read_csv('linear_regression_data/y_train_data.csv')
x_test = pd.read_csv('linear_regression_data/x_test_data.csv')
y_test = pd.read_csv('linear_regression_data/y_test_data.csv')

print(f'\n\nРазмер обучающей выборки: {x_train.shape}')
print(f'Размер тестовой выборки: {x_test.shape}')
sleep(5)

# Создание и обучение модели
model = LinearRegression()
model.fit(x_train, y_train)

# Предсказания
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# =============================================================================
# 2. Метрики качеста модели
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Метрики качеста модели')
print('=' * 80)
sleep(2)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
retraining = r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print('\n\nМетрики качества модели:')
print(f'R² на обучающей выборке: {r2_train}')
print(f'R² на тестовой выборке: {r2_test}')
print(f'Переобучение (разница R²): {retraining}')
print(f'MAE на тестовой выборке: {mae}')
print(f'MSE на тестовой выборке: {mse}')
print(f'RMSE на тестовой выборке: {rmse}')
sleep(10)


# =============================================================================
# 3. Бутстрап для доверительных интервалов метрик качества
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Бутстрап для доверительных интервалов метрик качества')
print('=' * 80)
sleep(2)

# Параметры бутстрапа
n_bootstraps = 1000
confidence_level = 0.95

# Массивы для хранения метрик на бутстрап-выборках
r2_scores = []
mae_scores = []
rmse_scores = []

print(f'\n\nВыполняется бутстрап ({n_bootstraps} итераций)...')
for i in range(n_bootstraps):
    if (i + 1) % 100 == 0:
        print(f'Завершено итераций: {i + 1}/{n_bootstraps}')

    # Создание бутстрап-выборки (с повторениями)
    X_boot, y_boot = resample(x_test, y_test, random_state=i)

    # Предсказание на бутстрап-выборке
    y_pred_boot = model.predict(X_boot)

    # Вычисление метрик
    r2_scores.append(r2_score(y_boot, y_pred_boot))
    mae_scores.append(mean_absolute_error(y_boot, y_pred_boot))
    rmse_scores.append(np.sqrt(mean_squared_error(y_boot, y_pred_boot)))

def calculate_confidence_interval(scores, confidence=0.95):
    """Вычисляет доверительный интервал для массива scores"""
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper

# Доверительные интервалы для каждой метрики
r2_ci = calculate_confidence_interval(r2_scores, confidence_level)
mae_ci = calculate_confidence_interval(mae_scores, confidence_level)
rmse_ci = calculate_confidence_interval(rmse_scores, confidence_level)

# Вывод результатов
print('\n\nДоверительные интервалы метрик (бутстрап):')
print('R²:')
print(f'Среднее значение интервала: {r2_test}')
print(f'Интервал: [{mae_ci[0]}, {mae_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {mae_ci[1] - mae_ci[0]})')
sleep(10)

print('\nMAE:')
print(f'Среднее значение интервала: {mae}')
print(f'Интервал: [{mae_ci[0]}, {mae_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {mae_ci[1] - mae_ci[0]})')
sleep(10)

print('\nRMSE:')
print(f'Среднее значение интервала: {rmse}')
print(f'Интервал: [{rmse_ci[0]}, {rmse_ci[1]}]')
print(f'Описание интервала: (95% ДИ, ширина: {rmse_ci[1] - rmse_ci[0]})')
sleep(10)

# Анализ практической полезности модели
print('\n\nАнализ практической полезности модели:')
print(f'В худшем случае (нижняя граница ДИ):')
print(f'  R² может быть всего {r2_ci[0]}')
print(f'  MAE может достигать {mae_ci[1]}')
print(f'  RMSE может достигать {rmse_ci[1]}')
sleep(10)


# =============================================================================
# 4. Интерпретация коэффициентов модели
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Интерпретация коэффициентов модели')
print('=' * 80)
sleep(2)

print('\n\nИнтерпретация коэффициентов:')
feature_importance = pd.DataFrame({
    'feature': x_train.columns,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
})

# Сортировка по важности
feature_importance = feature_importance.sort_values(
    'abs_coefficient', ascending=False)
for _, row in feature_importance.iterrows():
    sign = '+' if row['coefficient'] > 0 else '-'
    print(f'{row['feature']}: {sign}{abs(row['coefficient'])}')
sleep(10)


# =============================================================================
# 5. Визуализация предсказаний vs фактические значения
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Визуализация предсказаний vs фактические значения')
print('=' * 80)
sleep(2)

print('\n\nВыполняется анализ графиков...')

sns.set(rc={'figure.figsize': (11.7, 8.27)})
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказания vs Фактические значения')

plt.subplot(1, 2, 2)
residuals = y_test.values.flatten() - y_test_pred.flatten()
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Анализ остатков')

plt.tight_layout()
plt.show()

print('\n\nОбучение и анализ завершены!\n\n')

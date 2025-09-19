from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder


# =============================================================================
# 1. Загрузка и первичное исследование данных об удовлетворённости полётом
# =============================================================================
# Загрузка данных
flight_data = pd.read_csv('data/satisfaction_survey.csv')
sns.set(rc={'figure.figsize': (11.7, 8.27)})

# Разделение факторов от целевой переменной
feature = flight_data.drop(columns='satisfaction')
target = flight_data['satisfaction']

# Визуализация распределения целевой переменной
print('Выполняется анализ графика...')
plt.figure(figsize=(10, 6))
sns.countplot(x=target)
plt.title('Распределение удовлетворённости полётом')
plt.xlabel('Удовлетворённость полётом')
plt.ylabel('Количество пассажиров')
plt.show()


# =============================================================================
# 2. Формирование обучающей и тестовой выборки
# =============================================================================
# Разделение данных на обучающую и тестовую выборки (75%:25%)
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.25, random_state=52, stratify=target)

# Соединение данных для анализа
train_data = x_train.copy(deep=True)
train_data['satisfaction'] = y_train

print('\n\nСоздание обучающей и тестовой выборки...')
sleep(5)
print(f'Размер обучающей выборки: {len(x_train)}')
print(f'Размер тестовой выборки: {len(x_test)}')
sleep(5)


# =============================================================================
# 3. Обработка пропущенных значений
# =============================================================================
print('\n\nПроверка пропущенных значений в данных:')
print(train_data.isna().sum())
sleep(10)

# В факторе "Arrival Delay in Minutes" есть пропуски,
# заполним их медианным значением
arrival_delay_median = train_data['Arrival Delay in Minutes'].median()
train_data['Arrival Delay in Minutes'] = train_data[
    'Arrival Delay in Minutes'].fillna(arrival_delay_median)
x_test['Arrival Delay in Minutes'] = x_test[
    'Arrival Delay in Minutes'].fillna(arrival_delay_median)

print('\n\nЗаполняем пропуски медианными значениями...')
omissions = train_data.isna().sum().sum()
print(f'Общее количество пропусков после заполнения данных: {omissions}')
sleep(5)


# =============================================================================
# 4. Определение типов факторов
# =============================================================================
# Численные факторы
numeric_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes',
                    'Arrival Delay in Minutes']

# Бинарные категориальные факторы
binary_features = ['Gender', 'Customer Type', 'Type of Travel']

# Многоклассовые категориальные факторы
multiclass_features = ['Class']

# Факторы-оценки (0-5, обрабатываем как категориальные)
rating_features = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness']

# Все категориальные факторы
categorical_features = binary_features + multiclass_features + rating_features


# =============================================================================
# 5. Анализ взаимосвязей между численными факторами и удовлетворённостью
# =============================================================================
# Анализ корреляционных связей
numeric_data = train_data[numeric_features + ['satisfaction']].copy(deep=True)
correlation_matrix = numeric_data.corr()['satisfaction']

print('\n\nКорреляция численных факторов с удовлетворённостью:')
print(correlation_matrix.sort_values(ascending=False))
sleep(10)

# Анализ значимых корреляций
significant_correlations = correlation_matrix[
    (correlation_matrix >= 0.051) | (correlation_matrix <= -0.051)]
print('\n\nСтатистически значимые корреляции:')
print(significant_correlations.sort_values(ascending=False))
sleep(10)

# Визуализация распределений численных факторов
print('\n\nВыполняется анализ графиков...')
for feature in numeric_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x='satisfaction', y=feature, data=train_data)
    plt.title(f'Распределение фактора "{feature}" по удовлетворённости')

    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=feature, hue='satisfaction',
                 kde=True, alpha=0.6)
    plt.title(f'Гистограмма фактора "{feature}"')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. Анализ взаимосвязей между категориальными факторами и удовлетворённостью
# =============================================================================
def analyze_categorical_influence(data, categorical_cols,
                                  target_col='satisfaction'):
    """Анализирует влияние категориальных признаков на целевую переменную"""
    influence_metrics = []

    for feature in categorical_cols:
        # Агрегирует средние значения целевой переменной по категориям
        aggregated_data = data.groupby(feature, as_index=False).agg({
            target_col: ['mean', 'count']})
        aggregated_data.columns = [feature, 'satisfaction_mean', 'count']

        # Вычисляет метрики влияния
        target_values = aggregated_data['satisfaction_mean'].values
        influence_metrics.append({
            'factor': feature,
            'range': np.max(target_values) - np.min(target_values),
            'min_satisfaction': np.min(target_values),
            'max_satisfaction': np.max(target_values),
            'categories count': len(aggregated_data)
        })

    return pd.DataFrame(influence_metrics)

# Анализ влияния категориальных факторов
categorical_analysis = analyze_categorical_influence(
    train_data, categorical_features)
categorical_analysis = categorical_analysis.sort_values(
    'range', ascending=False)

print('\n\nАнализ влияния категориальных факторов на удовлетворённость:')
print(categorical_analysis.to_string(index=False))
sleep(20)

# Автоматический отбор наиболее значимых категориальных 
significant_categorical_features = categorical_analysis[
    categorical_analysis['range'] > 0.2
]['factor'].tolist()

print('\n\nЗначимые категориальные факторы:')
print(significant_categorical_features)
sleep(10)

# Детальный анализ значимых факторов
for feature in significant_categorical_features:
    print(f'\nДетальный анализ для фактора "{feature}":')
    analysis_table = train_data.groupby(
        [feature, 'satisfaction'], as_index=False).agg(
            {'Age': 'count'}).pivot(
                index=feature, columns='satisfaction', values='Age')
    analysis_table['satisfaction_rate'] = (
        analysis_table[True] / (analysis_table[True] + analysis_table[False]))
    print(analysis_table)
    sleep(10)


# =============================================================================
# 7. Анализ взаимосвязей между категориальными факторами
# =============================================================================
print('\n\nАнализ вероятных взаимосвязей между категориальными факторами:')

# Анализ взаимосвязи между услугами
service_features = ['Inflight wifi service', 'Food and drink',
                    'Seat comfort', 'Inflight entertainment']
for i in range(len(service_features)):
    for j in range(i+1, len(service_features)):
        col_x = service_features[i]
        col_y = service_features[j]

        print(f'\nВзаимосвязь между факторами "{col_x}" и "{col_y}":')
        cross_table = train_data.groupby([col_x, col_y], as_index=False).agg({
            'satisfaction': 'count'}).pivot(
                index=col_x, columns=col_y, values='satisfaction')
        print(cross_table.fillna(0).astype(int))
        sleep(10)

# Анализ взаимосвязи типа путешествия и класса
print('\nВзаимосвязь "Type of Travel" и "Class":')
travel_class_table = train_data.groupby(
    ['Type of Travel', 'Class'], as_index=False).agg(
        {'satisfaction': ['count', 'mean']})
print(travel_class_table)
sleep(10)


# =============================================================================
# 8. Отбор финального набора факторов
# =============================================================================
# Исключаем незначащие или сильно скоррелированные с другими факторы
excluded_features = ['Arrival Delay in Minutes', 'Gate location',
                     'Baggage handling']

# Итоговый набор факторов
final_numeric_features = [f for f in numeric_features
                          if f not in excluded_features]
final_categorical_features = [f for f in significant_categorical_features
                              if f not in excluded_features]

# Разделяем итоговые категориальные факторы по типам
final_binary_features = [f for f in final_categorical_features
                         if f in binary_features]
final_multiclass_features = [f for f in final_categorical_features
                             if f in multiclass_features]
final_rating_features = [f for f in final_categorical_features
                         if f in rating_features]

print('\n\nИтоговый набор численных факторов:')
print(final_numeric_features)
sleep(5)
print('\nИтоговый набор бинарных факторов:')
print(final_binary_features)
sleep(5)
print('\nИтоговый набор многоклассовых факторов:')
print(final_multiclass_features)
sleep(5)
print('\nИтоговый набор факторов-оценок:')
print(final_rating_features)
sleep(5)


# =============================================================================
# 9. Подготовка данных для моделирования
# =============================================================================
# Подготовка обучающей выборки
x_train_processed = x_train[
    final_numeric_features + final_categorical_features].copy(deep=True)
x_test_processed = x_test[
    final_numeric_features + final_categorical_features].copy(deep=True)

# Бинарное кодирование для всех категориальных факторов
binary_encoders = {}

# Кодирование бинарных факторов
for feature in final_binary_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Добавление закодированных колонок
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Удаление исходной колонки
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Кодирование многоклассовых факторов
for feature in final_multiclass_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Добавление закодированных колонок
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Удаление исходной колонки
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Кодирование факторов-оценок (0-5) как категориальных
for feature in final_rating_features:
    # Преобразование в строковый тип для корректного бинарного кодирования
    x_train_processed[feature] = x_train_processed[feature].astype(str)
    x_test_processed[feature] = x_test_processed[feature].astype(str)

    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Добавление закодированных колонок
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Удаление исходной колонки
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Определение колонок для масштабирования (только численные признаки)
numeric_cols = final_numeric_features
categorical_cols = [col for col in x_train_processed.columns
                   if col not in numeric_cols]

# Масштабирование только численных факторов
scaler = StandardScaler()
x_train_numeric_scaled = scaler.fit_transform(x_train_processed[numeric_cols])
x_test_numeric_scaled = scaler.transform(x_test_processed[numeric_cols])

# Создание финальных DataFrame с масштабированными численными и исходными
# категориальными признаками
x_train_final = pd.DataFrame(
    x_train_numeric_scaled,
    columns=numeric_cols,
    index=x_train_processed.index
)

x_test_final = pd.DataFrame(
    x_test_numeric_scaled,
    columns=numeric_cols,
    index=x_test_processed.index
)

# Добавление категориальных признаков без изменений
for col in categorical_cols:
    x_train_final[col] = x_train_processed[col].values
    x_test_final[col] = x_test_processed[col].values


# =============================================================================
# 10. Вывод подготовленных данных
# =============================================================================
print('\n\nЗакодированные данные для обучения (первые 5 строк):')
print(x_train_processed.head())
sleep(10)

print('\n\nМасштабированные данные для обучения (первые 5 строк):')
print(x_train_final.head())
sleep(10)

print('\n\nЗакодированные данные для тестирования (первые 5 строк):')
print(x_test_processed.head())
sleep(10)

print('\n\nМасштабированные данные для тестирования (первые 5 строк):')
print(x_test_final.head())
sleep(10)


# =============================================================================
# 11. Сохранение обучающей и тестовой выборок в CSV файл
# =============================================================================
x_train_final.to_csv('logistic_regression_data/x_train_data.csv', index=False)
y_train.to_csv('logistic_regression_data/y_train_data.csv', index=False)
x_test_final.to_csv('logistic_regression_data/x_test_data.csv', index=False)
y_test.to_csv('logistic_regression_data/y_test_data.csv', index=False)

print('\n\nСохранение данных прошло успешно!')

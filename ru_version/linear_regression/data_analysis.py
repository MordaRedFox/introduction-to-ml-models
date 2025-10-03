from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. Загрузка и первичное исследование образовательных данных
# =============================================================================
print('=' * 80)
print('1. Загрузка и первичное исследование образовательных данных')
print('=' * 80)
sleep(2)

# Чтение данных
sns.set(rc={'figure.figsize': (11.7, 8.27)})
student_dataset = pd.read_csv('data/student_mat.csv')

print(f'\n\nРазмерность данных: {student_dataset.shape}')
sleep(5)
print('\nПервые 5 строк данных:')
print(student_dataset.head())
sleep(10)

print('\n\nВыполняется анализ графиков...')

# Визуализация распределения итоговых оценок учеников
sns.histplot(student_dataset['G3'])
plt.title('Распределение итоговых оценок')
plt.xlabel('Итоговая оценка (G3)')
plt.ylabel('Количество учеников')
plt.show()
sleep(2)

# Наблюдаем много итоговых оценок, равных нулю. Посмотрим оценки за первое
# полугодие (G1) у таких людей
sns.histplot(student_dataset['G3'])
sns.histplot(student_dataset[student_dataset['G3'] == 0]['G1'])
plt.title('Распределение итоговых оценок и предварительных результатов')
plt.xlabel('Итоговая оценка (G3) и промежуточная оценка (G1)')
plt.ylabel('Количество учеников')
plt.show()
sleep(2)

# За первое полугодие у учеников с G3 == 0 результат G1 далёк от нуля. Скорее
# всего это выброс. Удалим аномальные данные
filtered_data = student_dataset[student_dataset['G3'] != 0]
sns.histplot(filtered_data['G3'])
plt.title('Распределение итоговых оценок после очистки данных')
plt.xlabel('Итоговая оценка (G3)')
plt.ylabel('Количество учеников')
plt.show()
sleep(2)


# =============================================================================
# 2. Формирование обучающей и тестовой выборок
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Формирование обучающей и тестовой выборок')
print('=' * 80)
sleep(2)

predictors = filtered_data.drop(columns=['G3'])
target_variable = filtered_data['G3']

# Так как данных слишком мало, в тестовой выборке будет всего 50 значений, а
# валидационная выборка не будет создана
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
    predictors, target_variable, test_size=50, random_state=52)

print(f'\n\nРазмер обучающей выборки: {x_train_split.shape[0]}')
print(f'Размер тестовой выборки: {x_test_split.shape[0]}')
sleep(10)


# =============================================================================
# 3. Анализ взаимосвязей между численными факторами и итоговой оценкой
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Анализ взаимосвязей между численными факторами и итоговой оценкой')
print('=' * 80)
sleep(2)

training_data_with_target = x_train_split.copy(deep=True)
training_data_with_target['G3'] = y_train_split
numeric_features = training_data_with_target.select_dtypes(
    include=['int64', 'float64'])
correlation_analysis = numeric_features.corr()['G3']

print('\n\nАнализ корреляционных связей:')
print(correlation_analysis)
sleep(10)

# Отбор статистически значимых корреляций
significant_correlations = correlation_analysis[
    (correlation_analysis >= 0.1) | (correlation_analysis <= -0.1)]

print('\n\nСтатистически значимые корреляции:')
print(significant_correlations.sort_values(ascending=False))
sleep(10)

print('\n\nВыполняется анализ графиков...')

# Анализ графиков для переменных с высокой корреляцией
selected_numeric_features = correlation_analysis.drop('G3').index.tolist()
for variable in selected_numeric_features:
    # Первый график зависимости
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=training_data_with_target[variable],
                    y=training_data_with_target['G3'])
    plt.title(f'Зависимость итоговой оценки от  фактора "{variable}"')
    plt.xlabel(variable)
    plt.ylabel('Итоговая оценка (G3)')
    plt.show()

    # Второй график зависимости
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=training_data_with_target[variable],
                y=training_data_with_target['G3'])
    plt.title(f'Зависимость итоговой оценки от  фактора "{variable}"')
    plt.xlabel(variable)
    plt.ylabel('Итоговая оценка (G3)')
    plt.show()
    sleep(2)

# Дополнительные анализы графиков

# Анализ возрастной динамики успеваемости
plt.figure(figsize=(12, 6))
sns.lineplot(x=training_data_with_target['age'],
             y=training_data_with_target['G3'], estimator='mean',
             errorbar=None)
plt.title('Динамика средней итоговой оценки по возрасту учащихся')
plt.xlabel('Возраст студента')
plt.ylabel('Средняя итоговая оценка')
plt.grid(True)
plt.show()
sleep(2)

# Комплексный анализ затраченного на учебу времени и его влияния на результаты
plt.figure(figsize=(10, 6))
sns.violinplot(x=training_data_with_target['studytime'],
               y=training_data_with_target['G3'])
plt.title('Влияние затраченного на учебу времени на итоговые результаты')
plt.xlabel('Время на учебу (часы в неделю)')
plt.ylabel('Итоговая оценка')
plt.show()
sleep(2)

print('\n\nОтобранные числовые признаки для моделирования:')
print(selected_numeric_features)
sleep(5)


# =============================================================================
# 4. Анализ взаимосвязей между категориальными факторами и итоговой оценкой
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Анализ взаимосвязей между категориальными факторами и итоговой '
      'оценкой')
print('=' * 80)
sleep(2)

categorical_features = [
    'school', 'gender', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

def analyze_categorical_influence(data, categorical_cols, target_col='G3'):
    """Анализирует влияние категориальных признаков на целевую переменную"""
    influence_metrics = []

    for feature in categorical_cols:
        # Агрегирование средних значений целевой переменной по категориям
        aggregated_data = data.groupby(
            feature, as_index=False).agg({target_col: 'mean'})

        # Вычисление статистических метрик
        target_values = aggregated_data[target_col].values
        influence_metrics.append({
            'sign': feature,
            'range': np.max(target_values) - np.min(target_values)
        })
    return pd.DataFrame(influence_metrics)

# Анализирование влияния категориальных факторов
categorical_analysis_df = analyze_categorical_influence(
    training_data_with_target, categorical_features)

# Сортировка по диапазону влияния (наибольшее влияние сверху)
categorical_analysis_df = categorical_analysis_df.sort_values(
    'range', ascending=False)

print('\n\nАнализ влияния категориальных факторов на итоговую оценку:')
print(categorical_analysis_df.to_string(index=False))
sleep(10)

# Автоматический отбор наиболее значимых признаков
significant_categorical_features = categorical_analysis_df[
    categorical_analysis_df['range'] > 1.0]['sign'].tolist()

print('\n\nЗначимые категориальные факторы:')
print(significant_categorical_features)
sleep(10)


# =============================================================================
# 5. Подготовка обучающей выборки
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Подготовка обучающей выборки')
print('=' * 80)
sleep(2)

# Кодирование значимых категориальных признаков
def encode_categorical_features(data, features_to_encode, target_col='G3'):
    """Выполняет target encoding для указанных категориальных признаков"""
    encoders = {}
    encoded_data = data.copy(deep=True)

    for feature in features_to_encode:
        encoder = TargetEncoder()
        encoder.fit(data[feature], data[target_col])
        encoded_data[f'{feature}_encoded'] = encoder.transform(data[feature])
        encoders[feature] = encoder
    return encoded_data, encoders

training_data_encoded, feature_encoders = encode_categorical_features(
    training_data_with_target, significant_categorical_features)

# Формирование финального набора признаков для моделирования
final_features = selected_numeric_features + [
    f'{feature}_encoded' for feature in significant_categorical_features
]
x_train_processed = training_data_encoded[final_features].copy(deep=True)

print('\n\nЗакодированные данные для обучения (первые 5 строк):')
print(x_train_processed.head())
sleep(10)

# Масштабирование данных
data_scaler = MinMaxScaler()
data_scaler.fit(x_train_processed)
x_train_scaled = data_scaler.transform(x_train_processed)

# Перевод полученных данных из матрицы обратно в таблицу
x_train_scaled_df = pd.DataFrame(
    x_train_scaled,
    columns=x_train_processed.columns,
    index=x_train_processed.index
)

print('\n\nМасштабированные данные для обучения (первые 5 строк):')
print(x_train_scaled_df.head())
sleep(10)


# =============================================================================
# 6. Подготовка тестовой выборки
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Подготовка тестовой выборки')
print('=' * 80)
sleep(2)

test_data_with_target = x_test_split.copy(deep=True)
test_data_with_target['G3'] = y_test_split

# Применение target encoding к категориальным признакам с использованием
# энкодеров, обученных на тренировочных данных
test_data_encoded = test_data_with_target.copy(deep=True)
for feature in significant_categorical_features:
    # Использование энкодера, обученного на тренировочных данных
    encoder = feature_encoders[feature]
    test_data_encoded[f'{feature}_encoded'] = encoder.transform(
        test_data_with_target[feature])

# Формирование финального набора признаков для тестовой выборки
x_test_processed = test_data_encoded[final_features].copy(deep=True)

print('\n\nЗакодированные данные для тестирования (первые 5 строк):')
print(x_test_processed.head())
sleep(10)

# Масштабирование тестовых данных с использованием scaler, обученного на
# тренировочных данных
x_test_scaled = data_scaler.transform(x_test_processed)

# Перевод полученных данных из матрицы обратно в таблицу
x_test_scaled_df = pd.DataFrame(
    x_test_scaled,
    columns=x_test_processed.columns,
    index=x_test_processed.index
)

print('\n\nМасштабированные данные для тестирования (первые 5 строк):')
print(x_test_scaled_df.head())
sleep(5)


# =============================================================================
# 7. Сохранение обучающей и тестовой выборок в CSV файл
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Сохранение обучающей и тестовой выборок в CSV файл')
print('=' * 80)
sleep(2)

x_train_scaled_df.to_csv(
    'data_linear_regression/x_train_data.csv', index=False)
y_train_split.to_csv(
    'data_linear_regression/y_train_data.csv', index=False)
x_test_scaled_df.to_csv(
    'data_linear_regression/x_test_data.csv', index=False)
y_test_split.to_csv(
    'data_linear_regression/y_test_data.csv', index=False)

print('\n\nСохранение данных прошло успешно!\n\n')

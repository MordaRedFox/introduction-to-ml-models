from time import sleep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# =============================================================================
# 1. Загрузка и первичное исследование данных MNIST
# =============================================================================
print('=' * 80)
print('1. Загрузка и первичное исследование данных MNIST')
print('=' * 80)
sleep(2)

# Чтение данных
sns.set(rc={'figure.figsize': (11.7, 8.27)})
data = pd.read_csv('data/images.csv')

print(f'\n\nРазмерность данных: {data.shape}')
sleep(5)
print('\nПервые 5 строк данных:')
print(data.head())
sleep(10)

# Проверка на наличие пропущенных значений
missing_values = data.isnull().sum().sum()
print(f'\nКоличество пропущенных значений: {missing_values}')


# =============================================================================
# 2. Анализ распределения классов (цифр 0-9)
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Анализ распределения классов (цифр 0-9)')
print('=' * 80)
sleep(2)

class_distribution = data['label'].value_counts().sort_index()
print('\n\nВыполняется анализ графика...')

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Распределение цифр в наборе данных MNIST')
plt.xlabel('Цифра')
plt.ylabel('Количество изображений')

# Добавление значений на столбцы
for i, v in enumerate(class_distribution.values):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 3. Разделение данных на обучающую и тестовую выборки
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Разделение данных на обучающую и тестовую выборки')
print('=' * 80)
sleep(2)

x = data.drop(columns='label')
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=52)

print(f'\n\nРазмер обучающей выборки: {x_train.shape[0]}')
print(f'Размер тестовой выборки: {x_test.shape[0]}')
sleep(10)


# =============================================================================
# 4. Визуализация примеров изображений
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Визуализация примеров изображений')
print('=' * 80)
sleep(2)

def plot_digit(image_flat, ax=None):
    """Визуализирует одно изображение цифры"""
    image = image_flat.reshape(28, 28).astype('uint8')

    if ax is None:
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    else:
        ax.imshow(image, cmap='gray')
        ax.axis('off')

# Создание объединенного DataFrame с признаками и метками
x_y_train = x_train.copy(deep=True)
x_y_train['label'] = y_train

# Визуализация 10 случайных примеров для каждого класса
print('\n\nВыполняется анализ графика...')
fig, axs = plt.subplots(10, 10, figsize=(12, 10), sharex=True, sharey=True)
group_n = 0
for _, group in x_y_train.groupby('label', as_index=False):
    random_numbers = group.sample(10).drop(columns='label')
    image_vects = [x.values for _, x in random_numbers.iterrows()]

    image_n = 0
    for image in image_vects:
        plot_digit(image, ax=axs[group_n][image_n])
        image_n += 1

    group_n += 1

plt.suptitle('Примеры изображений цифр из набора данных MNIST')
plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 5. Анализ статистических характеристик пикселей
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Анализ статистических характеристик пикселей')
print('=' * 80)
sleep(2)

# Статистика по пикселям
pixel_stats = x_train.describe()
print('\n\nСтатистика по пикселям:')
print(pixel_stats.loc[['mean', 'std', 'min', 'max']].T)
sleep(10)

# Анализ средних значений пикселей для каждой цифры
mean_digits = []
for digit in range(10):
    digit_mean = x_train[y_train == digit].mean().values.reshape(28, 28)
    mean_digits.append(digit_mean)

# Визуализация средних изображений для каждой цифры
print('\n\nВыполняется анализ графика...')
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for digit in range(10):
    axes[digit].imshow(mean_digits[digit], cmap='gray')
    axes[digit].set_title(f'Среднее для цифры {digit}')
    axes[digit].axis('off')

plt.suptitle('Средние изображения для каждой цифры')
plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 6. Анализ вариативности пикселей
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Анализ вариативности пикселей')
print('=' * 80)
sleep(2)

print('\n\nВыполняется анализ графиков...')

# Тепловая карта средних значений пикселей
plt.figure(figsize=(10, 8))
sns.heatmap(pixel_stats.loc['mean'].values.reshape(28, 28), cmap='viridis',
            square=True, cbar_kws={'label': 'Средняя интенсивность'})
plt.title('Тепловая карта средних значений пикселей')
plt.axis('off')
plt.show()
sleep(2)

# Тепловая карта стандартных отклонений пикселей
plt.figure(figsize=(10, 8))
sns.heatmap(pixel_stats.loc['std'].values.reshape(28, 28), cmap='plasma',
            square=True, cbar_kws={'label': 'Стандартное отклонение'})
plt.title('Тепловая карта вариативности пикселей')
plt.axis('off')
plt.show()
sleep(2)


# =============================================================================
# 7. Отбор информативных факторов (пикселей)
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Отбор информативных факторов (пикселей)')
print('=' * 80)
sleep(2)

# Анализ пикселей с низкой вариативностью
low_variance_pixels = (pixel_stats.loc['std'] < 5).sum()
percent_low_variance_pixels = low_variance_pixels / 784 * 100
print('\n\nОтбор информативных факторов:')
print(f'Количество пикселей с низкой вариативностью: {low_variance_pixels}')
print(f'Процент маловариативных пикселей: {percent_low_variance_pixels:.1f}%')
sleep(10)

# Отбор факторов на основе порога дисперсии
selector = VarianceThreshold(threshold=0.1)
selector.fit(x_train)

# Анализ отобранных факторов
selected_features = selector.get_feature_names_out()
print(f'\n\nВсего факторов: {x_train.shape[1]}')
print(f'Отобрано информативных факторов: {len(selected_features)}')
sleep(10)

# Применение отбора факторов
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

print('\nРазмерность после отбора факторов:')
print(f'Обучающая выборка: {x_train_selected.shape}')
print(f'Тестовая выборка: {x_test_selected.shape}')
sleep(10)

print('\n\nПервые 5 строк оптимизированных данных:')
print(x_train_selected.head())
sleep(10)


# =============================================================================
# 8. Сохранение обучающей и тестовой выборок в CSV файл
# =============================================================================
print('\n\n' + '=' * 80)
print('8. Сохранение обучающей и тестовой выборок в CSV файл')
print('=' * 80)
sleep(2)

x_train_selected.to_csv('data_decision_tree/x_train_data.csv', index=False)
x_test_selected.to_csv('data_decision_tree/x_test_data.csv', index=False)
y_train.to_csv('data_decision_tree/y_train_data.csv', index=False)
y_test.to_csv('data_decision_tree/y_test_data.csv', index=False)

print('\n\nСохранение данных прошло успешно!\n\n')

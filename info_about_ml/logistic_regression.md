# Информация о логистической регрессии

## Содержание
1. [Основные понятия](#основные-понятия)
2. [Математическая модель](#математическая-модель)
3. [Обучение модели](#обучение-модели)
4. [Классификация и пороговое значение](#классификация-и-пороговое-значение)
5. [Метрики качества](#метрики-качества)
6. [Практические аспекты](#практические-аспекты)
7. [Бутстрап и доверительные интервалы](#бутстрап-и-доверительные-интервалы)
8. [Интерпретация коэффициентов модели логистической регрессии](#интерпретация-коэффициентов-модели-логистической-регрессии)

---

## Основные понятия

### Что такое логистическая регрессия?
Логистическая регрессия — статистический метод моделирования вероятности принадлежности объекта к одному из двух классов.

### Особенности метода
- Решает задачу **бинарной классификации**
- Предсказывает **вероятность** принадлежности к положительному классу
- Использует **сигмоидную функцию** для преобразования линейной комбинации в вероятность

### Геометрическая интерпретация
Модель находит разделяющую гиперплоскость в пространстве признаков. Расстояние от точки до этой гиперплоскости определяет "уверенность" модели в предсказании.

### Преимущества логистической регрессии
- **Интерпретируемость**: коэффициенты показывают направление и силу влияния признаков
- **Быстрота**: быстрое обучение и предсказание
- **Вероятности**: предсказание вероятностей, а не только классов
- **Устойчивость**: хорошая работа с зашумленными данными

### Ограничения
- **Линейность**: предполагает линейную разделимость классов
- **Предобработка**: требует тщательной подготовки данных
- **Выбросы**: чувствительна к выбросам в данных
- **Сложные зависимости**: не подходит для сложных нелинейных зависимостей

---

## Математическая модель

### Основная формула
Вероятность принадлежности к положительному классу:

**`p̂(x) = σ(⟨w,x⟩) = 1 / (1 + e^(-⟨w,x⟩))`**

где:
- `⟨w,x⟩ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ` — линейная комбинация
- `σ(z)` — сигмоидная функция
- `w` — вектор весов модели
- `x` — вектор признаков объекта

### Логит-функция
Преобразование вероятности в шансы (odds ratio):

**`logit(p) = ln(p / (1-p)) = ⟨w,x⟩`**

### Интерпретация вероятностей
- `p̂(x) > 0.5` — объект вероятнее относится к положительному классу
- `p̂(x) < 0.5` — объект вероятнее относится к отрицательному классу
- `p̂(x) = 0.5` — неопределенность классификации

---

## Обучение модели

### Функция правдоподобия
Максимизация правдопоподобия данных:

**`L(w) = ∏ p̂(yᵢ|xᵢ)`**

где:
**`p̂(yᵢ|xᵢ) = p̂(xᵢ)^( (1+yᵢ)/2 ) * (1-p̂(xᵢ))^( (1-yᵢ)/2 )`**

### Лог-правдоподобие
Минимизация функции потерь:

**`loss(w) = -∑ ln(p̂(yᵢ|xᵢ)) = ∑ ln(1 + e^(-yᵢ⟨w,xᵢ⟩))`**

### Градиентный спуск
Частная производная по параметру wⱼ:

**`∂loss/∂wⱼ = -∑ yᵢxᵢⱼσ(-yᵢ⟨w,xᵢ⟩)`**

где:
- `xᵢⱼ` — значение j-го признака i-го объекта
- `σ(z)` — сигмоидная функция

---

## Классификация и пороговое значение

### Пороговое правило
**`ŷ = sign(p̂(x) - t)`**

где:
- `t` — пороговое значение (по умолчанию 0.5)
- `ŷ` — предсказанный класс (1 или -1)

### Выбор порога
- **По умолчанию**: t = 0.5
- **При дисбалансе классов**: подбор на валидационной выборке
- **Критерий**: максимизация accuracy или F1-score

### Стратегия выбора порога
Пример поиска оптимального порогового значения:

```python
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
```

---

## Метрики качества

### Матрица ошибок

<div align="center">

<table>
    <tr align="center">
        <th rowspan="2">Факт \ Прогноз</th>
        <th colspan="2">Предсказание</th>
    </tr>
    <tr align="center">
        <th>Положительный (1)</th>
        <th>Отрицательный (0)</th>
    </tr>
    <tr align="center">
        <td><strong>Положительный (1)</strong></td>
        <td>TP (True Positive)</td>
        <td>FN (False Negative)</td>
    </tr>
    <tr align="center">
        <td><strong>Отрицательный (0)</strong></td>
        <td>FP (False Positive)</td>
        <td>TN (True Negative)</td>
    </tr>
</table>

</div>

### Основные метрики

<div align="center">

<table>
    <tr align="center">
        <th>Метрика</th>
        <th>Формула</th>
        <th>Интерпретация</th>
    </tr>
    <tr align="center">
        <td><strong>Accuracy</strong></td>
        <td><code>(TP + TN) / (TP + TN + FP + FN)</code></td>
        <td>Общая доля верных предсказаний</td>
    </tr>
    <tr align="center">
        <td><strong>Precision</strong></td>
        <td><code>TP / (TP + FP)</code></td>
        <td>Точность положительных предсказаний</td>
    </tr>
    <tr align="center">
        <td><strong>Recall</strong></td>
        <td><code>TP / (TP + FN)</code></td>
        <td>Полнота положительных предсказаний</td>
    </tr>
    <tr align="center">
        <td><strong>F1-score</strong></td>
        <td><code>2 * (Precision * Recall) / (Precision + Recall)</code></td> <td>Гармоническое среднее precision и recall</td>
    </tr>
</table>

</div>

### Дополнительные метрики
- **ROC-AUC**: площадь под ROC-кривой
- **Precision-Recall AUC**: площадь под Precision-Recall кривой
- **Log Loss**: логарифмическая функция потерь

### Особенности метрик
- **Accuracy**: плохо работает при дисбалансе классов
- **Precision**: важна, когда стоимость FP высока (например, спам-фильтр)
- **Recall**: важна, когда стоимость FN высока (например, медицинская диагностика)
- **F1-score**: балансирует между precision и recall

---

## Практические аспекты

### Предобработка данных
- **Стандартизация**: обязательна для сходимости градиентного спуска
- **Работа с дисбалансом**: взвешивание классов, oversampling/undersampling
- **Отбор признаков**: исключение мультиколлинеарных признаков

### Регуляризация
- **L1-регуляризация (Lasso)**: отбор признаков, обнуление коэффициентов
- **L2-регуляризация (Ridge)**: уменьшение переобучения
- **Elastic Net**: комбинация L1 и L2 регуляризации

### Работа с категориальными признаками
Пример из кода:

```python
# Бинарное кодирование категориальных признаков
from category_encoders import BinaryEncoder

binary_encoders = {}
for feature in final_binary_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])
    
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]
    
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)
    binary_encoders[feature] = encoder
```

---

## Бутстрап и доверительные интервалы

### Метод бутстрапа
Статистический метод для оценки неопределенности метрик через многократное ресэмплирование данных.

### Реализация бутстрапа
Пример из кода:

```python
boot_accuracies = []
boot_precisions = []
boot_recalls = []
boot_f1_scores = []

n_bootstraps = 1000

for i in range(n_bootstraps):
    x_y_test_boot = x_y_test.sample(len(x_y_test), replace=True)
    x_test_boot = x_y_test_boot.drop(columns='satisfaction')
    y_test_boot = x_y_test_boot['satisfaction']
    
    predicted_probas = model.predict_proba(x_test_boot)
    y_pred = predicted_probas[:, 1] >= best_threshold
    
    boot_accuracies.append(accuracy_score(y_test_boot, y_pred))
    boot_precisions.append(precision_score(y_test_boot, y_pred, zero_division=0))
    boot_recalls.append(recall_score(y_test_boot, y_pred, zero_division=0))
    boot_f1_scores.append(f1_score(y_test_boot, y_pred, zero_division=0))
```

### Расчет доверительных интервалов
Пример из кода:

```python
def calculate_confidence_interval(metric_values):
    """Вычисляет доверительные интервалы (95%)"""
    sorted_metrics = np.sort(metric_values)
    lower_bound = sorted_metrics[int(0.025 * len(sorted_metrics))]
    upper_bound = sorted_metrics[int(0.975 * len(sorted_metrics))]
    return lower_bound, upper_bound
```

---

## Интерпретация коэффициентов модели логистической регрессии

### Шансы (Odds Ratio)
`OR = e^w` — во сколько раз изменятся шансы при изменении признака на 1 единицу

### Интерпретация коэффициентов
- **Положительный коэффициент**: увеличение признака увеличивает вероятность положительного класса
- **Отрицательный коэффициент**: увеличение признака уменьшает вероятность положительного класса
- **Величина коэффициента**: сила влияния на логарифм шансов

Пример из кода:

```python
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

print('\n\nИнтерпретация коэффициентов модели...')
print('\nКоэффициенты модели (отсортированы по влиянию):')
print(coefficients_sorted.to_string(index=False))
```

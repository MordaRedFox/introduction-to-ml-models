<div align="center">

# 🧠 Введение в машинное обучение

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Change language: [English](README.en.md)

</div>

---

## 🌟 Обзор проекта

Данный проект был разработан в рамках изучения основ машинного обучения. Его цель — не только практическая реализация, но и всесторонний анализ ключевых алгоритмов. В качестве учебного полигона в проекте представлены четыре фундаментальные модели машинного обучения:

<div align="center">

<table>
    <tr>
        <td align="center" width="25%">
            <h3 align="center">Линейная регрессия</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Логистическая регрессия</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Дерево принятия решений</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Случайный лес</h3>
        </td>
    </tr>
    <tr>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔮 Предсказание итоговых оценок учеников</li>
                <li>🔍 Анализ влияющих факторов</li>
                <li>📈 Точное численное предсказание</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>✈️ Анализ удовлетворённости авиаперелётами</li>
                <li>📜 Интерпретация коэффициентов модели</li>
                <li>🚩 Выявление ключевых факторов влияния</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔢 Распознавание рукописных цифр</li>
                <li>🔥 Простая и интерпретируемая модель</li>
                <li>⚡ Быстрое обучение и предсказание</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔢 Распознавание рукописных цифр</li>
                <li>🥇 Наилучшая точность предсказания</li>
                <li>🔄 Ансамбль из 1000 деревьев</li>
                <li>🔰 Устойчивость к переобучению</li>
            </ul>
        </td>
    </tr>
</table>

</div>

---

## 🎯 Цели и задачи

### 📚 Образовательные цели
- Понимание основ машинного обучения и его применения
- Практическая реализация ключевых алгоритмов ML
- Анализ данных и подготовка признаков для разных типов задач
- Оценка качества моделей с использованием современных метрик
- Интерпретация результатов для принятия бизнес-решений

### 🎓 Академическая ценность
Проект демонстрирует различные подходы к решению задач ML:
- **Регрессия** - предсказание численных значений
- **Классификация** - бинарная и многоклассовая
- **Анализ важности признаков** - интерпретация моделей
- **Ансамблевые методы** - улучшение точности предсказаний

---

## 📊 Используемые данные

<div align="center">

<table>
    <tr align="center">
        <th>Набор данных</th>
        <th>Тип задачи</th>
        <th>Размер</th>
        <th>Источник</th>
        <th>Описание</th>
    </tr>
    <tr align="center">
        <td><strong>Student Performance</strong></td>
        <td>Регрессия</td>
        <td>395 записей</td>
        <td>P. Cortez and A. Silva</td>
        <td>Данные об успеваемости учеников португальских школ</td>
    </tr>
    <tr align="center">
        <td><strong>Airline Satisfaction</strong></td>
        <td>Классификация</td>
        <td>130000 записей</td>
        <td>Анонимный опрос</td>
        <td>Данные об удовлетворённости авиаперелётами</td>
    </tr>
    <tr align="center">
        <td><strong>MNIST</strong></td>
        <td>Классификация</td>
        <td>70000 изображений</td>
        <td>Стандартный датасет</td>
        <td>Рукописные цифры 28x28 пикселей</td>
    </tr>
</table>

</div>

---

## 🛠 Технологический стек

<div align="center">

<table>
    <tr>
        <th>Компонент</th>
        <th>Версия</th>
        <th>Назначение</th>
        <th>Бейдж</th>
    </tr>
    <tr align="center">
        <td>Python</td>
        <td>3.10+</td>
        <td>Основной язык программирования</td>
        <td><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></td>
    </tr>
    <tr align="center">
        <td>Scikit-learn</td>
        <td>1.3+</td>
        <td>Библиотека машинного обучения</td>
        <td><img src="https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn"></td>
    </tr>
    <tr align="center">
        <td>Pandas</td>
        <td>2.0+</td>
        <td>Анализ и обработка данных</td>
        <td><img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas"></td>
    </tr>
    <tr align="center">
        <td>NumPy</td>
        <td>1.24+</td>
        <td>Научные вычисления</td>
        <td><img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy"></td>
    </tr>
    <tr align="center">
        <td>Matplotlib</td>
        <td>3.7+</td>
        <td>Визуализация данных</td>
        <td><img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?logo=python&logoColor=white" alt="Matplotlib"></td>
    </tr>
    <tr align="center">
        <td>Seaborn</td>
        <td>0.12+</td>
        <td>Статистическая визуализация</td>
        <td><img src="https://img.shields.io/badge/Seaborn-0.12%2B-3776AB?logo=python&logoColor=white" alt="Seaborn"></td>
    </tr>
    <tr align="center">
        <td>Jupyter</td>
        <td>1.0+</td>
        <td>Интерактивная разработка</td>
        <td><img src="https://img.shields.io/badge/Jupyter-1.0%2B-F37626?logo=jupyter&logoColor=white" alt="Jupyter"></td>
    </tr>
</table>

</div>

---

## 📁 Структура проекта

```bash
introduction-to-ml-models/
├── .venv/                          # Виртуальное окружение Python
├── data/                           # Исходные данные
│   ├── images.csv                  # MNIST датасет в CSV формате
│   ├── satisfaction_survey.csv     # Опрос удовлетворённости авиаперелётом
│   └── student_mat.csv             # Данные об успеваемости учеников
├── data_decision_tree/             # Подготовленные данные для деревьев
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── data_linear_regression/         # Подготовленные данные для линейной регрессии
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── data_logistic_regression/       # Подготовленные данные для логистической регрессии
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── en_version/                     # Английская версия проекта
│   ├── decision_tree/              # Реализация дерева принятия решений и случайного леса
│   │   ├── data_analysis.py        # Анализ данных
│   │   ├── decision_tree.ipynb     # Jupyter ноутбук с комплексным исследованием
│   │   ├── data_description.txt    # Описание данных
│   │   └── model.py                # Модель машинного обучения
│   ├── info_about_ml/              # Теоретические материалы
│   │   ├── decision_tree.md        # Материалы о дереве принятия решений и случайном лесе
│   │   ├── linear_regression.md    # Материалы о линейной регрессии
│   │   └── logistic_regression.md  # Материалы о логистической регрессии
│   ├── linear_regression/          # Реализация линейной регрессии
│   │   ├── data_analysis.py
│   │   ├── linear_regression.ipynb
│   │   ├── data_description.txt
│   │   └── model.py
│   └── logistic_regression/        # Реализация логистической регрессии
│       ├── data_analysis.py
│       ├── logistic_regression.ipynb
│       ├── data_description.txt
│       └── model.py
├── ru_version/                     # Русская версия проекта
│   ├── decision_tree/              # Реализация дерева принятия решений и случайного леса
│   │   ├── data_analysis.py        # Анализ данных
│   │   ├── decision_tree.ipynb     # Jupyter ноутбук с комплексным исследованием
│   │   ├── data_description.txt    # Описание данных
│   │   └── model.py                # Модель машинного обучения
│   ├── info_about_ml/              # Теоретические материалы
│   │   ├── decision_tree.md        # Материалы о дереве принятия решений и случайном лесе
│   │   ├── linear_regression.md    # Материалы о линейной регрессии
│   │   └── logistic_regression.md  # Материалы о логистической регрессии
│   ├── linear_regression/          # Реализация линейной регрессии
│   │   ├── data_analysis.py
│   │   ├── linear_regression.ipynb
│   │   ├── data_description.txt
│   │   └── model.py
│   └── logistic_regression/        # Реализация логистической регрессии
│       ├── data_analysis.py
│       ├── logistic_regression.ipynb
│       ├── data_description.txt
│       └── model.py
├── .gitignore                      # Файлы и директории, игнорируемые git
├── LICENSE                         # Лицензия проекта
├── README.md                       # Документация проекта на английском
├── README.ru.md                    # Документация проекта на русском
└── requirements.txt                # Зависимости проекта
```

---

## Быстрый старт

### 📋 Предварительные требования
- Python 3.10 или новее
- Рекомендуется использование виртуального окружения

### ⚙️ Установка и настройка
1. Клонируйте репозиторий проекта:

```bash
git clone https://github.com/MordaRedFox/introduction-to-ml-models.git
cd introduction-to-ml-models
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

### Использование отдельных моделей
Каждая модель находится в отдельной директории и содержит:
- `data_description.txt` - описание данных
- `data_analysis.py` - анализ данных
- `model.py` - реализация модели ML
- `*.ipynb` - интерактивный ноутбук с полным исследованием (анализ данных и модель)

---

## 📝 Ключевые результаты

### 📈 Линейная регрессия (Прогнозирование оценок)
- **Цель**: точное предсказание итоговой оценки ученика
- **Метрики**: R², MSE, MAE
- **Результат**: выявлены ключевые факторы влияния на успеваемость и создана модель с достаточно точными предсказаниями

### ✈️ Логистическая регрессия (Анализ удовлетворённости)
- **Цель**: интерпретация факторов удовлетворённости полётом
- **Метрики**: Accuracy, Precision, Recall, F1-score
- **Результат**: определены наиболее значимые параметры для улучшения сервиса

### 🔢 Дерево принятия решений и Случайный лес (Распознавание цифр)
- **Цель**: классификация рукописных цифр без свёрточных сетей
- **Метрики**: Accuracy, Precision, Recall, F1-score
- **Результат**: Случайный лес показал наивысшую точность классификации (более 95%)

### 🗂️ Образовательные материалы
Проект включает подробные теоретические материалы по каждому алгоритму с практическими примерами кода:
- 📖 **Линейная регрессия** - математические основы, метод наименьших квадратов, градиентный спуск, метрики качества (MSE, R²), регуляризация (Ridge, Lasso), проверка статистических предположений и анализ остатков
- 📖 **Логистическая регрессия** - сигмоидная функция, максимизация правдоподобия, интерпретация коэффициентов через шансы (Odds Ratio), матрица ошибок, подбор оптимального порога классификации и работа с дисбалансом классов
- 📖 **Дерево принятия решений и случайный лес** - алгоритмы построения деревьев, критерии разделения (Джини, энтропия), борьба с переобучением, бэггинг, оценка важности признаков и сравнение с линейными моделями
- 📖 **Машинное обучение** - типы ML задач, исследовательский анализ данных, подготовка признаков, подбор гиперпараметров (Grid Search, Random Search) и методы оценки качества моделей с доверительными интервалами

---

## ⚠️ Важное примечание
Этот проект был разработан начинающим программистом-самоучкой. Код может содержать:
- ❌ Ошибки и баги
- ⚡ Неоптимальные решения
- 🛡️ Недочёты в архитектуре

---

## 📩 Контакты
Я открыт для конструктивной критики и предложений по улучшению кода. Если вы нашли ошибку или знаете, как сделать что-то лучше - пожалуйста, свяжитесь со мной!

[![Telegram](https://img.shields.io/badge/-MordaRedFox-0088cc?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/MordaRedFox)
&nbsp;
[![Email](https://img.shields.io/badge/-mordaredfox@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mordaredfox@gmail.com)

> Машинное обучение — это не магия, это математика, код и много качественных данных

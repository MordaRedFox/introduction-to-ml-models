<div align="center">

# 🧠 Introduction to Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Сменить язык: [Русский](README.ru.md)

</div>

---

## ⚠️📢 Attention!
The English version is still in development

## 🌟 Project Overview

This project was developed as part of learning the fundamentals of machine learning. Its goal is not only practical implementation but also comprehensive analysis of key algorithms. As a training ground, the project presents four fundamental machine learning models:

<div align="center">

<table>
    <tr>
        <td align="center" width="25%">
            <h3 align="center">Linear Regression</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Logistic Regression</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Decision Tree</h3>
        </td>
        <td align="center" width="25%">
            <h3 align="center">Random Forest</h3>
        </td>
    </tr>
    <tr>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔮 Predicting students' final grades</li>
                <li>🔍 Analysis of influencing factors</li>
                <li>📈 Precise numerical prediction</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>✈️ Analysis of airline satisfaction</li>
                <li>📜 Interpretation of model coefficients</li>
                <li>🚩 Identification of key influencing factors</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔢 Handwritten digit recognition</li>
                <li>🔥 Simple and interpretable model</li>
                <li>⚡ Fast training and prediction</li>
            </ul>
        </td>
        <td align="left" valign="top" width="25%">
            <ul>
                <li>🔢 Handwritten digit recognition</li>
                <li>🥇 Best prediction accuracy</li>
                <li>🔄 Ensemble of 1000 trees</li>
                <li>🔰 Resistance to overfitting</li>
            </ul>
        </td>
    </tr>
</table>

</div>

---

## 🎯 Goals and Objectives

### 📚 Educational Goals
- Understanding the basics of machine learning and its applications
- Practical implementation of key ML algorithms
- Data analysis and feature preparation for different types of tasks
- Model evaluation using modern metrics
- Interpretation of results for business decisions

### 🎓 Academic Value
The project demonstrates various approaches to solving ML problems:
- **Regression** - predicting numerical values
- **Classification** - binary and multiclass
- **Feature importance analysis** - model interpretation
- **Ensemble methods** - improving prediction accuracy

---

## 📊 Used Data

<div align="center">

<table>
    <tr align="center">
        <th>Dataset</th>
        <th>Task Type</th>
        <th>Size</th>
        <th>Source</th>
        <th>Description</th>
    </tr>
    <tr align="center">
        <td><strong>Student Performance</strong></td>
        <td>Regression</td>
        <td>395 records</td>
        <td>P. Cortez and A. Silva</td>
        <td>Data on student performance in Portuguese schools</td>
    </tr>
    <tr align="center">
        <td><strong>Airline Satisfaction</strong></td>
        <td>Classification</td>
        <td>130000 records</td>
        <td>Anonymous survey</td>
        <td>Data on airline flight satisfaction</td>
    </tr>
    <tr align="center">
        <td><strong>MNIST</strong></td>
        <td>Classification</td>
        <td>70000 images</td>
        <td>Standard dataset</td>
        <td>Handwritten digits 28x28 pixels</td>
    </tr>
</table>

</div>

---

## 🛠 Technology Stack

<div align="center">

<table>
    <tr>
        <th>Component</th>
        <th>Version</th>
        <th>Purpose</th>
        <th>Badge</th>
    </tr>
    <tr align="center">
        <td>Python</td>
        <td>3.10+</td>
        <td>Main programming language</td>
        <td><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></td>
    </tr>
    <tr align="center">
        <td>Scikit-learn</td>
        <td>1.3+</td>
        <td>Machine learning library</td>
        <td><img src="https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn"></td>
    </tr>
    <tr align="center">
        <td>Pandas</td>
        <td>2.0+</td>
        <td>Data analysis and processing</td>
        <td><img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas"></td>
    </tr>
    <tr align="center">
        <td>NumPy</td>
        <td>1.24+</td>
        <td>Scientific computing</td>
        <td><img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy"></td>
    </tr>
    <tr align="center">
        <td>Matplotlib</td>
        <td>3.7+</td>
        <td>Data visualization</td>
        <td><img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?logo=python&logoColor=white" alt="Matplotlib"></td>
    </tr>
    <tr align="center">
        <td>Seaborn</td>
        <td>0.12+</td>
        <td>Statistical visualization</td>
        <td><img src="https://img.shields.io/badge/Seaborn-0.12%2B-3776AB?logo=python&logoColor=white" alt="Seaborn"></td>
    </tr>
    <tr align="center">
        <td>Jupyter</td>
        <td>1.0+</td>
        <td>Interactive development</td>
        <td><img src="https://img.shields.io/badge/Jupyter-1.0%2B-F37626?logo=jupyter&logoColor=white" alt="Jupyter"></td>
    </tr>
</table>

</div>

---

## 📁 Project Structure

```bash
introduction-to-ml-models/
├── .venv/                          # Python virtual environment
├── data/                           # Source data
│   ├── images.csv                  # MNIST dataset in CSV format
│   ├── satisfaction_survey.csv     # Airline satisfaction survey
│   └── student_mat.csv             # Student performance data
├── data_decision_tree/             # Prepared data for trees
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── data_linear_regression/         # Prepared data for linear regression
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── data_logistic_regression/       # Prepared data for logistic regression
│   ├── x_test_data.csv
│   ├── x_train_data.csv
│   ├── y_test_data.csv
│   └── y_train_data.csv
├── en_version/                     # English version of the project
│   ├── decision_tree/              # Decision tree and random forest implementation
│   │   ├── data_analysis.py        # Data analysis
│   │   ├── decision_tree.ipynb     # Jupyter notebook with comprehensive research
│   │   ├── data_description.txt    # Data description
│   │   └── model.py                # Machine learning model
│   ├── info_about_ml/              # Theoretical materials
│   │   ├── decision_tree.md        # Materials on decision tree and random forest
│   │   ├── linear_regression.md    # Materials on linear regression
│   │   └── logistic_regression.md  # Materials on logistic regression
│   ├── linear_regression/          # Linear regression implementation
│   │   ├── data_analysis.py
│   │   ├── linear_regression.ipynb
│   │   ├── data_description.txt
│   │   └── model.py
│   └── logistic_regression/        # Logistic regression implementation
│       ├── data_analysis.py
│       ├── logistic_regression.ipynb
│       ├── data_description.txt
│       └── model.py
├── ru_version/                     # Russian version of the project
│   ├── decision_tree/              # Decision tree and random forest implementation
│   │   ├── data_analysis.py        # Data analysis
│   │   ├── decision_tree.ipynb     # Jupyter notebook with comprehensive research
│   │   ├── data_description.txt    # Data description
│   │   └── model.py                # Machine learning model
│   ├── info_about_ml/              # Theoretical materials
│   │   ├── decision_tree.md        # Materials on decision tree and random forest
│   │   ├── linear_regression.md    # Materials on linear regression
│   │   └── logistic_regression.md  # Materials on logistic regression
│   ├── linear_regression/          # Linear regression implementation
│   │   ├── data_analysis.py
│   │   ├── linear_regression.ipynb
│   │   ├── data_description.txt
│   │   └── model.py
│   └── logistic_regression/        # Logistic regression implementation
│       ├── data_analysis.py
│       ├── logistic_regression.ipynb
│       ├── data_description.txt
│       └── model.py
├── .gitignore                      # Files and directories ignored by git
├── LICENSE                         # Project license
├── README.md                       # Project documentation in English
├── README.ru.md                    # Project documentation in Russian
└── requirements.txt                # Project dependencies
```

---

## Quick Start

### 📋 Prerequisites
- Python 3.10 or newer
- Using a virtual environment is recommended

### ⚙️ Installation and Setup
1. Clone the project repository:
```bash
git clone https://github.com/MordaRedFox/introduction-to-ml-models.git
cd introduction-to-ml-models
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Using Individual Models
Each model is located in a separate directory and contains:
- `data_description.txt` - data description
- `data_analysis.py` - data analysis
- `model.py` - ML model implementation
- `*.ipynb` - interactive notebook with full research (data analysis and model)

---

## 📝 Key Results

### 📈 Linear Regression (Grade Prediction)
- **Goal**: accurate prediction of student's final grade
- **Metrics**: R², MSE, MAE
- **Result**: key factors affecting academic performance were identified and a model with sufficiently accurate predictions was created

### ✈️ Logistic Regression (Satisfaction Analysis)
- **Goal**: interpretation of flight satisfaction factors
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Result**: the most significant parameters for service improvement were determined

### 🔢 Decision Tree and Random Forest (Digit Recognition)
- **Goal**: classification of handwritten digits without convolutional networks
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Result**: Random Forest showed the highest classification accuracy (over 95%)

### 🗂️ Educational Materials
The project includes detailed theoretical materials for each algorithm with practical code examples:
- 📖 **Linear Regression** - mathematical foundations, least squares method, gradient descent, quality metrics (MSE, R²), regularization (Ridge, Lasso), checking statistical assumptions and residual analysis
- 📖 **Logistic Regression** - sigmoid function, maximum likelihood estimation, coefficient interpretation through odds ratios, confusion matrix, optimal classification threshold selection and handling class imbalance
- 📖 **Decision Tree and Random Forest** - tree building algorithms, splitting criteria (Gini, entropy), combating overfitting, bagging, feature importance estimation and comparison with linear models
- 📖 **Machine Learning** - types of ML tasks, exploratory data analysis, feature preparation, hyperparameter tuning (Grid Search, Random Search) and model evaluation methods with confidence intervals

---

## ⚠️ Important Note
This project was developed by a beginner self-taught programmer. The code may contain:
- ❌ Errors and bugs
- ⚡ Suboptimal solutions
- 🛡️ Architectural shortcomings

---

## 📩 Contacts
I'm open to constructive criticism and suggestions for code improvement. If you found an error or know how to do something better - please contact me!

[![Telegram](https://img.shields.io/badge/-MordaRedFox-0088cc?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/MordaRedFox)
&nbsp;
[![Email](https://img.shields.io/badge/-mordaredfox@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mordaredfox@gmail.com)

> Machine learning is not magic, it's mathematics, code, and lots of quality data

from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. Loading and initial exploration of educational data
# =============================================================================
print('=' * 80)
print('1. Loading and initial exploration of educational data')
print('=' * 80)
sleep(2)

# Reading data
sns.set(rc={'figure.figsize': (11.7, 8.27)})
student_dataset = pd.read_csv('data/student_mat.csv')

print(f'\n\nData dimensions: {student_dataset.shape}')
sleep(5)
print('\nFirst 5 rows of data:')
print(student_dataset.head())
sleep(10)

print('\n\nPerforming graph analysis...')

# Visualization of student final grade distribution
sns.histplot(student_dataset['G3'])
plt.title('Final grade distribution')
plt.xlabel('Final grade (G3)')
plt.ylabel('Number of students')
plt.show()
sleep(2)

# We observe many final grades equal to zero. Let's look at the first semester
# grades (G1) for these people
sns.histplot(student_dataset['G3'])
sns.histplot(student_dataset[student_dataset['G3'] == 0]['G1'])
plt.title('Distribution of final grades and preliminary results')
plt.xlabel('Final grade (G3) and intermediate grade (G1)')
plt.ylabel('Number of students')
plt.show()
sleep(2)

# For the first semester, students with G3 == 0 have G1 results far from zero.
# Most likely these are outliers. Let's remove anomalous data
filtered_data = student_dataset[student_dataset['G3'] != 0]
sns.histplot(filtered_data['G3'])
plt.title('Final grade distribution after data cleaning')
plt.xlabel('Final grade (G3)')
plt.ylabel('Number of students')
plt.show()
sleep(2)


# =============================================================================
# 2. Formation of training and test sets
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Formation of training and test sets')
print('=' * 80)
sleep(2)

predictors = filtered_data.drop(columns=['G3'])
target_variable = filtered_data['G3']

# Since there is too little data, the test set will contain only 50 values, and
# a validation set will not be created
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
    predictors, target_variable, test_size=50, random_state=52)

print(f'\n\nTraining set size: {x_train_split.shape[0]}')
print(f'Test set size: {x_test_split.shape[0]}')
sleep(10)


# =============================================================================
# 3. Analysis of relationships between numerical factors and final grade
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Analysis of relationships between numerical factors and final grade')
print('=' * 80)
sleep(2)

training_data_with_target = x_train_split.copy(deep=True)
training_data_with_target['G3'] = y_train_split
numeric_features = training_data_with_target.select_dtypes(
    include=['int64', 'float64'])
correlation_analysis = numeric_features.corr()['G3']

print('\n\nCorrelation analysis:')
print(correlation_analysis)
sleep(10)

# Selection of statistically significant correlations
significant_correlations = correlation_analysis[
    (correlation_analysis >= 0.1) | (correlation_analysis <= -0.1)]

print('\n\nStatistically significant correlations:')
print(significant_correlations.sort_values(ascending=False))
sleep(10)

print('\n\nPerforming graph analysis...')

# Graph analysis for variables with high correlation
selected_numeric_features = correlation_analysis.drop('G3').index.tolist()
for variable in selected_numeric_features:
    # First dependency graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=training_data_with_target[variable],
                    y=training_data_with_target['G3'])
    plt.title(f'Dependency of final grade on factor "{variable}"')
    plt.xlabel(variable)
    plt.ylabel('Final grade (G3)')
    plt.show()

    # Second dependency graph
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=training_data_with_target[variable],
                y=training_data_with_target['G3'])
    plt.title(f'Dependency of final grade on factor "{variable}"')
    plt.xlabel(variable)
    plt.ylabel('Final grade (G3)')
    plt.show()
    sleep(2)

# Additional graph analyses

# Analysis of age dynamics of academic performance
plt.figure(figsize=(12, 6))
sns.lineplot(x=training_data_with_target['age'],
             y=training_data_with_target['G3'], estimator='mean',
             errorbar=None)
plt.title('Dynamics of average final grade by student age')
plt.xlabel('Student age')
plt.ylabel('Average final grade')
plt.grid(True)
plt.show()
sleep(2)

# Comprehensive analysis of study time and its impact on results
plt.figure(figsize=(10, 6))
sns.violinplot(x=training_data_with_target['studytime'],
               y=training_data_with_target['G3'])
plt.title('Impact of study time on final results')
plt.xlabel('Study time (hours per week)')
plt.ylabel('Final grade')
plt.show()
sleep(2)

print('\n\nSelected numerical features for modeling:')
print(selected_numeric_features)
sleep(5)


# =============================================================================
# 4. Analysis of relationships between categorical factors and final grade
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Analysis of relationships between categorical factors and final ' 
      'grade')
print('=' * 80)
sleep(2)

categorical_features = [
    'school', 'gender', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

def analyze_categorical_influence(data, categorical_cols, target_col='G3'):
    """Analyzes the influence of categorical features on the target variable"""
    influence_metrics = []

    for feature in categorical_cols:
        # Aggregation of mean target variable values by categories
        aggregated_data = data.groupby(
            feature, as_index=False).agg({target_col: 'mean'})

        # Calculation of statistical metrics
        target_values = aggregated_data[target_col].values
        influence_metrics.append({
            'feature': feature,
            'range': np.max(target_values) - np.min(target_values)
        })
    return pd.DataFrame(influence_metrics)

# Analysis of categorical factors influence
categorical_analysis_df = analyze_categorical_influence(
    training_data_with_target, categorical_features)

# Sorting by influence range (highest influence at top)
categorical_analysis_df = categorical_analysis_df.sort_values(
    'range', ascending=False)

print('\n\nAnalysis of categorical factors influence on final grade:')
print(categorical_analysis_df.to_string(index=False))
sleep(10)

# Automatic selection of most significant features
significant_categorical_features = categorical_analysis_df[
    categorical_analysis_df['range'] > 1.0]['feature'].tolist()

print('\n\nSignificant categorical factors:')
print(significant_categorical_features)
sleep(10)


# =============================================================================
# 5. Preparation of training set
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Preparation of training set')
print('=' * 80)
sleep(2)

# Encoding significant categorical features
def encode_categorical_features(data, features_to_encode, target_col='G3'):
    """Performs target encoding for specified categorical features"""
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

# Formation of final feature set for modeling
final_features = selected_numeric_features + [
    f'{feature}_encoded' for feature in significant_categorical_features
]
x_train_processed = training_data_encoded[final_features].copy(deep=True)

print('\n\nEncoded training data (first 5 rows):')
print(x_train_processed.head())
sleep(10)

# Data scaling
data_scaler = MinMaxScaler()
data_scaler.fit(x_train_processed)
x_train_scaled = data_scaler.transform(x_train_processed)

# Converting resulting data from matrix back to table
x_train_scaled_df = pd.DataFrame(
    x_train_scaled,
    columns=x_train_processed.columns,
    index=x_train_processed.index
)

print('\n\nScaled training data (first 5 rows):')
print(x_train_scaled_df.head())
sleep(10)


# =============================================================================
# 6. Preparation of test set
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Preparation of test set')
print('=' * 80)
sleep(2)

test_data_with_target = x_test_split.copy(deep=True)
test_data_with_target['G3'] = y_test_split

# Applying target encoding to categorical features using encoders trained on
# training data
test_data_encoded = test_data_with_target.copy(deep=True)
for feature in significant_categorical_features:
    # Using encoder trained on training data
    encoder = feature_encoders[feature]
    test_data_encoded[f'{feature}_encoded'] = encoder.transform(
        test_data_with_target[feature])

# Formation of final feature set for test set
x_test_processed = test_data_encoded[final_features].copy(deep=True)

print('\n\nEncoded test data (first 5 rows):')
print(x_test_processed.head())
sleep(10)

# Scaling test data using scaler trained on training data
x_test_scaled = data_scaler.transform(x_test_processed)

# Converting resulting data from matrix back to table
x_test_scaled_df = pd.DataFrame(
    x_test_scaled,
    columns=x_test_processed.columns,
    index=x_test_processed.index
)

print('\n\nScaled test data (first 5 rows):')
print(x_test_scaled_df.head())
sleep(5)


# =============================================================================
# 7. Saving training and test sets to CSV file
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Saving training and test sets to CSV file')
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

print('\n\nData saved successfully!\n\n')

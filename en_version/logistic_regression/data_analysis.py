from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder


# =============================================================================
# 1. Loading and initial exploration of flight satisfaction data
# =============================================================================
print('=' * 80)
print('1. Loading and initial exploration of flight satisfaction data')
print('=' * 80)
sleep(2)

# Loading data
flight_data = pd.read_csv('data/satisfaction_survey.csv')
sns.set(rc={'figure.figsize': (11.7, 8.27)})

# Separating features from target variable
feature = flight_data.drop(columns='satisfaction')
target = flight_data['satisfaction']

# Visualization of target variable distribution
print('\n\nAnalyzing chart...')
plt.figure(figsize=(10, 6))
sns.countplot(x=target)
plt.title('Flight satisfaction distribution')
plt.xlabel('Flight satisfaction')
plt.ylabel('Number of passengers')
plt.show()
sleep(2)


# =============================================================================
# 2. Creating training and test sets
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Creating training and test sets')
print('=' * 80)
sleep(2)

# Splitting data into training and test sets (75%:25%)
x_train, x_test, y_train, y_test = train_test_split(
    feature, target, test_size=0.25, random_state=52, stratify=target)

# Combining data for analysis
train_data = x_train.copy(deep=True)
train_data['satisfaction'] = y_train

print('\n\nCreating training and test sets...')
sleep(5)
print(f'Training set size: {len(x_train)}')
print(f'Test set size: {len(x_test)}')
sleep(5)


# =============================================================================
# 3. Handling missing values
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Handling missing values')
print('=' * 80)
sleep(2)

print('\n\nChecking for missing values in data:')
print(train_data.isna().sum())
sleep(10)

# There are missing values in the "Arrival Delay in Minutes" feature,
# we'll fill them with median value
arrival_delay_median = train_data['Arrival Delay in Minutes'].median()
train_data['Arrival Delay in Minutes'] = train_data[
    'Arrival Delay in Minutes'].fillna(arrival_delay_median)
x_test['Arrival Delay in Minutes'] = x_test[
    'Arrival Delay in Minutes'].fillna(arrival_delay_median)

print('\n\nFilling missing values with median values...')
omissions = train_data.isna().sum().sum()
print(f'Total number of missing values after filling: {omissions}')
sleep(5)


# =============================================================================
# 4. Identifying feature types
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Identifying feature types')
print('=' * 80)
sleep(2)

# Numerical features
numeric_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes',
                    'Arrival Delay in Minutes']

# Binary categorical features
binary_features = ['Gender', 'Customer Type', 'Type of Travel']

# Multiclass categorical features
multiclass_features = ['Class']

# Rating features (0-5, treated as categorical)
rating_features = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness']

# All categorical features
categorical_features = binary_features + multiclass_features + rating_features

print('\n\nFeatures have been categorized...')
sleep(2)


# =============================================================================
# 5. Analyzing relationships between numerical features and satisfaction
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Analyzing relationships between numerical features and satisfaction')
print('=' * 80)
sleep(2)

# Analyzing correlation relationships
numeric_data = train_data[numeric_features + ['satisfaction']].copy(deep=True)
correlation_matrix = numeric_data.corr()['satisfaction']

print('\n\nCorrelation of numerical features with satisfaction:')
print(correlation_matrix.sort_values(ascending=False))
sleep(10)

# Analyzing significant correlations
significant_correlations = correlation_matrix[
    (correlation_matrix >= 0.051) | (correlation_matrix <= -0.051)]
print('\n\nStatistically significant correlations:')
print(significant_correlations.sort_values(ascending=False))
sleep(10)

# Visualizing distributions of numerical features
print('\n\nAnalyzing charts...')
for feature in numeric_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x='satisfaction', y=feature, data=train_data)
    plt.title(f'Distribution of "{feature}" by satisfaction')

    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=feature, hue='satisfaction',
                 kde=True, alpha=0.6)
    plt.title(f'Histogram of "{feature}"')

    plt.tight_layout()
    plt.show()

sleep(2)


# =============================================================================
# 6. Analyzing relationships between categorical features and satisfaction
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Analyzing relationships between categorical features and '
      'satisfaction')
print('=' * 80)
sleep(2)

def analyze_categorical_influence(data, categorical_cols,
                                  target_col='satisfaction'):
    """Analyzes the influence of categorical features on the target variable"""
    influence_metrics = []

    for feature in categorical_cols:
        # Aggregating mean values of target variable by categories
        aggregated_data = data.groupby(feature, as_index=False).agg({
            target_col: ['mean', 'count']})
        aggregated_data.columns = [feature, 'satisfaction_mean', 'count']

        # Calculating influence metrics
        target_values = aggregated_data['satisfaction_mean'].values
        influence_metrics.append({
            'factor': feature,
            'range': np.max(target_values) - np.min(target_values),
            'min_satisfaction': np.min(target_values),
            'max_satisfaction': np.max(target_values),
            'categories count': len(aggregated_data)
        })

    return pd.DataFrame(influence_metrics)

# Analyzing influence of categorical features
categorical_analysis = analyze_categorical_influence(
    train_data, categorical_features)
categorical_analysis = categorical_analysis.sort_values(
    'range', ascending=False)

print('\n\nAnalysis of categorical features influence on satisfaction:')
print(categorical_analysis.to_string(index=False))
sleep(20)

# Automatic selection of most significant categorical features
significant_categorical_features = categorical_analysis[
    categorical_analysis['range'] > 0.2
]['factor'].tolist()

print('\n\nSignificant categorical features:')
print(significant_categorical_features)
sleep(10)

# Detailed analysis of significant features
for feature in significant_categorical_features:
    print(f'\nDetailed analysis for feature "{feature}":')
    analysis_table = train_data.groupby(
        [feature, 'satisfaction'], as_index=False).agg(
            {'Age': 'count'}).pivot(
                index=feature, columns='satisfaction', values='Age')
    analysis_table['satisfaction_rate'] = (
        analysis_table[True] / (analysis_table[True] + analysis_table[False]))
    print(analysis_table)
    sleep(10)


# =============================================================================
# 7. Analyzing relationships between categorical features
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Analyzing relationships between categorical features')
print('=' * 80)
sleep(2)

print('\n\nAnalyzing probable relationships between categorical features:')

# Analyzing relationships between services
service_features = ['Inflight wifi service', 'Food and drink',
                    'Seat comfort', 'Inflight entertainment']
for i in range(len(service_features)):
    for j in range(i+1, len(service_features)):
        col_x = service_features[i]
        col_y = service_features[j]

        print(f'\nRelationship between "{col_x}" and "{col_y}":')
        cross_table = train_data.groupby([col_x, col_y], as_index=False).agg({
            'satisfaction': 'count'}).pivot(
                index=col_x, columns=col_y, values='satisfaction')
        print(cross_table.fillna(0).astype(int))
        sleep(10)

# Analyzing relationship between travel type and class
print('\nRelationship between "Type of Travel" and "Class":')
travel_class_table = train_data.groupby(
    ['Type of Travel', 'Class'], as_index=False).agg(
        {'satisfaction': ['count', 'mean']})
print(travel_class_table)
sleep(10)


# =============================================================================
# 8. Selecting final feature set
# =============================================================================
print('\n\n' + '=' * 80)
print('8. Selecting final feature set')
print('=' * 80)
sleep(2)

# Excluding insignificant or highly correlated features
excluded_features = ['Arrival Delay in Minutes', 'Gate location',
                     'Baggage handling']

# Final feature set
final_numeric_features = [f for f in numeric_features
                          if f not in excluded_features]
final_categorical_features = [f for f in significant_categorical_features
                              if f not in excluded_features]

# Separating final categorical features by type
final_binary_features = [f for f in final_categorical_features
                         if f in binary_features]
final_multiclass_features = [f for f in final_categorical_features
                             if f in multiclass_features]
final_rating_features = [f for f in final_categorical_features
                         if f in rating_features]

print('\n\nFinal numerical features:')
print(final_numeric_features)
sleep(5)
print('\nFinal binary features:')
print(final_binary_features)
sleep(5)
print('\nFinal multiclass features:')
print(final_multiclass_features)
sleep(5)
print('\nFinal rating features:')
print(final_rating_features)
sleep(5)


# =============================================================================
# 9. Preparing data for modeling
# =============================================================================
print('\n\n' + '=' * 80)
print('9. Preparing data for modeling')
print('=' * 80)
sleep(2)

# Preparing training set
x_train_processed = x_train[
    final_numeric_features + final_categorical_features].copy(deep=True)
x_test_processed = x_test[
    final_numeric_features + final_categorical_features].copy(deep=True)

# Binary encoding for all categorical features
binary_encoders = {}

# Encoding binary features
for feature in final_binary_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Adding encoded columns
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Removing original column
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Encoding multiclass features
for feature in final_multiclass_features:
    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Adding encoded columns
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Removing original column
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Encoding rating features (0-5) as categorical
for feature in final_rating_features:
    # Converting to string type for proper binary encoding
    x_train_processed[feature] = x_train_processed[feature].astype(str)
    x_test_processed[feature] = x_test_processed[feature].astype(str)

    encoder = BinaryEncoder()
    encoded_train = encoder.fit_transform(x_train_processed[feature])
    encoded_test = encoder.transform(x_test_processed[feature])

    # Adding encoded columns
    for col in encoded_train.columns:
        x_train_processed[col] = encoded_train[col]
        x_test_processed[col] = encoded_test[col]

    # Removing original column
    x_train_processed.drop(columns=[feature], inplace=True)
    x_test_processed.drop(columns=[feature], inplace=True)

    binary_encoders[feature] = encoder

# Identifying columns for scaling (only numerical features)
numeric_cols = final_numeric_features
categorical_cols = [col for col in x_train_processed.columns
                   if col not in numeric_cols]

# Scaling only numerical features
scaler = StandardScaler()
x_train_numeric_scaled = scaler.fit_transform(x_train_processed[numeric_cols])
x_test_numeric_scaled = scaler.transform(x_test_processed[numeric_cols])

# Creating final DataFrame with scaled numerical and original categorical
# features
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

# Adding categorical features without changes
for col in categorical_cols:
    x_train_final[col] = x_train_processed[col].values
    x_test_final[col] = x_test_processed[col].values

print('\n\nData prepared!')
sleep(2)


# =============================================================================
# 10. Displaying prepared data
# =============================================================================
print('\n\n' + '=' * 80)
print('10. Displaying prepared data')
print('=' * 80)
sleep(2)

print('\n\nEncoded training data (first 5 rows):')
print(x_train_processed.head())
sleep(10)

print('\n\nScaled training data (first 5 rows):')
print(x_train_final.head())
sleep(10)

print('\n\nEncoded test data (first 5 rows):')
print(x_test_processed.head())
sleep(10)

print('\n\nScaled test data (first 5 rows):')
print(x_test_final.head())
sleep(10)


# =============================================================================
# 11. Saving training and test sets to CSV files
# =============================================================================
print('\n\n' + '=' * 80)
print('11. Saving training and test sets to CSV files')
print('=' * 80)
sleep(2)

x_train_final.to_csv('data_logistic_regression/x_train_data.csv', index=False)
y_train.to_csv('data_logistic_regression/y_train_data.csv', index=False)
x_test_final.to_csv('data_logistic_regression/x_test_data.csv', index=False)
y_test.to_csv('data_logistic_regression/y_test_data.csv', index=False)

print('\n\nData saved successfully!\n\n')

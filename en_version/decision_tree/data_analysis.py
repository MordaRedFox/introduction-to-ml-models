from time import sleep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# =============================================================================
# 1. Loading and initial exploration of MNIST data
# =============================================================================
print('=' * 80)
print('1. Loading and initial exploration of MNIST data')
print('=' * 80)
sleep(2)

# Reading data
sns.set(rc={'figure.figsize': (11.7, 8.27)})
data = pd.read_csv('data/images.csv')

print(f'\n\nData dimensions: {data.shape}')
sleep(5)
print('\nFirst 5 rows of data:')
print(data.head())
sleep(10)

# Checking for missing values
missing_values = data.isnull().sum().sum()
print(f'\nNumber of missing values: {missing_values}')


# =============================================================================
# 2. Analysis of class distribution (digits 0-9)
# =============================================================================
print('\n\n' + '=' * 80)
print('2. Analysis of class distribution (digits 0-9)')
print('=' * 80)
sleep(2)

class_distribution = data['label'].value_counts().sort_index()
print('\n\nAnalyzing the plot...')

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Digit distribution in MNIST dataset')
plt.xlabel('Digit')
plt.ylabel('Number of images')

# Adding values on bars
for i, v in enumerate(class_distribution.values):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 3. Splitting data into training and test sets
# =============================================================================
print('\n\n' + '=' * 80)
print('3. Splitting data into training and test sets')
print('=' * 80)
sleep(2)

x = data.drop(columns='label')
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=52)

print(f'\n\nTraining set size: {x_train.shape[0]}')
print(f'Test set size: {x_test.shape[0]}')
sleep(10)


# =============================================================================
# 4. Visualization of image examples
# =============================================================================
print('\n\n' + '=' * 80)
print('4. Visualization of image examples')
print('=' * 80)
sleep(2)

def plot_digit(image_flat, ax=None):
    """Visualizes a single digit image"""
    image = image_flat.reshape(28, 28).astype('uint8')

    if ax is None:
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    else:
        ax.imshow(image, cmap='gray')
        ax.axis('off')

# Creating combined DataFrame with features and labels
x_y_train = x_train.copy(deep=True)
x_y_train['label'] = y_train

# Visualization of 10 random examples for each class
print('\n\nAnalyzing the plot...')
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

plt.suptitle('Example digit images from MNIST dataset')
plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 5. Analysis of pixel statistical characteristics
# =============================================================================
print('\n\n' + '=' * 80)
print('5. Analysis of pixel statistical characteristics')
print('=' * 80)
sleep(2)

# Pixel statistics
pixel_stats = x_train.describe()
print('\n\nPixel statistics:')
print(pixel_stats.loc[['mean', 'std', 'min', 'max']].T)
sleep(10)

# Analysis of mean pixel values for each digit
mean_digits = []
for digit in range(10):
    digit_mean = x_train[y_train == digit].mean().values.reshape(28, 28)
    mean_digits.append(digit_mean)

# Visualization of mean images for each digit
print('\n\nAnalyzing the plot...')
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for digit in range(10):
    axes[digit].imshow(mean_digits[digit], cmap='gray')
    axes[digit].set_title(f'Mean for digit {digit}')
    axes[digit].axis('off')

plt.suptitle('Mean images for each digit')
plt.tight_layout()
plt.show()
sleep(2)


# =============================================================================
# 6. Analysis of pixel variability
# =============================================================================
print('\n\n' + '=' * 80)
print('6. Analysis of pixel variability')
print('=' * 80)
sleep(2)

print('\n\nAnalyzing plots...')

# Heatmap of pixel mean values
plt.figure(figsize=(10, 8))
sns.heatmap(pixel_stats.loc['mean'].values.reshape(28, 28), cmap='viridis',
            square=True, cbar_kws={'label': 'Mean intensity'})
plt.title('Heatmap of pixel mean values')
plt.axis('off')
plt.show()
sleep(2)

# Heatmap of pixel standard deviations
plt.figure(figsize=(10, 8))
sns.heatmap(pixel_stats.loc['std'].values.reshape(28, 28), cmap='plasma',
            square=True, cbar_kws={'label': 'Standard deviation'})
plt.title('Heatmap of pixel variability')
plt.axis('off')
plt.show()
sleep(2)


# =============================================================================
# 7. Informative feature (pixel) selection
# =============================================================================
print('\n\n' + '=' * 80)
print('7. Informative feature (pixel) selection')
print('=' * 80)
sleep(2)

# Analysis of pixels with low variability
low_variance_pixels = (pixel_stats.loc['std'] < 5).sum()
percent_low_variance_pixels = low_variance_pixels / 784 * 100
print('\n\nInformative feature selection:')
print(f'Number of pixels with low variability: {low_variance_pixels}')
print(f'Percentage of low-variance pixels: {percent_low_variance_pixels:.1f}%')
sleep(10)

# Feature selection based on variance threshold
selector = VarianceThreshold(threshold=0.1)
selector.fit(x_train)

# Analysis of selected features
selected_features = selector.get_feature_names_out()
print(f'\n\nTotal features: {x_train.shape[1]}')
print(f'Selected informative features: {len(selected_features)}')
sleep(10)

# Applying feature selection
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

print('\nDimensions after feature selection:')
print(f'Training set: {x_train_selected.shape}')
print(f'Test set: {x_test_selected.shape}')
sleep(10)

print('\n\nFirst 5 rows of optimized data:')
print(x_train_selected.head())
sleep(10)


# =============================================================================
# 8. Saving training and test sets to CSV files
# =============================================================================
print('\n\n' + '=' * 80)
print('8. Saving training and test sets to CSV files')
print('=' * 80)
sleep(2)

x_train_selected.to_csv('data_decision_tree/x_train_data.csv', index=False)
x_test_selected.to_csv('data_decision_tree/x_test_data.csv', index=False)
y_train.to_csv('data_decision_tree/y_train_data.csv', index=False)
y_test.to_csv('data_decision_tree/y_test_data.csv', index=False)

print('\n\nData saved successfully!\n\n')

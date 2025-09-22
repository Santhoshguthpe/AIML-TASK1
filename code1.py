import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("E:\Titanic-Dataset.csv")

# Display the first 5 rows of the dataframe
print("Initial Data:")
print(df.head())

# Get information about the data, including data types and missing values
print("\nData Info:")
print(df.info())


# Impute missing 'Age' values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Impute missing 'Fare' values with the median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Impute missing 'Embarked' values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Verify that all missing values have been handled
print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# Label encode the 'Sex' column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encode the 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nAfter Encoding Categorical Features:")
print(df.head())

# Select the numerical features to scale
numerical_features = ['Age', 'Fare']

# Standardize the numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nAfter Normalizing Numerical Features:")
print(df.head())

# Visualize the 'Fare' column for outliers before removal
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare (Before Outlier Removal)')
plt.show()

# Calculate the IQR for the 'Fare' column
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the DataFrame
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# Visualize the 'Fare' column for outliers after removal
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare (After Outlier Removal)')
plt.show()

print("\nShape of DataFrame after removing outliers:", df.shape)


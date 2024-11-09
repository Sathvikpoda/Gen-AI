import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset (modify the path to your file)
df = pd.read_csv(r"C:\Users\sathv\Downloads\cleaned_dailyActivity.csv")

# Preview the first few rows of the dataset
print(df.head())
print(df.tail())

# Check for missing values in each column
print("Missing values per column:\n", df.isnull().sum())

# Check for duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

# Get a summary of numeric columns
print("Summary statistics for numerical columns:\n", df.describe())

# Drop the specified columns
columns_to_drop = ['Id', 'ActivityDate', 'TotalSteps', 'TrackerDistance']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')  # Use errors='ignore' to avoid errors if columns don't exist

# Convert categorical data into numerical data using one-hot encoding if necessary
# Assuming there are categorical columns, for example, 'Gender' or 'ActivityType'
if 'Gender' in df.columns:
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)  # Convert 'Gender' to numerical

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap for correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Define features and target variable
X = df.drop(columns=['Calories'])  # Drop target variable
y = df['Calories']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optional: Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.title('Actual vs Predicted Calories Burned')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.show()

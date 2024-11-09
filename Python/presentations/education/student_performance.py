import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset (modify the path to your file)
df = pd.read_csv(r"C:\Users\sathv\Downloads\Student_performance_data _.csv")

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line breaks in the output

# Preview the first few rows of the dataset
print(df.head())

# Get the structure of the dataset
print(df.info())  # Gives an overview of the data
print("Shape of the dataset:", df.shape)  # Number of rows and columns

# Check for missing values in each column
print("Missing values per column:\n", df.isnull().sum())

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)  # Fill missing values with the mean of each column

# Specify the features and target variable
# Change target variable to 'GPA'
target_variable = 'GPA'  # Update to the correct target variable name
features = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular']  # Add more features as needed

# Print available columns before encoding
print("Available columns before encoding:", df.columns.tolist())

# Encode categorical variables if necessary (example: if 'Tutoring' is categorical)
df = pd.get_dummies(df, columns=['Tutoring', 'ParentalSupport', 'Extracurricular'], drop_first=True)

# Print available columns after encoding
print("Available columns after encoding:", df.columns.tolist())

# Update features list based on available columns
available_features = [feature for feature in features if feature in df.columns]
print("Updated features list:", available_features)

# Split the data into training and testing sets
X = df[available_features]
y = df[target_variable]
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
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs Predicted GPA')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.show()

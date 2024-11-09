# here 0 is they are not having the diabetes and 1 is they are having the diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv(r'C:\Users\sathv\Downloads\Healthcare-Diabetes.csv')
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Sample DataFrame
column_names=['Glucose','BloodPressure','Insulin','BMI']

# Function to replace 0 with the mean of the column
def replace_zeros_with_mean(df, column_names):
    for column in column_names:
        # Calculate the mean, excluding zeros
        col_mean = df[df[column] != 0][column].mean()

        # Replace 0 values with the calculated mean
        df[column] = df[column].replace(0, col_mean)

# List of columns where you want to replace 0 with mean
columns_to_replace = ['Glucose','BloodPressure','Insulin','BMI']

# Apply the function
replace_zeros_with_mean(df, columns_to_replace)

print(df)
print(df.describe())

plt.boxplot(x=df['Insulin'])
plt.show()

# Calculate the correlation matrix excluding 'Id', 'Age', and 'SkinThickness'
columns_to_exclude = ['Id', 'Age', 'SkinThickness']
correlation_matrix = df.drop(columns=columns_to_exclude).corr()

# Set up the matplotlib figure for the heatmap
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Title for the heatmap
plt.title('Correlation Heatmap (Excluding Id, Age, and SkinThickness)')

# Show the plot
plt.show()

# Exclude the 'Id' and 'Outcome' columns for independent variables
X = df.drop(columns=['Id', 'Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy of the Logistic Regression model:", accuracy)

# Plotting the predicted probabilities
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=30, color='skyblue', edgecolor='black')
plt.title('Predicted Probabilities of Diabetes')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.axvline(x=0.5, color='red', linestyle='--')  # Threshold line
plt.show()

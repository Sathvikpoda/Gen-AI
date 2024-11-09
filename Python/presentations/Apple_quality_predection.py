import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset (modify the path to your file)
df = pd.read_csv(r"C:\Users\sathv\Downloads\apple_quality.csv")

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

# Convert categorical data into numerical data using one-hot encoding if necessary
# Assuming 'Quality' is the target variable and is categorical
if 'Quality' in df.columns:
    # Map 'good' to 1 and 'bad' to 0
    df['Quality'] = df['Quality'].map({'good': 1, 'bad': 0})

# Drop the ID column if it exists
if 'A_id' in df.columns:
    df.drop(columns=['A_id'], inplace=True)

# Visualize the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap for correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Define features and target variable
# Assuming the target variable is now 'Quality'
X = df.drop(columns=['Quality'])  # Drop target variable
y = df['Quality']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

# Optional: Visualize the predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Plotting the predicted probabilities
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_prob, alpha=0.5)
plt.xlabel('Actual Quality (1=Good, 0=Bad)')
plt.ylabel('Predicted Probability of Good Quality')
plt.title('Predicted Probabilities of Good Quality')
plt.axhline(0.5, color='red', linestyle='--')  # Threshold line
plt.show()

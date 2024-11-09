import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\sathv\Downloads\filtered_data_month_1.csv"
filtered_df = pd.read_csv(file_path)

# Check the dataset structure
print(filtered_df.head())
print(filtered_df.columns)

# Check for missing values
missing_values = filtered_df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Check for duplicate rows
duplicate_rows = filtered_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Basic statistics of the dataset
basic_stats = filtered_df.describe()
print("Basic statistics of the dataset:")
print(basic_stats)



# Display dataset information after conversions
print(filtered_df.info())

# Convert categorical data into numerical data using one-hot encoding


# Remove the first four columns from the DataFrame for correlation analysis
filtered_df_reduced = filtered_df.iloc[:, 4:]  # Adjust the index based on your dataset structure

# Calculate the correlation matrix
correlation_matrix = filtered_df_reduced.corr()

# Plot the heatmap for correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap (Excluding First 4 Columns)')
plt.show()



# Define features and target variable
X = filtered_df.drop(columns=['total_amount', 'fare_amount'])  # Drop target variables
y = filtered_df['fare_amount']  # Use fare_amount as the target variable

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
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.title('Actual vs Predicted Fare Amount')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.show()

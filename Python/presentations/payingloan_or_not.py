import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Load your dataset (modify the path to your file)
df = pd.read_csv(r"C:\Users\sathv\Downloads\Loan_default.csv\Loan_default.csv")

# Display the first few rows of the dataset
print(df.head())

# Get the structure of the dataset
print(df.info())  # Gives an overview of the data
print("Shape of the dataset:", df.shape)  # Number of rows and columns

# Check for missing values (optional)
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", categorical_cols)

# Check the number of unique values in each categorical column
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# Convert categorical data into numerical data using one-hot encoding
# Consider dropping high cardinality columns or using alternative encoding
if 'LoanID' in categorical_cols:
    categorical_cols = categorical_cols.drop('LoanID')

# Convert remaining categorical columns into numerical data using one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop the LoanID column if it still exists
if 'LoanID' in df.columns:
    df.drop(columns=['LoanID'], inplace=True)

# Calculate the correlation matrix excluding dummy variables
# Select only numeric columns (excluding dummy variables)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()

# Plot the heatmap for correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap (Excluding Dummy Variables)')
plt.show()

# Define features and target variable
# Assuming 'Default' is the target variable indicating if the loan was paid back
X = df.drop(columns=['Default'])  # Drop target variable
y = df['Default']  # Use Default as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)

# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Default', 'Default'], yticklabels=['Not Default', 'Default'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plotting the predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=30, color='skyblue', edgecolor='black')
plt.title('Predicted Probabilities of Default')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.axvline(x=0.5, color='red', linestyle='--')  # Threshold line
plt.show()

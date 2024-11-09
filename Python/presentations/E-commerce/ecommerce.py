import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency, ttest_ind

# Step 1: Create a synthetic e-commerce dataset
data = pd.DataFrame({
    'Age': np.random.randint(18, 65, 100),  # Random ages between 18 and 65
    'Gender': np.random.choice(['Male', 'Female'], 100),  # Random gender selection
    'Time_Spent': np.random.randint(5, 50, 100),  # Random time spent on the website in minutes
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Books'], 100),  # Random product categories
    'Purchased': np.random.choice([0, 1], 100)  # 0 for no purchase, 1 for purchase
})

# Display the first few rows of the dataset
print("Sample Data:")
print(data.head())

# Step 2: Understanding the data - Basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Step 3: Numerical Parameters to Represent Data
mean_time_spent = data['Time_Spent'].mean()
median_time_spent = data['Time_Spent'].median()
mode_product_category = data['Product_Category'].mode()[0]

print("\nMean Time Spent:", mean_time_spent)
print("Median Time Spent:", median_time_spent)
print("Most Frequent Product Category (Mode):", mode_product_category)

# Step 4: Probability and Chi-Square Test
# Check the association between Gender and Purchase using a chi-square test
contingency_table = pd.crosstab(data['Gender'], data['Purchased'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results: p-value = {p}")

# Step 5: Hypothesis Testing - t-test
# Test if the average time spent is significantly different between purchasers and non-purchasers
purchasers = data[data['Purchased'] == 1]['Time_Spent']
non_purchasers = data[data['Purchased'] == 0]['Time_Spent']
t_stat, p_value = ttest_ind(purchasers, non_purchasers)
print(f"\nT-test Results: p-value = {p_value}")

# Step 6: Clustering - Visualizing the relationship
sns.scatterplot(x='Age', y='Time_Spent', hue='Purchased', data=data)
plt.title('Age vs. Time Spent with Purchase Status')
plt.show()

# Step 7: Regression Modeling - Logistic Regression
# Prepare the data for modeling
X = data[['Age', 'Time_Spent']]  # Features
y = data['Purchased']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluating the Model
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)

# Step 9: Data Visualization - Boxplot
sns.boxplot(x='Purchased', y='Time_Spent', data=data)
plt.title('Time Spent vs. Purchase Status')
plt.show()

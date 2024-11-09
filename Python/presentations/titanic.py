import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
url = r"C:\Users\sathv\Downloads\tested.csv"  # Ensure this path is correct
data = pd.read_csv(url)

# Data preprocessing
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Verify no missing values remain
if data.isna().sum().sum() > 0:
    print("There are still missing values in the data.")
    print(data.isna().sum())
else:
    print("No missing values detected in the data.")

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=200)

# Perform k-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.2f}")

# Train the model on the full dataset for final evaluation
model.fit(X, y)

# Analyze survival rates by age group
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['0-12', '13-18', '19-35', '36-60', '60+'])
age_group_survival = data.groupby('AgeGroup')['Survived'].mean() * 100

# Display survival rates by age group
print("\nSurvival rates by age group:")
print(age_group_survival)

# Visualization of survival by age group
plt.figure(figsize=(10, 6))
sns.barplot(x=age_group_survival.index, y=age_group_survival.values)
plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)
plt.show()

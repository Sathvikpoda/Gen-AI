# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map the target numbers to actual species names
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_mapping)

# Separate features and target variable
X = df.drop(columns=['species'])
y = df['species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply PCA to reduce to 2 components on the training data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Transform the test data using the same PCA transformation
X_test_pca = pca.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression()

# Perform k-fold cross-validation on the training data
cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5)  # 5-fold cross-validation

# Print cross-validated accuracy scores and mean accuracy
print(f"Cross-validated accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.2f}")

# Fit the model on the training data
model.fit(X_train_pca, y_train)

# Evaluate the model on the test data
test_accuracy = model.score(X_test_pca, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Print the coefficients of the model
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Calculate training accuracy
train_accuracy = model.score(X_train_pca, y_train)
print(f"Training accuracy: {train_accuracy:.2f}")

# Optional: Plot the PCA results for the training set
pca_train_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
pca_train_df['species'] = y_train.reset_index(drop=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_train_df, x='PC1', y='PC2', hue='species', palette='viridis')
plt.title('PCA of Iris Dataset (Training Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

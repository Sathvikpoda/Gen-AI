# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score

# Load the Wholesale Customers dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)

# Display the first few rows
print("Dataset Head:\n", df.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA to reduce to 2 components for visualization and clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA-transformed Data Head:\n", pd.DataFrame(X_pca, columns=['PC1', 'PC2']).head())

# KMeans Clustering and K-Fold Cross Validation
kmeans = KMeans(n_clusters=5, random_state=42)  # Set an appropriate number of clusters

# Use KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# Perform cross-validation
for train_index, test_index in kfold.split(X_pca):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    # Fit the model and predict
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    
    # Measure clustering consistency using the inertia score
    score = kmeans.inertia_
    scores.append(score)

# Print the mean inertia score (lower is better for KMeans consistency)
print("\nMean Inertia Score (K-Fold Cross-Validation):", np.mean(scores))

# Fit KMeans to the entire PCA-transformed dataset for visualization
kmeans.fit(X_pca)
labels = kmeans.labels_

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_pca, labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualize the clusters on the 2D PCA plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100)
plt.title('Customer Segmentation based on PCA and KMeans')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

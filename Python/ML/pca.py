import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\sathv\Downloads\IRIS.csv"  # Adjust path as needed
df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop(columns=['species'])  # Replace 'species' with the actual target column if different
y = df['species']

# Visualization 1: Original Features (Petal Length vs. Petal Width)
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='viridis')
plt.title('Without PCA: Petal Length vs. Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Standardize the features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y

# Visualization 2: PCA (PC1 vs. PC2)
plt.subplot(1, 2, 2)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', palette='viridis')
plt.title('With PCA: Principal Component 1 vs. Principal Component 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()

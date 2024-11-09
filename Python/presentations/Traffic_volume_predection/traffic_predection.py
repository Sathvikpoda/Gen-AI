import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define the file path
file_path = r"C:\Users\sathv\Downloads\Metro_Interstate_Traffic_Volume.csv\Metro_Interstate_Traffic_Volume.csv"

# Initialize df to None
df = None

# Check if the file exists
if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path)
        print("File loaded successfully.")
    except PermissionError:
        print("Permission denied. Please check the file permissions.")
else:
    print("File does not exist. Please check the file path.")

# Check if df is defined before trying to use it
if df is not None:
    # Preview the first few rows of the dataset
    print(df.head())

    # Get the structure of the dataset
    print(df.info())  # Gives an overview of the data
    print("Shape of the dataset:", df.shape)  # Number of rows and columns

    # Check for missing values in each column
    print("Missing values per column:\n", df.isnull().sum())

    # Fill null values in the 'holiday' column with the string 'nan'
    df['holiday'].fillna('nan', inplace=True)

    # Check for duplicate rows
    print("Number of duplicate rows before removal:", df.duplicated().sum())

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Check the number of duplicate rows after removal
    print("Number of duplicate rows after removal:", df.duplicated().sum())

    # Get a summary of numeric columns
    print("Summary statistics for numerical columns:\n", df.describe())

    # Visualize the correlation matrix before predictions
    # Convert categorical data into numerical data using one-hot encoding
    df = pd.get_dummies(df, columns=['holiday', 'weather_main', 'weather_description'], drop_first=True)

    # Drop non-numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot the heatmap for correlation
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()

    # Remove outliers using the IQR method
    for column in ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the DataFrame to remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Check the shape of the dataset after removing outliers
    print("Shape of the dataset after removing outliers:", df.shape)

    # Define features and target variable
    X = df.drop(columns=['traffic_volume', 'date_time'])  # Drop target and non-numeric columns
    y = df['traffic_volume']

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
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
    plt.show()

    # Set the style for seaborn
    sns.set(style="whitegrid")

    # 1. Histogram for numerical features
    plt.figure(figsize=(12, 6))
    df['traffic_volume'].hist(bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Traffic Volume')
    plt.xlabel('Traffic Volume')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Box Plot for traffic volume
    plt.figure(figsize=(12, 6))
    sns.boxplot(y=df['traffic_volume'])
    plt.title('Box Plot of Traffic Volume')
    plt.ylabel('Traffic Volume')
    plt.show()

    # 3. Scatter Plot between temperature and traffic volume (before removing outliers)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='temp', y='traffic_volume', data=df, alpha=0.5)
    plt.title('Traffic Volume vs Temperature (Outliers Visible)')
    plt.xlabel('Temperature')
    plt.ylabel('Traffic Volume')
    plt.show()

    # 4. Scatter Plot for rain_1h vs traffic_volume (before removing outliers)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='rain_1h', y='traffic_volume', data=df, alpha=0.5)
    plt.title('Traffic Volume vs Rain in Last Hour (Outliers Visible)')
    plt.xlabel('Rain (in mm)')
    plt.ylabel('Traffic Volume')
    plt.show()

else:
    print("DataFrame is not defined. Cannot proceed with analysis.")

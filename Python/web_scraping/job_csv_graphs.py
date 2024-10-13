import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

def convert_relative_date(relative_date):
    """Convert a relative date string into an actual date."""
    match = re.match(r'Posted (\d+) (day|days) ago', relative_date)
    if match:
        days_ago = int(match.group(1))
        return datetime.now() - timedelta(days=days_ago)
    return None  # Handle unexpected formats

def plot_job_roles(csv_filename):
    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Print the columns for debugging
    print("Columns in CSV:", df.columns)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure the necessary columns exist
    if 'Job Title' not in df.columns:
        print("Required columns are missing.")
        return

    # List of roles to categorize
    roles = ['Frontend Developer', 'Backend Developer', 'Software Engineer', 'Web Developer', 'Java Developer']

    # Create a dictionary to count the occurrences of each role
    role_counts = {role: 0 for role in roles}

    # Count the occurrences of each role in the "Job Title" column
    for job_title in df['Job Title']:
        for role in roles:
            if role.lower() in str(job_title).lower():
                role_counts[role] += 1

    # Filter out roles with zero count
    role_counts = {role: count for role, count in role_counts.items() if count > 0}

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.pie(role_counts.values(), labels=role_counts.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Percentage of Job Listings by Role')
    plt.show()

def plot_daily_postings(csv_filename):
    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure the necessary columns exist
    if 'Date Posted' not in df.columns:
        print("Required columns are missing.")
        return

    # Convert relative date strings to actual dates
    df['Date Posted'] = df['Date Posted'].apply(convert_relative_date)

    # Check for NaT values
    if df['Date Posted'].isna().sum() > 0:
        print(f"Warning: {df['Date Posted'].isna().sum()} dates could not be parsed.")
    
    # Count job postings per day
    df['Date Posted'] = df['Date Posted'].dt.date  # Convert to date only
    daily_counts = df['Date Posted'].value_counts().sort_index()

    # Check if daily_counts is empty
    if daily_counts.empty:
        print("No valid date entries found. Exiting.")
        return

    # Bar chart for daily job postings
    plt.figure(figsize=(10, 6))
    daily_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Job Postings Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Postings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Usage
csv_filename = r'C:\Users\sathv\Gen-AI\python\Python\webscraping\dice_full_stack_development_jobs.csv'

# Plot job roles
plot_job_roles(csv_filename)

# Plot daily postings
plot_daily_postings(csv_filename)

import numpy as np
import random

# Simulating customer spending data for a large population (in dollars)
np.random.seed(0)
population_spending = np.random.normal(loc=200, scale=50, size=10000)  # Mean = 200, Std Dev = 50

# Randomly selecting a sample of 100 customers from the population
sample_size = 100
sample_spending = random.sample(list(population_spending), sample_size)

# Point estimation: Calculate the sample mean to estimate the population mean
sample_mean = np.mean(sample_spending)

# Actual population mean for comparison
population_mean = np.mean(population_spending)

# Displaying the results
print(f"Estimated Average Spending (Sample Mean): ${sample_mean:.2f}")
print(f"Actual Average Spending (Population Mean): ${population_mean:.2f}")

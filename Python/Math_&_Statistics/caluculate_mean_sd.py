import numpy as np
import scipy.stats as stats

# Data set
data = np.random.normal(50,5,1000)

mean = np.mean(data)

# Calculating variance
variance = np.var(data, ddof=0)  # Population variance

# Calculating standard deviation
standard_deviation = np.sqrt(variance)

# Calculating skewness
skewness = stats.skew(data)

# Display the results
print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", standard_deviation)
print("Skewness:", skewness)

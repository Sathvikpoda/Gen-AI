import numpy as np
from scipy.stats import ttest_ind
#Step 1: A/B Testing - Compare Two Versions of the Checkout Process

# Simulated data: Conversion outcomes for 100 users for each checkout process
# 1 represents a conversion (purchase), 0 represents no conversion
checkout_A = np.random.binomial(1, 0.20, 100)  # 20% conversion rate for Checkout A
checkout_B = np.random.binomial(1, 0.30, 100)  # 30% conversion rate for Checkout B

# Perform T-test for A/B testing
t_stat, p_value = ttest_ind(checkout_A, checkout_B)

print(f"A/B Testing - T-statistic: {t_stat:.2f}, P-value: {p_value:.2f}")

# Interpret the results
if p_value < 0.05:
    print("There is a significant difference between Checkout A and Checkout B.")
else:
    print("No significant difference between Checkout A and Checkout B.")

#Step 2: Parametric Test - T-test for Average Order Value Before and After a Sale
# Simulated data: Average order values before and after a promotional sale
before_sale = np.random.normal(loc=100, scale=20, size=50)  # Average order value $100
after_sale = np.random.normal(loc=110, scale=20, size=50)   # Average order value $110

# Perform T-test to compare the means
t_stat, p_value = ttest_ind(before_sale, after_sale)

print(f"Parametric Test - T-statistic: {t_stat:.2f}, P-value: {p_value:.2f}")

# Interpret the results
if p_value < 0.05:
    print("There is a significant difference in average order value before and after the sale.")
else:
    print("No significant difference in average order value before and after the sale.")

# Step 3: Non-Parametric Test - Mann-Whitney U Test for Customer Satisfaction Scores
from scipy.stats import mannwhitneyu

# Simulated data: Customer satisfaction scores for three categories
electronics = np.random.randint(1, 6, 30)  # Scores between 1 and 5
clothing = np.random.randint(1, 6, 30)
home_decor = np.random.randint(1, 6, 30)

# Mann-Whitney U test to compare Electronics vs. Clothing
u_stat, p_value = mannwhitneyu(electronics, clothing)

print(f"Non-Parametric Test - Mann-Whitney U statistic: {u_stat:.2f}, P-value: {p_value:.2f}")

# Interpret the results
if p_value < 0.05:
    print("There is a significant difference in customer satisfaction between Electronics and Clothing.")
else:
    print("No significant difference in customer satisfaction between Electronics and Clothing.")

#Step 4: Chi-Square Test - Association Between Payment Method and Device Type
import pandas as pd
from scipy.stats import chi2_contingency

# Sample data: Contingency table of payment methods vs device types
data = pd.DataFrame({
    'Credit Card': [30, 20, 50],
    'PayPal': [25, 35, 40],
    'Bank Transfer': [15, 25, 30]
}, index=['Desktop', 'Mobile', 'Tablet'])

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(data)

print(f"Chi-Square Test - Chi-Square statistic: {chi2_stat:.2f}, P-value: {p_value:.2f}")

# Interpret the results
if p_value < 0.05:
    print("There is a significant association between payment method and device type.")
else:
    print("No significant association between payment method and device type.")

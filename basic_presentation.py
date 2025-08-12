# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

# Define file paths
base_dir = Path('C:/python/census_ecommerce/data/synthetic')
macro_file = Path('C:/python/census_ecommerce/data/processed/ecommerce_makro_indicators.csv')

# Load the data
print("Loading data...")
customers = pd.read_csv(base_dir / 'customers.csv')
orders = pd.read_csv(base_dir / 'orders.csv')
products = pd.read_csv(base_dir / 'products.csv')
macro = pd.read_csv(macro_file)

# Convert date columns
orders['Order_Date'] = pd.to_datetime(orders['Order_Date'])
customers['Signup_Date'] = pd.to_datetime(customers['Signup_Date'])
macro['Date'] = pd.to_datetime(macro['Date'])

# Display basic info
print("\n=== Basic Information ===")
print("Customers shape:", customers.shape)
print("Orders shape:", orders.shape)
print("Products shape:", products.shape)
print("Macro indicators shape:", macro.shape)

# Show data types and missing values
print("\n=== Data Types and Missing Values ===")
for name, df in [('Customers', customers), ('Orders', orders),
                ('Products', products), ('Macro', macro)]:
    print(f"\n{name} Info:")
    print(df.info())
    print(f"\n{name} Missing Values:")
    print(df.isnull().sum())

# Basic statistics for numerical columns
print("\n=== Basic Statistics ===")
print("\nOrders Summary:")
print(orders[['Sales', 'Quantity', 'Profit', 'Discount']].describe())

print("\nProducts Summary:")
print(products[['Price', 'Cost_Price']].describe())

print("\nMacro Indicators Summary:")
print(macro.describe())

# Time-based analysis
print("\n=== Time-Based Analysis ===")
orders_by_month = orders.set_index('Order_Date').resample('M')['Order_ID'].count()
print("\nOrders per Month:")
print(orders_by_month)

# Plotting (will show in IPython if using Jupyter or with plt.show() in regular IPython)
plt.figure(figsize=(12, 6))
sns.lineplot(x=orders_by_month.index, y=orders_by_month.values)
plt.title('Orders Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Customer analysis
print("\n=== Customer Analysis ===")
print("\nTop 5 Regions by Customer Count:")
print(customers['Region'].value_counts().head())

# Order value analysis
print("\nOrder Value Analysis:")
print(f"Average Order Value: ${orders['Sales'].mean():.2f}")
print(f"Median Order Value: ${orders['Sales'].median():.2f}")

# Correlation matrix for numerical variables
plt.figure(figsize=(10, 8))
numeric_cols = ['Sales', 'Quantity', 'Profit', 'Discount']
sns.heatmap(orders[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nAnalysis complete! The following DataFrames are available:")
print("- customers: Customer information")
print("- orders: Order details")
print("- products: Product catalog")
print("- macro: Macroeconomic indicators")
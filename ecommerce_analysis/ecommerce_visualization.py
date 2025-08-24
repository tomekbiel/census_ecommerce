import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Set the style for the plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Constants
DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
OUTPUT_DIR = Path(__file__).parent.parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the data
print("Loading data...")
customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=['join_date'])
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=['order_date'])
products = pd.read_csv(DATA_DIR / "products.csv")

# Data Preparation
print("Preparing data...")
# Create order_details from orders data
order_details = orders.copy()
# Extract product information from the products_list column
order_details['products'] = order_details['products_list'].str.split('|')
order_details = order_details.explode('products')

# Extract product_id and quantity from products string (format: "PROD123(x1)")
# First, handle any potential NaN values in the products column
order_details = order_details.dropna(subset=['products'])

# Extract product_id and quantity with proper error handling
extracted = order_details['products'].str.extract(r'([A-Z0-9]+)\(x(\d+)\)')
extracted.columns = ['product_id', 'quantity']

# Only keep rows where we successfully extracted both product_id and quantity
order_details = order_details[extracted['product_id'].notna() & extracted['quantity'].notna()].copy()
order_details['product_id'] = extracted['product_id']
order_details['quantity'] = extracted['quantity'].astype(int)

# Merge with products to get product details
order_details = order_details.merge(
    products[['product_id', 'name', 'category', 'price']], 
    on='product_id', 
    how='left'
)

# Calculate item total
order_details['item_total'] = order_details['price'] * order_details['quantity'] * (1 - order_details['avg_discount'])

# Keep only relevant columns
order_details = order_details[[
    'order_id', 'customer_id', 'product_id', 'name', 'category', 
    'price', 'quantity', 'avg_discount', 'item_total', 'order_date'
]]

# Add year, month, and day of week to orders
orders['order_year'] = orders['order_date'].dt.year
orders['order_month'] = orders['order_date'].dt.month
orders['order_day_of_week'] = orders['order_date'].dt.dayofweek
orders['order_day_name'] = orders['order_date'].dt.day_name()
orders['month_year'] = orders['order_date'].dt.to_period('M').astype(str)

# Calculate order metrics
monthly_metrics = orders.groupby('month_year').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('total_amount', 'sum'),
    avg_order_value=('total_amount', 'mean'),
    unique_customers=('customer_id', 'nunique')
).reset_index()

# Convert month_year back to datetime for plotting
monthly_metrics['month_year_dt'] = pd.to_datetime(monthly_metrics['month_year'])

# Create visualizations
print("Creating visualizations...")
plt.figure(figsize=(12, 6))
plt.plot(monthly_metrics['month_year_dt'], monthly_metrics['total_revenue'], 
         marker='o', linestyle='-', linewidth=2, markersize=6)
plt.title('Monthly Revenue Trend', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12, labelpad=10)
plt.ylabel('Revenue ($)', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monthly_revenue_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Customer Cohort Analysis
print("Creating cohort analysis...")
# Create a period based on the order date
orders['order_period'] = orders['order_date'].dt.to_period('M')

# Create a cohort based on the first purchase
df_cohort = orders.copy()
cohorts = df_cohort.groupby('customer_id')['order_date'].min().dt.to_period('M')
cohorts = cohorts.reset_index()
cohorts = cohorts.rename(columns={'order_date': 'cohort'})

# Merge cohort with orders
df_cohort = df_cohort.merge(cohorts, on='customer_id', how='left')

# Calculate cohort index (months since first purchase)
df_cohort['cohort_index'] = (df_cohort.order_period - df_cohort.cohort).apply(lambda x: x.n)

# Calculate retention matrix
cohort_pivot = df_cohort.pivot_table(
    index='cohort',
    columns='cohort_index',
    values='customer_id',
    aggfunc=pd.Series.nunique
)

# Calculate retention rates
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

# Plot retention heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    retention_matrix,
    annot=True,
    fmt='.1%',
    cmap='YlGnBu',
    vmin=0.0,
    vmax=0.5,
    cbar_kws={'label': 'Retention Rate'}
)
plt.title('Customer Retention by Cohort (as % of First Month)', fontsize=14, pad=20)
plt.xlabel('Months Since First Purchase', fontsize=12, labelpad=10)
plt.ylabel('Cohort (First Purchase Month)', fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'customer_retention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Category Analysis
category_sales = order_details.groupby('category').agg(
    item_total=('price', 'sum'),
    order_count=('order_id', 'nunique'),
    avg_price=('price', 'mean')
).sort_values('item_total', ascending=False).reset_index()

# Top Categories by Revenue
plt.figure(figsize=(10, 6))
top_cats = category_sales.head(10).copy()
top_cats['dummy'] = 'Revenue'  # Add a dummy variable for hue
sns.barplot(x='item_total', y='category', data=top_cats, hue='dummy', palette='viridis', legend=False)
plt.title('Top 10 Categories by Revenue', fontsize=14, pad=20)
plt.xlabel('Total Revenue ($)', fontsize=12, labelpad=10)
plt.ylabel('Category', fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total Customers: {len(customers):,}")
print(f"Total Orders: {len(orders):,}")
print(f"Total Revenue: ${orders['total_amount'].sum():,.2f}")
print(f"Average Order Value: ${orders['total_amount'].mean():.2f}")
print(f"Top Category by Revenue: {category_sales.iloc[0]['category']} (${category_sales.iloc[0]['item_total']:,.2f})")
print(f"\nVisualization files saved to: {OUTPUT_DIR.absolute()}")

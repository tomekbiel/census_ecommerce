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
OUTPUT_DIR = Path(__file__).parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the data
print("Loading data...")
customers = pd.read_csv(DATA_DIR / "customers2.csv", parse_dates=['join_date'])
orders = pd.read_csv(DATA_DIR / "orders2.csv", parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR / "order_items2.csv")
products = pd.read_csv(DATA_DIR / "products2.csv")

# Data Preparation
print("Preparing data...")
# Create order_details by merging orders with order_items
order_details = pd.merge(
    order_items,
    orders[['order_id', 'order_date', 'customer_id', 'status']],
    on='order_id',
    how='left'
)

# Merge with products to get product details
order_details = pd.merge(
    order_details,
    products[['product_id', 'name', 'category', 'price']],
    on='product_id',
    how='left'
)

# Calculate total price for each order item
order_details['total_price'] = order_details['price'] * order_details['quantity']

# Convert price to numeric
order_details['price'] = pd.to_numeric(order_details['price'], errors='coerce')
order_details = order_details.dropna(subset=['price', 'quantity'])

# Set proper data types
order_details['product_id'] = order_details['product_id'].astype(str)
order_details['customer_id'] = order_details['customer_id'].astype(str)

# Add month and year columns for time-based analysis
# Calculate total price for each order item
order_details['item_total'] = order_details['price'] * order_details['quantity']

# Add year, month, and day of week to order_details
order_details['order_year'] = order_details['order_date'].dt.year
order_details['order_month'] = order_details['order_date'].dt.month
order_details['month_year'] = order_details['order_date'].dt.to_period('M').astype(str)
order_details['order_day_of_week'] = order_details['order_date'].dt.dayofweek
order_details['order_day_name'] = order_details['order_date'].dt.day_name()

# Calculate monthly metrics
monthly_metrics = order_details.groupby('month_year').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('item_total', 'sum'),
    avg_order_value=('item_total', 'mean'),
    unique_customers=('customer_id', 'nunique')
).reset_index()

# Convert month_year back to datetime for plotting
monthly_metrics['month_year_dt'] = pd.to_datetime(monthly_metrics['month_year'])

# Add year, month, and day of week to orders
orders['order_year'] = orders['order_date'].dt.year
orders['order_month'] = orders['order_date'].dt.month
orders['order_day_of_week'] = orders['order_date'].dt.dayofweek
orders['order_day_name'] = orders['order_date'].dt.day_name()
orders['month_year'] = orders['order_date'].dt.to_period('M').astype(str)

# Calculate order metrics from the order_details which has the total prices
order_metrics = order_details.groupby(['order_id', 'month_year']).agg({
    'item_total': 'sum',
    'customer_id': 'first'
}).reset_index()

# Now calculate monthly metrics
monthly_metrics = order_metrics.groupby('month_year').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('item_total', 'sum'),
    avg_order_value=('item_total', 'mean'),
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
print(f"Total Customers: {customers['customer_id'].nunique():,}")
print(f"Total Orders: {orders['order_id'].nunique():,}")
print(f"Total Revenue: ${order_details['item_total'].sum():,.2f}")
print(f"Average Order Value: ${order_details.groupby('order_id')['item_total'].sum().mean():,.2f}")
print(f"Average Items per Order: {order_details.groupby('order_id')['quantity'].sum().mean():.1f}")
top_category = order_details['category'].mode()[0] if not order_details['category'].empty else 'N/A'
print(f"Top Selling Category: {top_category}")
print(f"\nVisualization files saved to: {OUTPUT_DIR.absolute()}")

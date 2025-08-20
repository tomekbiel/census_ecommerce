import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime

# Set the style for the plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Constants
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the data
print("Loading data...")
customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=['join_date'])
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=['order_date'])
products = pd.read_csv(DATA_DIR / "products.csv")
order_items = pd.read_csv(DATA_DIR / "order_items.csv")

# Data Preparation
print("Preparing data...")
# Merge order items with products
order_details = order_items.merge(products[['product_id', 'name', 'category', 'price']], 
                                 on='product_id', how='left')

# Merge orders with customers
orders = orders.merge(customers[['customer_id', 'join_date', 'region']], on='customer_id', how='left')

# Add year, month, and day of week to orders
orders['order_year'] = orders['order_date'].dt.year
orders['order_month'] = orders['order_date'].dt.month
orders['order_day_of_week'] = orders['order_date'].dt.dayofweek
orders['order_day_name'] = orders['order_date'].dt.day_name()

# Add month-year for time series
orders['month_year'] = orders['order_date'].dt.to_period('M').astype(str)

# Calculate order metrics
monthly_metrics = orders.groupby('month_year').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('total_amount', 'sum'),
    avg_order_value=('total_amount', 'mean'),
    unique_customers=('customer_id', 'nunique')
).reset_index()

# Convert month_year back to datetime for plotting
monthly_metrics['month_year'] = pd.to_datetime(monthly_metrics['month_year'])

# Plot 1: Monthly Sales Trend
print("Creating monthly sales trend plot...")
plt.figure(figsize=(14, 7))
plt.plot(monthly_metrics['month_year'], monthly_metrics['total_revenue'], 
         marker='o', linewidth=2, markersize=8)
plt.title('Monthly Revenue Trend', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monthly_revenue_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Average Order Value Over Time
print("Creating average order value plot...")
plt.figure(figsize=(14, 7))
plt.plot(monthly_metrics['month_year'], monthly_metrics['avg_order_value'], 
         marker='o', color='orange', linewidth=2, markersize=8)
plt.title('Average Order Value Over Time', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Order Value ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'avg_order_value_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Sales by Day of Week
print("Creating sales by day of week plot...")
dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dow_sales = orders.groupby(['order_day_of_week', 'order_day_name'])['total_amount'].sum().reset_index()
dow_sales = dow_sales.sort_values('order_day_of_week')

dow_sales['order_day_name'] = pd.Categorical(dow_sales['order_day_name'], 
                                           categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                      'Friday', 'Saturday', 'Sunday'],
                                           ordered=True)

dow_sales = dow_sales.sort_values('order_day_name')

plt.figure(figsize=(12, 6))
sns.barplot(x='order_day_name', y='total_amount', data=dow_sales, 
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Total Sales by Day of Week', fontsize=16, pad=20)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sales_by_day_of_week.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Top Selling Categories
print("Creating top categories plot...")
category_sales = order_details.groupby('category')['item_total'].sum().reset_index()
category_sales = category_sales.sort_values('item_total', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='item_total', y='category', data=category_sales, palette='viridis')
plt.title('Top Selling Categories by Revenue', fontsize=16, pad=20)
plt.xlabel('Total Revenue ($)', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Customer Cohort Analysis
print("Creating customer cohort analysis...")
# Create a period based on the order date
orders['order_period'] = orders['order_date'].dt.to_period('M')

# Create a cohort based on the first purchase
df_cohort = orders.copy()
df_cohort['cohort'] = df_cohort.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
df_cohort['cohort_index'] = (df_cohort.order_period - df_cohort.cohort).apply(lambda x: x.n)

# Count the number of unique customers in each group
cohort_data = df_cohort.groupby(['cohort', 'cohort_index'])['customer_id'].nunique().reset_index()
cohort_counts = cohort_data.pivot(index='cohort', columns='cohort_index', values='customer_id')

# Calculate retention rates
cohort_sizes = cohort_counts.iloc[:, 0]
retention_matrix = cohort_counts.divide(cohort_sizes, axis=0)
retention_matrix = retention_matrix.round(3) * 100

# Plot the retention matrix
plt.figure(figsize=(14, 8))
plt.title('Cohort Analysis - Customer Retention Rates (%)', fontsize=16, pad=20)
sns.heatmap(retention_matrix, annot=True, fmt='.0f', cmap='YlGnBu', 
            mask=retention_matrix.isnull(), vmin=0, vmax=100)
plt.xlabel('Cohort Index (months since first purchase)', fontsize=12)
plt.ylabel('Cohort Month', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'customer_retention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Customer Acquisition by Month
print("Creating customer acquisition plot...")
customer_acquisition = customers.set_index('join_date').resample('M')['customer_id'].nunique()

plt.figure(figsize=(14, 7))
customer_acquisition.plot(kind='bar', color='teal')
plt.title('New Customer Acquisition by Month', fontsize=16, pad=20)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of New Customers', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'customer_acquisition.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Order Value Distribution
print("Creating order value distribution plot...")
plt.figure(figsize=(12, 6))
sns.histplot(orders['total_amount'], bins=50, kde=True, color='purple')
plt.title('Distribution of Order Values', fontsize=16, pad=20)
plt.xlabel('Order Value ($)', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.axvline(orders['total_amount'].mean(), color='red', linestyle='--', 
            label=f'Mean: ${orders["total_amount"].mean():.2f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'order_value_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 8: Top Products by Revenue
print("Creating top products plot...")
top_products = order_details.groupby(['product_id', 'name'])['item_total'].sum().reset_index()
top_products = top_products.sort_values('item_total', ascending=False).head(10)

top_products['product_name'] = top_products['product_id'] + ' - ' + top_products['name'].str[:30] + '...'

plt.figure(figsize=(12, 8))
sns.barplot(x='item_total', y='product_name', data=top_products, palette='viridis')
plt.title('Top 10 Products by Revenue', fontsize=16, pad=20)
plt.xlabel('Total Revenue ($)', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_products.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 9: Customer Region Distribution
print("Creating customer region distribution plot...")
region_dist = orders.groupby('region')['customer_id'].nunique().reset_index()
region_dist = region_dist.sort_values('customer_id', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='customer_id', y='region', data=region_dist, palette='viridis')
plt.title('Number of Customers by Region', fontsize=16, pad=20)
plt.xlabel('Number of Customers', fontsize=12)
plt.ylabel('Region', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'customer_region_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 10: Monthly Active Customers
print("Creating monthly active customers plot...")
monthly_active = orders.groupby('month_year')['customer_id'].nunique().reset_index()
monthly_active['month_year'] = pd.to_datetime(monthly_active['month_year'])

plt.figure(figsize=(14, 7))
plt.plot(monthly_active['month_year'], monthly_active['customer_id'], 
         marker='o', color='green', linewidth=2, markersize=8)
plt.title('Monthly Active Customers', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Active Customers', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monthly_active_customers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization complete! All plots have been saved to {OUTPUT_DIR}")

# Generate a summary report
print("\n=== E-commerce Data Analysis Summary ===")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Customers: {customers['customer_id'].nunique():,}")
print(f"Total Orders: {orders['order_id'].nunique():,}")
print(f"Total Revenue: ${orders['total_amount'].sum():,.2f}")
print(f"Average Order Value: ${orders['total_amount'].mean():.2f}")
print(f"Top Category by Revenue: {category_sales.iloc[0]['category']} (${category_sales.iloc[0]['item_total']:,.2f})")
print("\nVisualization files created:")
for plot_file in OUTPUT_DIR.glob('*.png'):
    print(f"- {plot_file.name}")

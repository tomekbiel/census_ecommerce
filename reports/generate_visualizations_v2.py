import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

2# Set up file paths
data_dir = os.path.join('..', 'data', 'synthetic')
output_dir = os.path.join('reports', 'images')
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'), parse_dates=['order_date'])

# Extract year from order_date for analysis
orders['year'] = orders['order_date'].dt.year

# Set style for better-looking plots
plt.style.use('ggplot')

# 2. Yearly Order Distribution
print("Generating yearly order distribution...")
yearly_orders = orders['year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
yearly_orders.plot(kind='bar', color='skyblue')
plt.title('Yearly Order Distribution')
plt.xlabel('Year')
plt.ylabel('Number of Orders')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yearly_orders_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

# Calculate total orders per customer
customer_orders = orders2['customer_id'].value_counts().reset_index()
customer_orders.columns = ['customer_id', 'total_orders']
customers = pd.merge(customers2, customer_orders, on='customer_id', how='left')

# 3. Customer Loyalty Analysis
print("Generating customer loyalty analysis...")
plt.figure(figsize=(12, 6))
plt.scatter(customers['loyalty_score'], customers['total_orders'], alpha=0.6, c='green')
plt.title('Customer Loyalty vs Total Orders')
plt.xlabel('Loyalty Score')
plt.ylabel('Total Orders')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'customer_loyalty_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4. Loyalty Discount Analysis
print("Generating loyalty discount analysis...")
orders_with_loyalty = pd.merge(orders2, customers[['customer_id', 'loyalty_score']], on='customer_id')

plt.figure(figsize=(12, 6))
plt.scatter(orders_with_loyalty['loyalty_score'], 
            orders_with_loyalty['loyalty_discount'],import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

# Set style for better-looking plots
plt.style.use('ggplot')

# File paths
data_dir = os.path.join('..', 'data', 'synthetic')
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Load data with updated file names
print("Loading data...")
customers = pd.read_csv(os.path.join(data_dir, 'customers2.csv'))
orders = pd.read_csv(os.path.join(data_dir, 'orders2.csv'), parse_dates=['order_date'])
products = pd.read_csv(os.path.join(data_dir, 'products2.csv'))
order_items = pd.read_csv(os.path.join(data_dir, 'order_items2.csv'))

# Process dates
orders['year'] = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders['year_month'] = orders['order_date'].dt.to_period('M')

# 1. Monthly Revenue Trend
print("Generating monthly revenue trend...")
monthly_revenue = orders.groupby('year_month')['total'].sum().reset_index()
monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)

plt.figure(figsize=(14, 6))
plt.plot(monthly_revenue['year_month'], monthly_revenue['total'], marker='o')
plt.title('Monthly Revenue Trend (2018-2024)')
plt.xlabel('Month')
plt.ylabel('Revenue (€)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_revenue_trend_v2.png')
            alpha=0.6, c='purple')
plt.title('Loyalty Score vs Loyalty Discount')
plt.xlabel('Loyalty Score')
plt.ylabel('Loyalty Discount (€)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loyalty_discount_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5. Top Selling Products
print("Generating top selling products...")
product_sales = order_items.groupby('product_id')['quantity'].sum().sort_values(ascending=False).head(10)
product_names = products.set_index('product_id')['name'].to_dict()
product_sales.index = product_sales.index.map(lambda x: product_names.get(x, f'Product {x}'))

plt.figure(figsize=(12, 6))
product_sales.plot(kind='barh', color='teal')
plt.title('Top 10 Best Selling Products')
plt.xlabel('Total Quantity Sold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_products_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All visualizations have been generated in the 'images' directory with '_v2' suffix.")

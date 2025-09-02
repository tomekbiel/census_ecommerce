import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

# Set style for better-looking plots
plt.style.use('ggplot')  # Using built-in 'ggplot' style instead of seaborn

# File paths
data_dir = os.path.join('..', 'data', 'synthetic')
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
customers = pd.read_csv(os.path.join(data_dir, 'customers.csv'))
orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'), parse_dates=['order_date'])
products = pd.read_csv(os.path.join(data_dir, 'products.csv'))

# Process dates
orders['year'] = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders['year_month'] = orders['order_date'].dt.to_period('M')

# 1. Monthly Revenue Trend
print("Generating monthly revenue trend...")
monthly_revenue = orders.groupby('year_month')['total_amount'].sum().reset_index()
monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)

plt.figure(figsize=(14, 6))
plt.plot(monthly_revenue['year_month'], monthly_revenue['total_amount'], marker='o')
plt.title('Monthly Revenue Trend (2018-2024)')
plt.xlabel('Month')
plt.ylabel('Revenue (€)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_revenue_trend.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2. Yearly Order Distribution
print("Generating yearly order distribution...")
yearly_orders = orders['year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
yearly_orders.plot(kind='bar', color='skyblue')
plt.title('Yearly Order Distribution')
plt.xlabel('Year')
plt.ylabel('Number of Orders')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yearly_orders.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. Order Value Distribution
print("Generating order value distribution...")
plt.figure(figsize=(10, 6))

# Calculate the mean order value for the label
mean_order_value = orders['total_amount'].mean()
total_revenue = orders['total_amount'].sum()
total_orders = len(orders)

# Create histogram
n, bins, patches = plt.hist(orders['total_amount'], bins=50, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.title('Order Value Distribution')
plt.xlabel('Order Value (€)', fontname='Arial')
plt.ylabel('Number of Orders', fontname='Arial')
plt.grid(alpha=0.3)

# Add mean line with proper Euro symbol
plt.axvline(mean_order_value, color='red', linestyle='dashed', linewidth=1, 
            label=f'Mean: {chr(8364)}{mean_order_value:,.2f}')
plt.legend(prop={'family': 'Arial'})

# Set x-axis limits to better show the distribution (excluding extreme outliers)
upper_limit = max(orders['total_amount'].quantile(0.99), mean_order_value * 3)
plt.xlim(0, upper_limit)

# Set font for all text elements
for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
             plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
    item.set_fontname('Arial')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'order_value_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Print statistics for verification with proper encoding
print(f"Average Order Value: {chr(8364)}{mean_order_value:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Revenue: {chr(8364)}{total_revenue:,.2f}")

# Write metrics to a file for the dashboard
with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
    import json
    metrics = {
        'total_customers': 15500,
        'total_orders': total_orders,
        'total_revenue': f"{chr(8364)}{total_revenue:,.2f}",
        'avg_order_value': f"{chr(8364)}{mean_order_value:,.2f}"
    }
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# 4. Payment Method Distribution
print("Generating payment method distribution...")
payment_dist = orders['payment_method'].value_counts()

plt.figure(figsize=(10, 6))
payment_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                 colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Payment Method Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'payment_methods.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5. Order Status Distribution
print("Generating order status distribution...")
status_dist = orders['status'].value_counts()

plt.figure(figsize=(10, 6))
status_dist.plot(kind='bar', color='#8B5A2B')
plt.title('Order Status Distribution')
plt.xlabel('Status')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'order_status.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations generated successfully in the 'images' folder.")

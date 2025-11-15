import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime

# Set up the style for the plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['figure.dpi'] = 100

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'synthetic', 'orders2.csv')
output_dir = os.path.join(current_dir, 'reports')
os.makedirs(output_dir, exist_ok=True)

# Load and prepare the data
print("Loading and preparing data...")
df = pd.read_csv(data_path, parse_dates=['order_date'])

# Add time-based features
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['year_month'] = df['order_date'].dt.to_period('M')
df['day_of_week'] = df['order_date'].dt.day_name()
df['hour'] = df['order_date'].dt.hour

# Create a color palette
colors = sns.color_palette('viridis', 5)

# 1. Sales Trend Analysis
print("\nGenerating sales trend analysis...")
plt.figure(figsize=(16, 8))

# Monthly sales trend
monthly_sales = df.groupby('year_month')['total'].sum().reset_index()
monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)

sns.lineplot(data=monthly_sales, x='year_month', y='total', 
             marker='o', linewidth=2.5, color=colors[0])
plt.title('Monthly Sales Trend', fontsize=16, pad=20)
plt.xlabel('Month', fontsize=12, labelpad=10)
plt.ylabel('Total Sales ($)', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_sales_trend.png'), dpi=120)
plt.close()

# 2. Payment Method Analysis
print("Analyzing payment methods...")
plt.figure(figsize=(14, 6))

payment_stats = df.groupby('payment_method').agg({
    'total': ['count', 'sum', 'mean']
}).round(2).sort_values(('total', 'sum'), ascending=False)

# Plot payment method distribution
plt.subplot(1, 2, 1)
payment_counts = df['payment_method'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, wedgeprops=dict(width=0.4))
plt.title('Order Distribution by Payment Method', pad=20)

# Plot average order value by payment method
plt.subplot(1, 2, 2)
sns.barplot(data=df, x='payment_method', y='total', 
            estimator=np.mean, errorbar=None, palette=colors)
plt.title('Average Order Value by Payment Method', pad=20)
plt.xlabel('Payment Method', labelpad=10)
plt.ylabel('Average Order Value ($)', labelpad=10)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'payment_method_analysis.png'), dpi=120)
plt.close()

# 3. Customer Analysis
print("Analyzing customer behavior...")
customer_orders = df.groupby('customer_id').agg({
    'order_id': 'count',
    'total': 'sum',
    'order_date': ['min', 'max']
}).reset_index()

customer_orders.columns = ['customer_id', 'order_count', 'total_spent', 'first_order', 'last_order']
customer_orders['avg_order_value'] = customer_orders['total_spent'] / customer_orders['order_count']

plt.figure(figsize=(16, 6))

# Customer order frequency
plt.subplot(1, 2, 1)
sns.histplot(customer_orders['order_count'], bins=30, kde=True, color=colors[2])
plt.title('Distribution of Orders per Customer', pad=20)
plt.xlabel('Number of Orders', labelpad=10)
plt.ylabel('Number of Customers', labelpad=10)

# Customer lifetime value
plt.subplot(1, 2, 2)
sns.histplot(customer_orders['total_spent'], bins=30, kde=True, color=colors[3])
plt.title('Distribution of Customer Lifetime Value', pad=20)
plt.xlabel('Total Amount Spent ($)', labelpad=10)
plt.ylabel('Number of Customers', labelpad=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'customer_analysis.png'), dpi=120)
plt.close()

# 4. Time-based Analysis
print("Performing time-based analysis...")
plt.figure(figsize=(16, 6))

# Hourly order distribution
plt.subplot(1, 2, 1)
hourly_orders = df['hour'].value_counts().sort_index()
sns.barplot(x=hourly_orders.index, y=hourly_orders.values, color=colors[0])
plt.title('Order Distribution by Hour of Day', pad=20)
plt.xlabel('Hour of Day', labelpad=10)
plt.ylabel('Number of Orders', labelpad=10)

# Day of week order distribution
plt.subplot(1, 2, 2)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_orders = df['day_of_week'].value_counts().reindex(day_order)
sns.barplot(x=dow_orders.index, y=dow_orders.values, color=colors[1])
plt.title('Order Distribution by Day of Week', pad=20)
plt.xlabel('Day of Week', labelpad=10)
plt.ylabel('Number of Orders', labelpad=10)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_analysis.png'), dpi=120)
plt.close()

# 5. Order Value Analysis
print("Analyzing order values...")
plt.figure(figsize=(16, 6))

# Distribution of order values
plt.subplot(1, 2, 1)
sns.histplot(df['total'], bins=50, kde=True, color=colors[4])
plt.title('Distribution of Order Values', pad=20)
plt.xlabel('Order Value ($)', labelpad=10)
plt.ylabel('Number of Orders', labelpad=10)

# Boxplot of order values by payment method
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='payment_method', y='total', showfliers=False, palette=colors)
plt.title('Order Value Distribution by Payment Method', pad=20)
plt.xlabel('Payment Method', labelpad=10)
plt.ylabel('Order Value ($)', labelpad=10)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'order_value_analysis.png'), dpi=120)
plt.close()

# Generate a summary report
print("\n=== Analysis Summary ===")
print(f"Total Orders: {len(df):,}")
print(f"Total Revenue: ${df['total'].sum():,.2f}")
print(f"Average Order Value: ${df['total'].mean():.2f}")
print(f"Number of Unique Customers: {df['customer_id'].nunique():,}")
print(f"Most Common Payment Method: {df['payment_method'].mode()[0]}")
print(f"Date Range: {df['order_date'].min().strftime('%Y-%m-%d')} to {df['order_date'].max().strftime('%Y-%m-%d')}")

print("\nVisualizations have been saved to:", os.path.abspath(output_dir))

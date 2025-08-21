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
top_cats = category_sales.head(10)
sns.barplot(x='item_total', y='category', data=top_cats, palette='viridis')
plt.title('Top 10 Categories by Revenue', fontsize=14, pad=20)
plt.xlabel('Total Revenue ($)', fontsize=12, labelpad=10)
plt.ylabel('Category', fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_categories.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate HTML report
print("Generating HTML report...")
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>E-commerce Data Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        .visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            padding: 20px;
        }}
        .visualization h2 {{
            color: #3498db;
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #eee;
        }}
        .description {{
            margin: 15px 0;
            color: #555;
        }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            min-width: 200px;
            margin: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 5px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>E-commerce Data Visualizations</h1>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{customers:,.0f}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        <div class="metric">
            <div class="metric-value">${revenue:,.2f}</div>
            <div class="metric-label">Total Revenue</div>
        </div>
        <div class="metric">
            <div class="metric-value">{orders:,.0f}</div>
            <div class="metric-label">Total Orders</div>
        </div>
        <div class="metric">
            <div class="metric-value">${aov:,.2f}</div>
            <div class="metric-label">Avg. Order Value</div>
        </div>
    </div>
    
    <div class="visualization">
        <h2>1. Monthly Revenue Trend</h2>
        <div class="description">
            This chart shows the monthly revenue trend over time, highlighting seasonality and growth patterns.
        </div>
        <img src="monthly_revenue_trend.png" alt="Monthly Revenue Trend">
    </div>
    
    <div class="visualization">
        <h2>2. Customer Retention by Cohort</h2>
        <div class="description">
            This heatmap shows the retention rate of customers over time, grouped by their initial purchase month (cohort).
        </div>
        <img src="customer_retention_heatmap.png" alt="Customer Retention by Cohort">
    </div>
    
    <div class="visualization">
        <h2>3. Top Categories by Revenue</h2>
        <div class="description">
            This chart displays the top 10 product categories by total revenue.
        </div>
        <img src="top_categories.png" alt="Top Categories by Revenue">
    </div>
</body>
</html>
""".format(
    customers=customers['customer_id'].nunique(),
    revenue=orders['total_amount'].sum(),
    orders=orders['order_id'].nunique(),
    aov=orders['total_amount'].mean()
)

with open(OUTPUT_DIR / 'view_visualizations.html', 'w') as f:
    f.write(html_content)

print(f"Visualizations saved to {OUTPUT_DIR}")
print("\nKey Metrics:")
print(f"Total Customers: {customers['customer_id'].nunique():,}")
print(f"Total Orders: {orders['order_id'].nunique():,}")
print(f"Total Revenue: ${orders['total_amount'].sum():,.2f}")
print(f"Average Order Value: ${orders['total_amount'].mean():.2f}")
print(f"Top Category by Revenue: {category_sales.iloc[0]['category']} (${category_sales.iloc[0]['item_total']:,.2f})")
print(f"\nVisualization files saved to: {OUTPUT_DIR.absolute()}")
print("Open 'view_visualizations.html' in your browser to view the dashboard.")

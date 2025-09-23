# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
sns.set_palette("viridis")
plt.style.use('seaborn-v0_8')

# Load the generated data
def load_ecommerce_data():
    data_dir = "C:/python/census_ecommerce/data/synthetic/"
    customers = pd.read_csv(f"{data_dir}customers.csv")
    products = pd.read_csv(f"{data_dir}products.csv")
    orders = pd.read_csv(f"{data_dir}orders.csv")
    order_items = pd.read_csv(f"{data_dir}order_items.csv")
    
    # Convert date columns
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['year_month'] = orders['order_date'].dt.to_period('M')
    
    # Merge order items with products
    order_details = pd.merge(order_items, products, on='product_id')
    
    return customers, products, orders, order_items, order_details

def plot_sales_trends(orders, order_details):
    """Plot sales trends over time"""
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Monthly Revenue Trend', 'Monthly Order Count'))
    
    # Monthly revenue
    print("Columns in orders:", orders.columns.tolist())  # Debug print
    monthly_revenue = orders.groupby('year_month')['total'].sum().reset_index()
    monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)
    
    fig.add_trace(
        go.Bar(x=monthly_revenue['year_month'], y=monthly_revenue['total'],
               name='Monthly Revenue'),
        row=1, col=1
    )
    
    # Monthly order count
    monthly_orders = orders.groupby('year_month').size().reset_index(name='order_count')
    monthly_orders['year_month'] = monthly_orders['year_month'].astype(str)
    
    fig.add_trace(
        go.Scatter(x=monthly_orders['year_month'], y=monthly_orders['order_count'],
                  name='Order Count', mode='lines+markers'),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text='Sales Performance Over Time',
        showlegend=True,
        height=800
    )
    
    fig.write_html("sales_trends.html")
    return fig

def plot_customer_analysis(customers, orders):
    """Analyze customer behavior and demographics"""
    # Customer acquisition over time
    customers['join_date'] = pd.to_datetime(customers['join_date'])
    customer_acquisition = customers.groupby(pd.Grouper(key='join_date', freq='ME')).size().reset_index(name='new_customers')
    
    # Customer lifetime value
    customer_value = orders.groupby('customer_id')['total'].sum().reset_index()
    customer_value.columns = ['customer_id', 'lifetime_value']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'New Customers by Month',
            'Customer Lifetime Value Distribution',
            'Repeat Purchase Rate',
            'Customer Geography Distribution'
        ),
        specs=[[{"type": "bar"}, {"type": "box"}],
              [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Plot 1: New Customers
    fig.add_trace(
        go.Bar(x=customer_acquisition['join_date'], 
               y=customer_acquisition['new_customers'],
               name='New Customers'),
        row=1, col=1
    )
    
    # Plot 2: Customer LTV Distribution
    fig.add_trace(
        go.Box(y=customer_value['lifetime_value'],
               name='Lifetime Value',
               boxpoints=False),
        row=1, col=2
    )
    
    # Plot 3: Repeat Purchase Rate
    order_counts = orders['customer_id'].value_counts().reset_index()
    order_counts.columns = ['customer_id', 'order_count']
    repeat_customers = (order_counts['order_count'] > 1).mean() * 100
    
    fig.add_trace(
        go.Pie(labels=['Repeat Customers', 'One-time Buyers'],
               values=[repeat_customers, 100 - repeat_customers],
               name="Repeat Purchase Rate"),
        row=2, col=1
    )
    
    # Plot 4: Customer Geography
    state_dist = customers['state'].value_counts().reset_index()
    state_dist.columns = ['state', 'count']
    
    fig.add_trace(
        go.Bar(x=state_dist['state'],
               y=state_dist['count'],
               name='Customers by State'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='Customer Analysis Dashboard',
        showlegend=True,
        height=1000
    )
    
    fig.write_html("customer_analysis.html")
    return fig

def plot_product_analysis(products, order_details):
    """Analyze product performance"""
    # Product category performance
    category_sales = order_details.groupby('category')['total_price'].sum().sort_values(ascending=False).reset_index()
    
    # Top selling products
    top_products = order_details.groupby(['product_id', 'name'])['quantity'].sum().reset_index()
    top_products = top_products.sort_values('quantity', ascending=False).head(10)
    
    # Price distribution by category
    price_dist = order_details[['category', 'price']].drop_duplicates()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue by Category',
            'Top Selling Products',
            'Price Distribution by Category',
            'Inventory Status'
        )
    )
    
    # Plot 1: Revenue by Category
    fig.add_trace(
        go.Bar(x=category_sales['category'],
               y=category_sales['total_price'],
               name='Revenue by Category'),
        row=1, col=1
    )
    
    # Plot 2: Top Selling Products
    fig.add_trace(
        go.Bar(x=top_products['name'],
               y=top_products['quantity'],
               name='Top Products'),
        row=1, col=2
    )
    
    # Plot 3: Price Distribution by Category
    for category in price_dist['category'].unique():
        fig.add_trace(
            go.Box(y=price_dist[price_dist['category'] == category]['price'],
                   name=category,
                   showlegend=False),
            row=2, col=1
        )
    
    # Plot 4: Inventory Status
    inventory_status = products[products['stock_quantity'] < 50].sort_values('stock_quantity')
    fig.add_trace(
        go.Bar(x=inventory_status['name'],
               y=inventory_status['stock_quantity'],
               name='Low Inventory'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='Product Performance Dashboard',
        showlegend=True,
        height=1000
    )
    
    fig.write_html("product_analysis.html")
    return fig

def plot_marketing_performance(orders, customers):
    """Analyze marketing channel performance"""
    # Merge orders with customers (add dummy marketing channel if not exists)
    if 'marketing_channel' not in customers.columns:
        customers['marketing_channel'] = np.random.choice(
            ['Organic', 'Email', 'Social', 'Referral', 'Paid'], 
            size=len(customers),
            p=[0.3, 0.2, 0.2, 0.15, 0.15]
        )
    
    marketing_data = pd.merge(orders, customers[['customer_id', 'marketing_channel']], on='customer_id')
    
    # Marketing channel performance
    channel_performance = marketing_data.groupby('marketing_channel')['total'].agg(['sum', 'count', 'mean']).reset_index()
    channel_performance.columns = ['Channel', 'Total Revenue', 'Order Count', 'AOV']
    
    # Customer acquisition cost (simplified)
    cac_data = {
        'Channel': ['Organic', 'Email', 'Social', 'Referral', 'Paid'],
        'CAC': [0, 5, 15, 10, 25]  # Example values
    }
    cac_df = pd.DataFrame(cac_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue by Marketing Channel',
            'Customer Acquisition Cost',
            'Channel Conversion Rate',
            'Customer Lifetime Value by Channel'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
              [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Plot 1: Revenue by Channel
    fig.add_trace(
        go.Bar(x=channel_performance['Channel'],
               y=channel_performance['Total Revenue'],
               name='Revenue by Channel'),
        row=1, col=1
    )
    
    # Plot 2: Customer Acquisition Cost
    fig.add_trace(
        go.Bar(x=cac_df['Channel'],
               y=cac_df['CAC'],
               name='Customer Acquisition Cost'),
        row=1, col=2
    )
    
    # Plot 3: Conversion Rate (simplified)
    conversion_rate = channel_performance.copy()
    conversion_rate['Conversion Rate'] = (conversion_rate['Order Count'] / conversion_rate['Order Count'].sum()) * 100
    
    # Create a separate figure for the pie chart
    pie_fig = go.Figure(
        go.Pie(
            labels=conversion_rate['Channel'],
            values=conversion_rate['Conversion Rate'],
            name='Conversion Rate',
            textinfo='percent+label'
        )
    )
    pie_fig.update_layout(title_text='Channel Conversion Rate')
    
    # Add the pie chart to the subplot
    fig.add_trace(
        go.Pie(
            labels=conversion_rate['Channel'],
            values=conversion_rate['Conversion Rate'],
            name='Conversion Rate',
            textinfo='percent+label'
        ),
        row=2, col=1
    )
    
    # Plot 4: CLV by Channel (simplified)
    clv = marketing_data.groupby('marketing_channel')['total'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(x=clv['marketing_channel'],
               y=clv['total'],
               name='Average Order Value by Channel'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='Marketing Performance Dashboard',
        showlegend=True,
        height=1000
    )
    
    fig.write_html("marketing_performance.html")
    return fig

def main():
    print("Loading data...")
    customers, products, orders, order_items, order_details = load_ecommerce_data()
    
    print("Generating visualizations...")
    # Generate all visualizations
    plot_sales_trends(orders, order_details)
    plot_customer_analysis(customers, orders)
    plot_product_analysis(products, order_details)
    plot_marketing_performance(orders, customers)
    
    print("Visualizations generated successfully!")
    print("Check the following HTML files:")
    print("- sales_trends.html")
    print("- customer_analysis.html")
    print("- product_analysis.html")
    print("- marketing_performance.html")

if __name__ == "__main__":
    main()
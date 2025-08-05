"""
E-commerce Synthetic Data Generator
----------------------------------
This script generates realistic e-commerce data including customers, products, and orders,
synchronized with macroeconomic indicators from ecommerce_macro_indicators.csv.

The generated data includes:
- 15,000 unique customer records (plus 500 duplicates)
- 20% missing email addresses
- Inconsistent phone number formats
- Purchase data from 2018-2024
- Sales patterns correlated with macroeconomic indicators
"""
import numpy as np
import pandas as pd
from faker import Faker
import random
import uuid
from datetime import datetime
from pathlib import Path

def generate_ecommerce_data(
    macro_file: str = "../data/processed/ecommerce_macro_indicators.csv",
    output_dir: str = "../data/synthetic",
    n_customers: int = 15000,
    n_products: int = 100,
    email_missing_ratio: float = 0.2,
    duplicate_customers: int = 500,
    seed: int = 42
):
    """
    Generate synthetic e-commerce data synchronized with macroeconomic indicators.
    
    Args:
        macro_file: Path to the macro indicators CSV file
        output_dir: Directory to save generated files
        n_customers: Number of unique customers to generate
        n_products: Number of unique products to generate
        email_missing_ratio: Ratio of customers with missing email
        duplicate_customers: Number of duplicate customer records to add
        seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load macro data
    print(f"Loading macro indicators from {macro_file}...")
    try:
        macro_df = pd.read_csv(macro_file)
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        date_range = (macro_df['Date'].min(), macro_df['Date'].max())
        print(f"  Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error loading macro data: {e}")
        return

    # 1. Generate Customers
    print(f"\nGenerating {n_customers} customers (plus {duplicate_customers} duplicates)...")
    customers = []
    for _ in range(n_customers):
        # 20% missing email addresses
        email = fake.email() if random.random() > email_missing_ratio else None
        
        # Inconsistent phone number formats
        phone_format = random.choice([
            fake.phone_number(),
            fake.msisdn(),
            fake.numerify('08#-###-####'),
            fake.numerify('+353 8# ### ####'),
            fake.numerify('(0##) ###-####'),
            fake.numerify('+1 (###) ###-####')
        ])
        
        customers.append({
            "Customer_ID": str(uuid.uuid4()),
            "Name": fake.name(),
            "Email": email,
            "Phone": phone_format,
            "Region": fake.state(),
            "Signup_Date": fake.date_between(
                start_date=date_range[0],  # Align with macro data start
                end_date=date_range[1]     # Align with macro data end
            )
        })
    
    # Add duplicate customers
    if duplicate_customers > 0:
        customers += random.sample(customers, min(duplicate_customers, len(customers)))
    
    customers_df = pd.DataFrame(customers)
    
    # 2. Generate Products
    print(f"Generating {n_products} products...")
    categories = ["Tools", "Furniture", "Plants", "Lighting", "Garden"]
    products = []
    for i in range(n_products):
        price = np.round(np.random.lognormal(mean=3.2, sigma=0.4), 2)
        if random.random() < 0.05:
            price *= 3  # 5% of products are premium
            
        products.append({
            "Product_ID": f"P{i+1:04d}",
            "Name": f"{random.choice(categories)} Item {i+1}",
            "Category": random.choice(categories),
            "Price": max(price, 1.99),
            "Cost_Price": round(price * (1 - np.random.beta(2, 5)), 2)
        })
    
    products_df = pd.DataFrame(products)

    # 3. Generate Orders (synchronized with macro indicators)
    print("Generating orders...")
    orders = []
    order_id = 1
    
    for _, row in macro_df.iterrows():
        date = row['Date']
        
        # Get macro indicators (with fallbacks)
        ecommerce_sales = row.get('Ecommerce_Retail_Sales_Millions', 1000)
        sentiment = row.get('Consumer_Sentiment', 70)
        unemployment = row.get('Unemployment_Rate', 5)
        
        # Calculate number of orders based on economic conditions
        base_orders = int(2000 + (ecommerce_sales / 500))
        economic_factor = max(0.5, 1.5 - (unemployment * 0.1) - ((100 - sentiment) * 0.01))
        n_orders = int(base_orders * economic_factor)
        
        # Generate orders for this date
        for _ in range(n_orders):
            product = random.choice(products)
            customer = random.choice(customers)
            
            # Generate order details
            quantity = max(1, int(np.random.normal(loc=2, scale=1)))
            total_price = product["Price"] * quantity
            profit = round(total_price - (product.get("Cost_Price", 0) * quantity), 2)
            
            orders.append({
                "Order_ID": f"O{order_id:07d}",
                "Customer_ID": customer["Customer_ID"],
                "Order_Date": date.strftime("%Y-%m-%d"),
                "Product_ID": product["Product_ID"],
                "Sales": round(total_price, 2),
                "Quantity": quantity,
                "Profit": profit,
                "Discount": round(np.random.beta(2, 5) * 0.3, 2)  # 0-30% discount
            })
            order_id += 1
    
    orders_df = pd.DataFrame(orders)
    
    # 4. Save to files
    print("\nSaving generated data...")
    output_files = {
        "customers.csv": customers_df,
        "products.csv": products_df,
        "orders.csv": orders_df
    }
    
    for filename, df in output_files.items():
        filepath = Path(output_dir) / filename
        df.to_csv(filepath, index=False)
        print(f"  ✓ {filepath} ({len(df):,} rows)")
    
    # 5. Generate summary
    print("\n✅ Data Generation Complete!")
    print(f"• {len(customers_df):,} customers (including duplicates)")
    print(f"• {len(products_df):,} products")
    print(f"• {len(orders_df):,} orders")
    print(f"• Date range: {orders_df['Order_Date'].min()} to {orders_df['Order_Date'].max()}")
    print(f"\nFiles saved to: {Path(output_dir).resolve()}")

if __name__ == "__main__":
    generate_ecommerce_data()

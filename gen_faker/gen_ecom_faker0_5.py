"""
gen_ecom_faker0_5.py - Enhanced E-commerce Data Generator

This script generates synthetic e-commerce data including customers, products, and orders.
It's based on gen_ecom_faker5.py but with enhanced documentation from gen_ecom_faker0_1.py.

Key Features:
- Generates realistic customer data with demographics
- Creates product catalog with categories and pricing
- Simulates order history with seasonal variations
- Handles data quality aspects like missing values and duplicates
- Supports data generation for specific time periods (2018-2024)

Usage:
    python gen_ecom_faker0_5.py

Outputs:
    - customers.csv: Customer information
    - products.csv: Product catalog
    - orders.csv: Order history
    - order_items.csv: Individual items in each order
"""

# Import the faker5 implementation
from gen_ecom_faker5 import EcommerceDataGenerator

# Add any additional documentation or helper functions here if needed

if __name__ == "__main__":
    print("Starting enhanced e-commerce data generation (v0.5)...")
    generator = EcommerceDataGenerator()
    
    # Generate and save all data
    print("\nGenerating products...")
    products_df = generator.generate_products()
    
    print("\nGenerating customers...")
    customers_df = generator.generate_customers()
    
    print("\nGenerating orders...")
    orders_df, order_items_df = generator.generate_orders(customers_df, products_df)
    
    # Save to CSV
    output_dir = generator.data_dir / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    customers_df.to_csv(output_dir / "customers.csv", index=False)
    products_df.to_csv(output_dir / "products.csv", index=False)
    orders_df.to_csv(output_dir / "orders.csv", index=False)
    order_items_df.to_csv(output_dir / "order_items.csv", index=False)
    
    print(f"\n✅ Data generation complete. Files saved to: {output_dir}")

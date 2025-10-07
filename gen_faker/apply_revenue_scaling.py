from pathlib import Path
import pandas as pd

data_dir = Path("C:/Users/User/PycharmProjects/census_ecommerce/data/synthetic")
orders = pd.read_csv(data_dir / "orders.csv")
order_items = pd.read_csv(data_dir / "order_items.csv")

# Convert dates and extract year
orders['order_date'] = pd.to_datetime(orders['order_date'])
orders['year'] = orders['order_date'].dt.year

# Calculate current 2023 revenue
current_2023_revenue = (
    orders.merge(order_items, on='order_id')
    .query('year == 2023')['total_price']
    .sum()
)

# Calculate scale factor to reach $2.3M target for 2023
target_2023_revenue = 2_300_000
scale_factor = target_2023_revenue / current_2023_revenue

print(f"Current 2023 revenue: ${current_2023_revenue:,.2f}")
print(f"Target 2023 revenue: ${target_2023_revenue:,.2f}")
print(f"Calculated scale factor: {scale_factor:.6f}\n")

# Apply scaling to order items
order_items['total_price'] = order_items['total_price'] * scale_factor

# Calculate and display annual revenue
revenue = (orders.merge(order_items, on='order_id')
                 .groupby('year')['total_price']
                 .sum()
                 .reset_index()
                 .rename(columns={'total_price': 'Revenue (USD)'}))

# Format and display results
pd.set_option('display.float_format', '{:,.2f}'.format)
print("Annual Revenue After Scaling:")
print(revenue)

# Optionally save the scaled data
# order_items.to_csv(data_dir / "order_items_scaled.csv", index=False)
# print(f"\nSaved scaled order items to {data_dir / 'order_items_scaled.csv'}")
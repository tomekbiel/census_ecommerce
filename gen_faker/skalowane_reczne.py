import pandas as pd
from pathlib import Path
from datetime import datetime

# Set up paths
data_dir = Path(r"C:\python\census_ecommerce\data\synthetic")

# Load the data
print("🔍# 1. Load the data...")
print("🔍 Wczytywanie danych...")
orders = pd.read_csv(data_dir / 'orders.csv')
order_items = pd.read_csv(data_dir / 'order_items.csv')

# 2. Convert dates to datetime
print("🗓️ Konwersja dat na format datetime...")
orders['order_date'] = pd.to_datetime(orders['order_date'])

# Create year mapping
print("🗺️ Tworzenie mapowania lat...")
order_year_map = orders.set_index('order_id')['order_date'].dt.year.to_dict()

# Remove 2025 data
print("🗑️ Usuwanie danych z 2025 roku...")
orders = orders[orders['order_date'].dt.year != 2025]
order_items = order_items[order_items['order_id'].isin(orders['order_id'])]

# 3. Define scaling factors
print("\n🎯 Obliczanie parametrów skalujących...")
TARGET_2022 = 2_300_000

# Calculate scaling factors
X = TARGET_2022 / orders[pd.to_datetime(orders['order_date']).dt.year == 2022]['total'].sum()
Y = (TARGET_2022 * 1.05) / orders[pd.to_datetime(orders['order_date']).dt.year == 2023]['total'].sum()  # 5% more than 2022
Z = (TARGET_2022 * 1.05) / orders[pd.to_datetime(orders['order_date']).dt.year == 2024]['total'].sum()  # 5% more than 2022

print(f"Parametry skalujące:")
print(f"X (2018-2022): {X:.6f}")
print(f"Y (2023): {Y:.6f}")
print(f"Z (2024): {Z:.6f}")

# 4. Scale order_items
def scale_row(row):
    year = order_year_map.get(row['order_id'])
    scale = X  # Default for 2018-2022
    if year == 2023:
        scale = Y
    elif year == 2024:
        scale = Z
        
    row['unit_price'] = round(row['unit_price'] * scale, 2)
    row['discount'] = round(row['discount'] * scale, 2)
    row['total_price'] = round((row['quantity'] * row['unit_price']) - row['discount'], 2)
    return row

print("\n🔄 Stosowanie skalowania do order_items...")
order_items = order_items.apply(scale_row, axis=1)

# 5. Process orders
print("\n🔄 Przetwarzanie zamówień...")

# Add year column for scaling
orders['year'] = pd.to_datetime(orders['order_date']).dt.year

# Calculate scaling factor for each order
orders['scale'] = X  # default for 2018-2022
orders.loc[orders['year'] == 2023, 'scale'] = Y
orders.loc[orders['year'] == 2024, 'scale'] = Z

# Calculate new values for orders
orders['subtotal'] = orders['subtotal'] * orders['scale']
orders['tax'] = round(orders['subtotal'] * 0.08, 2)  # 8% podatku
orders['shipping'] = orders['shipping'] * orders['scale']
orders['loyalty_discount'] = orders['discount'] * orders['scale']
orders['total'] = orders['subtotal'] + orders['tax'] + orders['shipping'] - orders['loyalty_discount']

# Round all monetary values to 2 decimal places
monetary_cols = ['subtotal', 'tax', 'shipping', 'loyalty_discount', 'total']
orders[monetary_cols] = orders[monetary_cols].round(2)

# Drop temporary columns
orders = orders.drop(columns=['year', 'scale', 'discount'])

def process_products(products_path, output_path, scale_factors, year_col='created_at'):
    """
    Process products data with the same scaling logic as order_items.
    
    Args:
        products_path: Path to the input products CSV file
        output_path: Path where to save the processed products CSV
        scale_factors: Dictionary with years as keys and scale factors as values
        year_col: Name of the column containing the date to determine the year
    """
    print("\n🔄 Przetwarzanie produktów...")
    
    # Load products data
    products = pd.read_csv(products_path)
    
    # Convert date column to datetime and extract year
    products['year'] = pd.to_datetime(products[year_col]).dt.year
    
    # Apply scaling factors based on year
    products['scale'] = products['year'].map(scale_factors)
    
    # If year is not in scale_factors, use the default (X for 2018-2022)
    products['scale'] = products['scale'].fillna(scale_factors.get('default', 1.0))
    
    # Scale price-related columns
    price_cols = ['price', 'cost']
    if 'original_price' in products.columns:
        price_cols.append('original_price')
    
    for col in price_cols:
        products[col] = round(products[col] * products['scale'], 2)
    
    # Drop temporary columns
    products = products.drop(columns=['year', 'scale'])
    
    # Save to CSV
    products.to_csv(output_path, index=False)
    print(f"✅ Zapisano przetworzone produkty do: {output_path}")

# 6. Save the results
print("\n💾 Zapisuję przeskalowane dane...")
output_orders = data_dir / 'orders2.csv'
output_order_items = data_dir / 'order_items2.csv'
output_products = data_dir / 'products2.csv'  # New output file for products

# Reorder columns to match original structure (with loyalty_discount instead of discount)
orders = orders[['order_id', 'customer_id', 'order_date', 'status', 'payment_method', 
                'subtotal', 'tax', 'shipping', 'loyalty_discount', 'total']]

# Save to CSV without index
orders.to_csv(output_orders, index=False)
order_items.to_csv(output_order_items, index=False)

# Process products with the same scaling factors
scale_factors = {
    'default': X,  # 2018-2022
    2023: Y,
    2024: Z
}
process_products(
    products_path=data_dir / 'products.csv',
    output_path=output_products,
    scale_factors=scale_factors
)

print(f"\n✅ ZAPISANO NOWE PLIKI:")
print(f"📁 {output_orders}")
print(f"📁 {output_order_items}")
print(f"📁 {output_products}")

# Validation
print("\n📊 WERYFIKACJA WYNIKÓW:")
try:
    # Group by year and sum total_price for order_items
    validation_items = order_items.merge(
        orders[['order_id', 'order_date']],
        on='order_id',
        how='left'
    )
    validation_items['year'] = pd.to_datetime(validation_items['order_date']).dt.year
    
    # Calculate yearly totals for order_items
    yearly_item_totals = validation_items.groupby('year')['total_price'].sum()
    
    # Calculate yearly totals for orders
    orders['year'] = pd.to_datetime(orders['order_date']).dt.year
    yearly_order_totals = orders.groupby('year')['total'].sum()
    
    print("\nSuma total_price z order_items2.csv:")
    for year in sorted(yearly_item_totals.index):
        print(f"  {year}: ${yearly_item_totals[year]:,.2f}")
    
    print("\nSuma total z orders2.csv:")
    for year in sorted(yearly_order_totals.index):
        print(f"  {year}: ${yearly_order_totals[year]:,.2f}")
    
    # Check if sums match (should be equal or very close due to rounding)
    for year in yearly_item_totals.index:
        item_total = yearly_item_totals[year]
        order_total = yearly_order_totals.get(year, 0)
        diff = abs(item_total - order_total)
        if diff > 0.01:  # Allow for small rounding differences
            print(f"  Uwaga: Różnica w sumach dla roku {year}: ${diff:,.2f}")
    
    # Drop the temporary year column
    orders = orders.drop(columns=['year'])
        
except Exception as e:
    print(f"  Uwaga: Wystąpił błąd podczas weryfikacji wyników: {str(e)}")

print("\n🎉 Przeskalowanie zakończone pomyślnie!")
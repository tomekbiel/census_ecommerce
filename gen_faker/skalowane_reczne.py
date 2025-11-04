import pandas as pd
from pathlib import Path

# Set up paths
data_dir = Path(r"C:\Users\User\PycharmProjects\census_ecommerce\data\synthetic")

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

# 6. Save the results
print("\n💾 Zapisuję przeskalowane dane...")
output_orders = data_dir / 'orders2.csv'
output_order_items = data_dir / 'order_items2.csv'

# Reorder columns to match original structure (with loyalty_discount instead of discount)
orders = orders[['order_id', 'customer_id', 'order_date', 'status', 'payment_method', 
                'subtotal', 'tax', 'shipping', 'loyalty_discount', 'total']]

# Save to CSV without index
orders.to_csv(output_orders, index=False)
order_items.to_csv(output_order_items, index=False)

print(f"\n✅ ZAPISANO NOWE PLIKI:")
print(f"📁 {output_orders}")
print(f"📁 {output_order_items}")

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
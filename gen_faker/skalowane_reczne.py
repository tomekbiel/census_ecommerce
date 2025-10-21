import pandas as pd
from pathlib import Path

# ✅ UŻYJ TEJ ŚCIEŻKI KTÓRĄ MASZ
data_dir = Path(r"C:\Users\User\PycharmProjects\census_ecommerce\data\synthetic")

# Wczytaj oryginalne dane z pełnymi ścieżkami
orders = pd.read_csv(data_dir / 'orders.csv')
order_items = pd.read_csv(data_dir / 'order_items.csv')

print("✅ Pliki wczytane poprawnie!")
print(f"Orders: {len(orders):,} wierszy")
print(f"Order items: {len(order_items):,} wierszy")

# Konwersja daty
orders['order_date'] = pd.to_datetime(orders['order_date'])
orders['year'] = orders['order_date'].dt.year

print("\n📊 ORYGINALNY PRZYCHÓD:")
original_revenue = orders.groupby('year')['total'].sum()
print(original_revenue)

# ✅ USUŃ 2025
orders = orders[orders['year'] != 2025]
order_items = order_items[order_items['order_id'].isin(orders['order_id'])]

print(f"\n🗑️ Usunięto zamówienia z 2025 roku")

# ✅ PRZESKALOWANIE
# Skala dla 2018-2023: zmniejsz do ~$2.3M w 2023
target_2023 = 2_300_000
current_2023 = orders[orders['year'] == 2023]['total'].sum()
scale_2018_2023 = target_2023 / current_2023

# Skala dla 2024 (zachowaj proporcje do 2023)
current_2024 = orders[orders['year'] == 2024]['total'].sum()
target_2024 = target_2023 * 0.95  # 5% mniej niż 2023
scale_2024 = target_2024 / current_2024

print(f"\n🎯 SKALE SKALOWANIA:")
print(f"2018-2023: × {scale_2018_2023:.4f}")
print(f"2024: × {scale_2024:.4f}")

# Zastosuj skalowanie do orders
monetary_cols = ['subtotal', 'tax', 'shipping', 'discount', 'total']

for idx, row in orders.iterrows():
    if row['year'] <= 2023:
        scale = scale_2018_2023
    else:  # 2024
        scale = scale_2024

    for col in monetary_cols:
        orders.at[idx, col] = row[col] * scale

# Przeskaluj order_items (używamy skali 2018-2023 dla spójności)
order_items['unit_price'] = order_items['unit_price'] * scale_2018_2023
order_items['total_price'] = order_items['total_price'] * scale_2018_2023
order_items['discount'] = order_items['discount'] * scale_2018_2023

# Zapisz nowe pliki
orders.to_csv(data_dir / 'orders2.csv', index=False)
order_items.to_csv(data_dir / 'order_items2.csv', index=False)

print(f"\n✅ ZAPISANO NOWE PLIKI:")
print(f"📁 {data_dir / 'orders2.csv'}")
print(f"📁 {data_dir / 'order_items2.csv'}")

# Walidacja
print(f"\n📊 PRZYCHÓD PO SKALOWANIU:")
final_revenue = orders.groupby('year')['total'].sum()
for year, revenue in final_revenue.items():
    status = "🎯" if year == 2023 and abs(revenue - target_2023) < 10000 else ""
    print(f"  {year}: ${revenue:,.2f} {status}")
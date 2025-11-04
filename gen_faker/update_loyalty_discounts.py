import pandas as pd
from pathlib import Path

# Set up paths
data_dir = Path(r"C:\Users\User\PycharmProjects\census_ecommerce\data\synthetic")

print("🔍 Wczytywanie danych...")
# Load the necessary files
customers = pd.read_csv(data_dir / 'customers.csv')
orders = pd.read_csv(data_dir / 'orders2.csv')

print("🔄 Aktualizacja zniżek lojalnościowych...")
# Merge orders with customers to get loyalty_scores
merged = pd.merge(orders, customers[['customer_id', 'loyalty_score']], on='customer_id', how='left')

# Calculate new loyalty_discount as 10% of subtotal multiplied by loyalty_score
# This means higher loyalty_score = higher discount (up to 10% of subtotal)
merged['new_loyalty_discount'] = (merged['subtotal'] * 0.10 * merged['loyalty_score']).round(2)

# Update the total to reflect the new discount
merged['total'] = merged['subtotal'] + merged['tax'] + merged['shipping'] - merged['new_loyalty_discount']

# Round all monetary values to 2 decimal places
monetary_cols = ['subtotal', 'tax', 'shipping', 'total']
merged[monetary_cols] = merged[monetary_cols].round(2)

# Prepare the final dataframe with original column order
final_orders = merged[['order_id', 'customer_id', 'order_date', 'status', 'payment_method',
                      'subtotal', 'tax', 'shipping', 'new_loyalty_discount', 'total']]

# Rename the discount column to maintain consistency
final_orders = final_orders.rename(columns={'new_loyalty_discount': 'loyalty_discount'})

# Save to a new file
output_file = data_dir / 'orders3.csv'
final_orders.to_csv(output_file, index=False)

print(f"✅ Zapisano zaktualizowane zamówienia do: {output_file}")

# Display some statistics for verification
print("\n📊 Statystyki nowych zniżek lojalnościowych:")
print(f"- Średnia wartość zniżki: ${final_orders['loyalty_discount'].mean():.2f}")
print(f"- Maksymalna wartość zniżki: ${final_orders['loyalty_discount'].max():.2f}")
print(f"- Minimalna wartość zniżki: ${final_orders['loyalty_discount'].min():.2f}")

# Check the correlation in the new file
merged = pd.merge(final_orders, customers[['customer_id', 'loyalty_score']], on='customer_id')
correlation = merged[['loyalty_discount', 'loyalty_score']].corr().iloc[0,1]
print(f"\n🔗 Nowa korelacja między loyalty_discount a loyalty_score: {correlation:.4f}")

print("\n🎉 Skrypt zakończył działanie pomyślnie!")

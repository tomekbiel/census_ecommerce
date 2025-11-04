import pandas as pd
import numpy as np
from pathlib import Path

def calculate_new_loyalty_scores(customers_df, orders_df):
    """
    Calculate new loyalty scores based on loyalty_discount from orders.
    
    Args:
        customers_df: DataFrame with customer data
        orders_df: DataFrame with order data including loyalty_discount
        
    Returns:
        DataFrame with updated loyalty_scores
    """
    # Calculate average loyalty_discount per customer (normalized by their average order value)
    customer_stats = orders_df.groupby('customer_id').agg(
        avg_discount=('loyalty_discount', 'mean'),
        avg_order_value=('subtotal', 'mean'),
        order_count=('order_id', 'count')
    ).reset_index()
    
    # Calculate discount as percentage of order value
    customer_stats['discount_pct'] = (customer_stats['avg_discount'] / customer_stats['avg_order_value']).clip(upper=0.10)
    
    # Scale to 0-1 range for loyalty_score (with some minimum value)
    min_discount = customer_stats['discount_pct'].min()
    max_discount = customer_stats['discount_pct'].max()
    
    # Apply min-max scaling with a floor of 0.1 for minimum loyalty
    customer_stats['new_loyalty'] = 0.1 + 0.9 * (
        (customer_stats['discount_pct'] - min_discount) / 
        (max_discount - min_discount + 1e-10)  # Avoid division by zero
    )
    
    # Cap at 0.99 to avoid perfect scores
    customer_stats['new_loyalty'] = customer_stats['new_loyalty'].clip(upper=0.99)
    
    # Round to 2 decimal places
    customer_stats['new_loyalty'] = customer_stats['new_loyalty'].round(2)
    
    # Merge with original customers
    updated_customers = customers_df.merge(
        customer_stats[['customer_id', 'new_loyalty']], 
        on='customer_id', 
        how='left'
    )
    
    # Fill any missing values with the original loyalty_score
    updated_customers['new_loyalty'] = updated_customers['new_loyalty'].fillna(
        updated_customers['loyalty_score']
    )
    
    # Replace the loyalty_score with the new values
    updated_customers['loyalty_score'] = updated_customers['new_loyalty']
    updated_customers = updated_customers.drop(columns=['new_loyalty'])
    
    return updated_customers

def main():
    # Set up paths
    data_dir = Path(r"C:\Users\User\PycharmProjects\census_ecommerce\data\synthetic")
    
    print("🔍 Wczytywanie danych...")
    # Load the data
    customers = pd.read_csv(data_dir / 'customers.csv')
    orders = pd.read_csv(data_dir / 'orders2.csv')
    
    # Calculate new loyalty scores
    print("🔄 Obliczanie nowych wyników lojalności...")
    updated_customers = calculate_new_loyalty_scores(customers, orders)
    
    # Save to new file
    output_file = data_dir / 'customers2.csv'
    updated_customers.to_csv(output_file, index=False)
    
    # Print some statistics
    print("\n📊 Statystyki nowych wyników lojalności:")
    print(f"- Średni wynik lojalności: {updated_customers['loyalty_score'].mean():.2f}")
    print(f"- Maksymalny wynik: {updated_customers['loyalty_score'].max():.2f}")
    print(f"- Minimalny wynik: {updated_customers['loyalty_score'].min():.2f}")
    
    # Check correlation with discounts
    merged = pd.merge(orders, updated_customers[['customer_id', 'loyalty_score']], on='customer_id')
    correlation = merged[['loyalty_discount', 'loyalty_score']].corr().iloc[0,1]
    print(f"\n🔗 Nowa korelacja między loyalty_discount a loyalty_score: {correlation:.4f}")
    
    print(f"\n✅ Zapisano zaktualizowanych klientów do: {output_file}")
    print("🎉 Skrypt zakończył działanie pomyślnie!")

if __name__ == "__main__":
    main()

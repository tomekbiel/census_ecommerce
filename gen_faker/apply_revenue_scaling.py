import pandas as pd
from gen_ecom_faker5 import EcommerceDataGenerator

def test_apply_revenue_scaling():
    # Inicjalizacja generatora
    generator = EcommerceDataGenerator(target_revenue=2_300_000)
    
    # Pobranie oryginalnych danych
    shopify_data = generator._load_shopify_data()
    
    # Utworzenie kopii danych do skalowania
    data_to_scale = shopify_data[['date', 'total_sales_usd']].copy()
    
    # Skalowanie danych
    scaled_data = generator._apply_revenue_scaling(data_to_scale.copy())
    
    # Dodanie kolumny z rokiem
    shopify_data['year'] = pd.to_datetime(shopify_data['date']).dt.year
    scaled_data['year'] = pd.to_datetime(scaled_data['date']).dt.year
    
    # Grupowanie roczne
    yearly_shopify = shopify_data.groupby('year')['total_sales_usd'].sum().reset_index()
    yearly_scaled = scaled_data.groupby('year')['total_sales_usd'].sum().reset_index()
    
    # Połączenie danych w jedną tabelę
    comparison = pd.merge(
        yearly_shopify, 
        yearly_scaled, 
        on='year', 
        suffixes=('_shopify', '_scaled')
    )
    
    # Obliczenie różnicy
    comparison['difference'] = comparison['total_sales_usd_scaled'] - comparison['total_sales_usd_shopify']
    
    # Formatowanie wyjścia
    pd.options.display.float_format = '{:,.2f}'.format
    print("\n=== PORÓWNANIE ROCZNE ===")
    print("Rok | Shopify | Po skalowaniu | Różnica")
    print("-" * 45)
    
    for _, row in comparison.iterrows():
        print(f"{int(row['year'])} | {row['total_sales_usd_shopify']:,.2f} | {row['total_sales_usd_scaled']:,.2f} | {row['difference']:,.2f}")
    
    return comparison

if __name__ == "__main__":
    test_apply_revenue_scaling()
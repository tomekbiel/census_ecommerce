from pathlib import Path
import pandas as pd
import re

# Wczytaj plik
shopify_file = Path("C:/python/census_ecommerce/data/synthetic/shopify_reports_2018-2024.csv")
df_shopify = pd.read_csv(shopify_file)

# Wyodrębnij wszystkie kategorie
all_categories = []
for categories in df_shopify['Top sales categories'].dropna():
    # Podziel po przecinku i usuń białe znaki
    cats = [cat.strip() for cat in str(categories).split(',')]
    all_categories.extend(cats)

# Usuń duplikaty i posortuj
unique_categories = sorted(list(set(all_categories)))

# Wyświetl unikalne kategorie
print("Znalezione kategorie:")
for cat in unique_categories:
    print(f"'{cat}'")

# Możesz teraz użyć tej listy w swoim generatorze Faker
categories_for_faker = unique_categories
print(f"\nLiczba unikalnych kategorii: {len(categories_for_faker)}")
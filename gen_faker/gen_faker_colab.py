# SEKCJA 1: Wersja dla Google Colab

!pip install faker
# File: gen_faker/gen_ecom_data_fixed.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import re


class EcommerceDataGenerator:
    def __init__(self, target_customers: int = 15000, target_revenue: float = 2300000,
                 missing_email_rate: float = 0.2):
        self.target_customers = target_customers
        self.target_revenue = target_revenue
        self.missing_email_rate = missing_email_rate
        self.fake = Faker()
        
        # ŚCIEŻKA DLA GOOGLE COLAB - zapis w folderze /content/
        self.data_dir = Path("/content/ecommerce_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data will be saved to: {self.data_dir}")

        # Ładowanie danych z GitHub
        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = self._load_monthly_stats()
        
# SEKCJA 1: Wersja z opcją wyboru (Colab lub lokalnie)

class EcommerceDataGenerator:
    def __init__(self, target_customers: int = 15000, target_revenue: float = 2300000,
                 missing_email_rate: float = 0.2, use_colab_path: bool = True):
        self.target_customers = target_customers
        self.target_revenue = target_revenue
        self.missing_email_rate = missing_email_rate
        self.fake = Faker()
        
        # WYBÓR ŚCIEŻKI - Colab lub lokalna
        if use_colab_path:
            # Dla Google Colab
            self.data_dir = Path("/content/ecommerce_data")
        else:
            # Dla lokalnego uruchomienia
            self.data_dir = Path("C:/python/census_ecommerce/data/synthetic")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data will be saved to: {self.data_dir}")

        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = self._load_monthly_stats()
        
# SEKCJA 1: Wersja automatyczna (wykrywa Colab)

class EcommerceDataGenerator:
    def __init__(self, target_customers: int = 15000, target_revenue: float = 2300000,
                 missing_email_rate: float = 0.2):
        self.target_customers = target_customers
        self.target_revenue = target_revenue
        self.missing_email_rate = missing_email_rate
        self.fake = Faker()
        
        # AUTOMATYCZNE WYKRYWANIE COLAB
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
        
        if is_colab:
            # Google Colab - zapis w /content/
            self.data_dir = Path("/content/ecommerce_data")
            print("✓ Running in Google Colab")
        else:
            # Lokalne uruchomienie - oryginalna ścieżka
            self.data_dir = Path("C:/python/census_ecommerce/data/synthetic")
            print("✓ Running locally")
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data will be saved to: {self.data_dir}")

        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = self._load_monthly_stats()

# W Colab możesz potem pobrać pliki:

# Po wygenerowaniu, aby pobrać pliki z Colab:
from google.colab import files

# Pobierz wszystkie wygenerowane pliki
for file in ['products.csv', 'customers.csv', 'orders.csv']:
    filepath = f"/content/ecommerce_data/{file}"
    if Path(filepath).exists():
        files.download(filepath)
    else:
        print(f"File {file} not found")


# SEKCJA 2: Ładowanie danych Shopify (wersja Google Colab)

    def _load_shopify_data(self) -> pd.DataFrame:
        """
        Load and preprocess Shopify sales data from GitHub URL.
        """
        try:
            # URL do pliku na GitHub
            github_url = "https://raw.githubusercontent.com/tomekbiel/census_ecommerce/refs/heads/master/data/synthetic/shopify_with_customers.csv"
            print(f"Loading Shopify data from GitHub: {github_url}")
            
            # Wczytaj dane bezpośrednio z URL
            df = pd.read_csv(github_url)
            
            # Check if required columns exist
            required_columns = ['Month', 'Top sales categories', 'Sales_Weight', 
                              'Est. Customer repeat rate (orders/customer)', 
                              'Avg. order value (USD)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in Shopify data: {', '.join(missing_columns)}")

            # Convert date-related columns
            df['date'] = pd.to_datetime(df['Month'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Ensure Estimated_customers is treated as float (it might be read as string)
            if 'Estimated_customers' in df.columns:
                df['Estimated_customers'] = pd.to_numeric(df['Estimated_customers'], errors='coerce')
                
            print(f"✓ Successfully loaded {len(df)} rows of Shopify data from GitHub")
            print(f"✓ Data covers years: {sorted(df['year'].unique())}")
            return df
            
        except Exception as e:
            print(f"✗ Error loading Shopify data from GitHub: {str(e)}")
            # Return an empty DataFrame with required columns to prevent further errors
            return pd.DataFrame(columns=['Month', 'Top sales categories', 'Sales_Weight', 
                                       'Est. Customer repeat rate (orders/customer)', 
                                       'Avg. order value (USD)', 'date', 'year', 'month'])

# Test komórka w Colab
import pandas as pd
test_url = "https://raw.githubusercontent.com/tomekbiel/census_ecommerce/refs/heads/master/data/synthetic/shopify_with_customers.csv"
try:
    test_df = pd.read_csv(test_url)
    print("✓ GitHub URL works!")
    print(f"Columns: {list(test_df.columns)}")
    print(f"First few rows:\n{test_df.head(2)}")
except Exception as e:
    print(f"✗ Error: {e}")                                       
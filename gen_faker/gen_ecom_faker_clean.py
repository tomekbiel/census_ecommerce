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
    def __init__(self, target_customers: int = 15000, target_revenue: float = 2300000, missing_email_rate: float = 0.2):
        self.yearly_revenue = None
        self.target_customers = target_customers
        self.target_revenue = target_revenue
        self.missing_email_rate = missing_email_rate
        self.fake = Faker()
        self.data_dir = Path("C:/python/census_ecommerce/data/synthetic")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = self._load_monthly_stats()

    def _load_shopify_data(self):
        try:
            df = pd.read_csv(self.data_dir / "shopify_sales_data.csv")
            df['Month'] = pd.to_datetime(df['Month'])
            df['year'] = df['Month'].dt.year
            df['month'] = df['Month'].dt.month
            df['Sales_Weight'] = pd.to_numeric(df['Sales_Weight'], errors='coerce')
            df['Est. Customer repeat rate'] = pd.to_numeric(df['Est. Customer repeat rate'].str.rstrip('%'), errors='coerce') / 100
            df['Avg. order value'] = pd.to_numeric(df['Avg. order value'].str.replace('[\$,]', '', regex=True), errors='coerce')
            return df
        except Exception as e:
            print(f"Error loading Shopify data: {e}")
            return pd.DataFrame()

    def _load_monthly_stats(self):
        monthly_stats = {}
        for _, row in self.shopify_data.iterrows():
            year = row['year']
            month = row['month']
            monthly_stats[(year, month)] = {
                'sales_weight': row['Sales_Weight'],
                'repeat_rate': row['Est. Customer repeat rate'],
                'avg_order_value': row['Avg. order value'],
                'top_categories': [c.strip() for c in str(row['Top sales categories']).split(',') if c.strip()],
                'category_weights': self._generate_monthly_category_weights(row['Top sales categories'], self.categories)
            }
        return monthly_stats

    def _generate_monthly_category_weights(self, raw_top, all_cats):
        if not isinstance(raw_top, str):
            return {cat: 1.0/len(all_cats) for cat in all_cats}
        
        top_cats = [c.strip().title() for c in raw_top.split(',') if c.strip()]
        top_cats = [c for c in top_cats if c in all_cats]
        
        if not top_cats:
            return {cat: 1.0/len(all_cats) for cat in all_cats}
            
        weights = {}
        top_weight = 0.8 / len(top_cats) if top_cats else 0
        other_weight = 0.2 / (len(all_cats) - len(top_cats)) if len(all_cats) > len(top_cats) else 0
        
        for cat in all_cats:
            if cat in top_cats:
                weights[cat] = top_weight
            else:
                weights[cat] = other_weight
                
        return {k: max(v, 0.01) for k, v in weights.items()}

    def _load_category_distribution(self):
        categories = set()
        for cats in self.shopify_data['Top sales categories'].dropna():
            if isinstance(cats, str):
                for cat in cats.split(','):
                    if cat.strip():
                        categories.add(cat.strip().title())
        return sorted(categories)

    def _get_category_based_on_month(self, year, month):
        key = (year, month)
        if key in self.monthly_stats:
            weights = self.monthly_stats[key]['category_weights']
            return random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return random.choice(self.categories)

    def _generate_email_from_name(self, name):
        base_email = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
        email = f"{base_email}@example.com"
        
        if email in self.email_cache:
            email = f"{base_email}{random.randint(1, 1000)}@example.com"
            
        self.email_cache[email] = True
        return email

    def _generate_phone_number(self):
        formats = [
            lambda: f"+48 {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}",
            lambda: f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(100, 999)}",
            lambda: f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(100, 999)}",
            lambda: f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}",
            lambda: f"{random.randint(100000000, 999999999)}"
        ]
        return random.choice(formats)()

    def _get_monthly_parameters(self, year, month):
        key = (year, month)
        if key in self.monthly_stats:
            return self.monthly_stats[key]
            
        return {
            'sales_weight': 1.0,
            'repeat_rate': 0.1,
            'avg_order_value': 100.0,
            'top_categories': random.sample(self.categories, min(3, len(self.categories))),
            'category_weights': {cat: 1.0/len(self.categories) for cat in self.categories}
        }

    def generate_products(self, num_products=200):
        products = []
        for i in range(1, num_products + 1):
            year = random.randint(2018, 2024)
            month = random.randint(1, 12)
            category = self._get_category_based_on_month(year, month)
            
            price = np.random.lognormal(3.5, 0.5) * 10
            price = round(max(5, min(price, 1000)), 2)
            cost = round(price * random.uniform(0.4, 0.7), 2)
            
            products.append({
                'product_id': f"P{i:05d}",
                'name': f"Product {i}",
                'category': category,
                'price': price,
                'cost': cost,
                'is_active': random.random() < 0.85
            })
            
        return pd.DataFrame(products)

    def generate_customers(self):
        customers = []
        retention_rates = {2018: 0.3, 2019: 0.4, 2020: 0.5, 2021: 0.6, 2022: 0.7, 2023: 0.8, 2024: 0.9}
        
        for year in range(2018, 2025):
            base_customers = int(self.target_customers * (retention_rates[year] ** (2024 - year)))
            
            for _ in range(base_customers):
                name = self.fake.name()
                email = self._generate_email_from_name(name) if random.random() > self.missing_email_rate else None
                
                customers.append({
                    'customer_id': f"C{len(customers) + 1:06d}",
                    'name': name,
                    'email': email,
                    'phone': self._generate_phone_number(),
                    'join_date': self.fake.date_between_dates(
                        date_start=datetime(year, 1, 1),
                        date_end=datetime(year, 12, 31)
                    ),
                    'region': self.fake.country(),
                    'loyalty_score': round(np.random.beta(2, 5), 2),
                    'cohort_year': year
                })
                
        return pd.DataFrame(customers)

    def generate_orders(self, customers_df, products_df):
        orders = []
        order_id = 1
        total_revenue = 0.0

        active_products = products_df[products_df['is_active'] == True].copy()
        date_range = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
        
        customers_df['join_year'] = pd.to_datetime(customers_df['join_date']).dt.year
        customers_by_year = customers_df.groupby('join_year').size().to_dict()
        
        self.yearly_revenue = {year: 0.0 for year in range(2018, 2025)}
        yearly_orders = {year: 0 for year in range(2018, 2025)}

        for date in tqdm(date_range, desc="Generating orders"):
            year = date.year
            month = date.month
            
            monthly_params = self._get_monthly_parameters(year, month)
            eligible_years = [y for y in customers_by_year.keys() if y <= year]
            eligible_customers = sum(customers_by_year[y] for y in eligible_years)
            
            base_daily_orders = (eligible_customers * monthly_params['repeat_rate'] / 365) * monthly_params['sales_weight']
            
            if year > 2022:
                base_daily_orders *= 0.7
                
            daily_orders = int(base_daily_orders * (1 + 0.2 * (date.weekday() >= 5)))
            daily_orders = max(1, min(daily_orders, 300))

            for _ in range(daily_orders):
                customer = customers_df[customers_df['join_date'] <= date.date()].sample(1).iloc[0]
                order_date = date + timedelta(hours=random.randint(9, 20))
                order_status = 'completed' if random.random() < 0.95 else 'cancelled'
                
                order_data = {
                    'order_id': f"O{order_id:07d}",
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'status': order_status,
                    'payment_method': random.choices(
                        ['credit_card', 'paypal', 'bank_transfer'],
                        weights=[0.70, 0.20, 0.10]
                    )[0],
                    'total_amount': 0.0,
                    'item_count': 0,
                    'total_quantity': 0,
                    'avg_discount': 0.0,
                    'product_categories': '',
                    'products_list': ''
                }
                
                if order_status == 'cancelled':
                    orders.append(order_data)
                    order_id += 1
                    continue

                num_items = min(int(np.random.poisson(lam=1.8) + 1), 10)
                order_total = 0.0
                total_quantity = 0
                total_discount = 0.0
                categories = []
                products_info = []
                
                for _ in range(num_items):
                    current_month_categories = monthly_params['top_categories']
                    suitable_products = active_products[
                        active_products['category'].isin([
                            cat for cat in self.categories
                            if any(c.lower() in cat.lower() for c in current_month_categories)
                        ])
                    ]

                    if len(suitable_products) > 0:
                        if year >= 2021:
                            product = suitable_products.nlargest(10, 'price').sample(1).iloc[0]
                        else:
                            product = suitable_products.sample(1).iloc[0]
                    else:
                        product = active_products.sample(1).iloc[0]

                    quantity = max(1, np.random.poisson(lam=1.8))
                    discount = round(random.uniform(0, 0.3), 2)
                    inflation_year = min(year, 2022)
                    price_multiplier = 1.0 + (inflation_year - 2018) * 0.03
                    adjusted_price = product['price'] * price_multiplier

                    item_total = adjusted_price * quantity * (1 - discount)
                    order_total += item_total
                    total_quantity += quantity
                    total_discount += discount * quantity
                    self.yearly_revenue[year] += item_total

                    categories.append(product['category'])
                    products_info.append(f"{product['product_id']}(x{quantity})")

                order_data.update({
                    'total_amount': round(order_total, 2),
                    'item_count': num_items,
                    'total_quantity': total_quantity,
                    'avg_discount': round(total_discount / num_items, 2) if num_items > 0 else 0.0,
                    'product_categories': ','.join(sorted(set(categories))),
                    'products_list': '|'.join(products_info),
                    'status': 'completed'
                })
                
                orders.append(order_data)
                order_id += 1
                yearly_orders[year] += 1
                total_revenue += order_total

        print("\nYearly Order Summary:")
        for year in range(2018, 2025):
            if yearly_orders[year] > 0:
                print(f"{year}: {yearly_orders[year]:,} orders (€{self.yearly_revenue[year]:,.2f})")

        orders_df = pd.DataFrame(orders)
        int_cols = ['item_count', 'total_quantity']
        for col in int_cols:
            if col in orders_df.columns:
                orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce').fillna(0).astype(int)

        print("\nOrder Generation Summary:")
        print(f"Total orders generated: {len(orders_df):,}")
        print(f"Total revenue: €{orders_df['total_amount'].sum():,.2f}")
        print(f"Average order value: €{orders_df['total_amount'].mean():.2f}")

        if 'status' in orders_df.columns:
            print("\nOrder Status Distribution:")
            print(orders_df['status'].value_counts())

        return orders_df

    def save_to_csv(self, df, filename):
        df.to_csv(self.data_dir / filename, index=False, encoding='utf-8')

    def generate_all_data(self):
        print("Generating products...")
        products_df = self.generate_products()
        
        print("\nGenerating customers...")
        customers_df = self.generate_customers()
        
        print("\nGenerating orders...")
        orders_df = self.generate_orders(customers_df, products_df)
        
        print("\nSaving data...")
        self.save_to_csv(products_df, 'products.csv')
        self.save_to_csv(customers_df, 'customers.csv')
        self.save_to_csv(orders_df, 'orders.csv')
        
        return products_df, customers_df, orders_df

def main():
    generator = EcommerceDataGenerator()
    generator.generate_all_data()

if __name__ == "__main__":
    main()

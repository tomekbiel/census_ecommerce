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
        self.data_dir = Path(__file__).parent.parent / "data/processed"
        self.data_dir.mkdir(exist_ok=True)

        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = self._load_monthly_stats()

    def _load_shopify_data(self) -> pd.DataFrame:
        shopify_file = Path(__file__).parent.parent / "data/synthetic/shopify_monthly_reports_2018-2024.csv"
        df = pd.read_csv(shopify_file)
        df['date'] = pd.to_datetime(df['Month'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df

    def _load_monthly_stats(self) -> Dict:
        monthly_stats = {}
        for _, row in self.shopify_data.iterrows():
            key = (row['year'], row['month'])
            monthly_stats[key] = {
                'sales_weight': row['Sales_Weight'],
                'repeat_rate': row['Est. Customer repeat rate (orders/customer)'],
                'avg_order_value': row['Avg. order value (USD)'] * 0.9,
                'top_categories': [cat.strip() for cat in str(row['Top sales categories']).split(',')],
                'estimated_customers': row.get('Estimated_customers', row['Active stores (mln)'] * 1e6 * 1.5)
            }
        return monthly_stats

    def _load_category_distribution(self) -> Dict[str, Dict]:
        category_hierarchy = {
            'Electronics': {'rank': 1, 'weight': 0.25},
            'electronics': {'rank': 1, 'weight': 0.23},
            'Fashion': {'rank': 2, 'weight': 0.20},
            'fashion': {'rank': 2, 'weight': 0.18},
            'apparel': {'rank': 2, 'weight': 0.17},
            'Health': {'rank': 3, 'weight': 0.15},
            'health': {'rank': 3, 'weight': 0.14},
            'home': {'rank': 4, 'weight': 0.12},
            'home furnishings': {'rank': 4, 'weight': 0.11},
            'furnishings': {'rank': 4, 'weight': 0.10},
            'sports': {'rank': 5, 'weight': 0.08},
            'luxury goods': {'rank': 6, 'weight': 0.05}
        }
        return category_hierarchy

    def _get_category_based_on_month(self, year: int, month: int) -> str:
        key = (year, month)
        if key in self.monthly_stats:
            top_categories = self.monthly_stats[key]['top_categories']
            if top_categories:
                primary_category = top_categories[0]
                available_categories = []
                weights = []

                for cat, stats in self.categories.items():
                    if cat.lower() in [c.lower() for c in top_categories]:
                        available_categories.append(cat)
                        weight = stats['weight'] * (1.5 if cat.lower() == primary_category.lower() else 1.0)
                        weights.append(weight)

                if available_categories:
                    return random.choices(available_categories, weights=weights, k=1)[0]

        categories = list(self.categories.keys())
        weights = [stats['weight'] for stats in self.categories.values()]
        return random.choices(categories, weights=weights, k=1)[0]

    def _generate_email_from_name(self, name: str) -> str:
        clean_name = re.sub(r'[^a-zA-Z\s]', '', name).lower().strip()
        parts = clean_name.split()

        if len(parts) >= 2:
            email_format = random.choice([
                f"{parts[0]}.{parts[-1]}",
                f"{parts[0][0]}{parts[-1]}",
                f"{parts[0]}_{parts[-1]}",
                f"{parts[0]}{random.randint(1, 99)}"
            ])
        else:
            email_format = parts[0] if parts else "customer"

        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
        email = f"{email_format}@{random.choice(domains)}"

        if email in self.email_cache.values():
            email = f"{email_format}{random.randint(100, 999)}@{random.choice(domains)}"

        return email

    def _get_monthly_parameters(self, year: int, month: int) -> Dict:
        key = (year, month)
        if key in self.monthly_stats:
            return self.monthly_stats[key]

        return {
            'sales_weight': 0.3,
            'repeat_rate': 2.0,
            'avg_order_value': 85.0,
            'top_categories': ['Electronics', 'Fashion', 'Health'],
            'estimated_customers': 1000000
        }

    def generate_products(self, num_products: int = 200) -> pd.DataFrame:
        products = []

        for i in range(1, num_products + 1):
            create_year = random.randint(2018, 2024)
            create_month = random.randint(1, 12)

            category = self._get_category_based_on_month(create_year, create_month)
            category_stats = self.categories[category]

            base_multiplier = 0.8 + (0.4 * (7 - category_stats['rank']) / 6)
            base_price = np.random.uniform(30 * base_multiplier, 200 * base_multiplier)

            products.append({
                'product_id': f"P{i:05d}",
                'name': f"Product {i} - {self.fake.word().capitalize()}",
                'category': category,
                'price': round(base_price, 2),
                'cost': round(base_price * random.uniform(0.4, 0.7), 2),
                'created_at': self.fake.date_between_dates(
                    date_start=datetime(create_year, 1, 1),
                    date_end=datetime(create_year, 12, 31)
                ),
                'is_active': random.choices([True, False], weights=[0.85, 0.15])[0]
            })

        return pd.DataFrame(products)

    def generate_customers(self) -> pd.DataFrame:
        customers = []

        # Fixed distribution to ensure exactly target_customers
        yearly_distribution = {2018: 0.10, 2019: 0.15, 2020: 0.20,
                               2021: 0.25, 2022: 0.15, 2023: 0.10, 2024: 0.05}

        customer_id = 1
        for year, percentage in yearly_distribution.items():
            num_customers = int(self.target_customers * percentage)

            for _ in tqdm(range(num_customers), desc=f"Generating {year} customers"):
                start_dt = datetime(year, 1, 1)
                end_dt = datetime(year, 12, 31)

                name = self.fake.name()

                customers.append({
                    'customer_id': f"C{customer_id:05d}",
                    'name': name,
                    'email': self._generate_email_from_name(
                        name) if random.random() > self.missing_email_rate else None,
                    'join_date': self.fake.date_between_dates(date_start=start_dt, date_end=end_dt),
                    'region': self.fake.state(),
                    'loyalty_score': round(np.random.beta(2, 5), 3)
                })
                customer_id += 1

        return pd.DataFrame(customers)

    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        orders = []
        order_id = 1

        active_products = products_df[products_df['is_active'] == True]
        date_range = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')

        # Pre-calculate customer counts by join date
        customers_df['join_year'] = pd.to_datetime(customers_df['join_date']).dt.year
        customers_by_year = customers_df.groupby('join_year').size().to_dict()

        for date in tqdm(date_range, desc="Generating orders 2018-2024"):
            year, month = date.year, date.month
            monthly_params = self._get_monthly_parameters(year, month)

            # Calculate base daily orders without excessive scaling
            repeat_rate = monthly_params['repeat_rate']
            sales_weight = monthly_params['sales_weight']

            # Count customers who could have ordered by this date
            eligible_years = [y for y in customers_by_year.keys() if y <= year]
            eligible_customers = sum(customers_by_year[y] for y in eligible_years)

            # FIXED: Remove excessive scaling, use sales_weight as multiplier
            daily_orders = int((eligible_customers * repeat_rate / 365) * sales_weight)
            daily_orders = max(1, min(daily_orders, 300))

            for _ in range(daily_orders):
                # Select from customers who joined before order date
                eligible_customers_df = customers_df[customers_df['join_date'] <= date.date()]
                if len(eligible_customers_df) == 0:
                    continue

                customer = eligible_customers_df.sample(1).iloc[0]
                order_date = date + timedelta(hours=random.randint(9, 20))

                order_status = random.choices(['completed', 'cancelled'], weights=[0.95, 0.05])[0]

                order_data = {
                    'order_id': f"O{order_id:07d}",
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'status': order_status,
                    'payment_method': random.choices(
                        ['credit_card', 'paypal', 'bank_transfer'], weights=[0.70, 0.20, 0.10])[0],
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

                # Generate order items
                num_items = min(np.random.geometric(p=0.4), 10)  # Increased items probability
                order_total = 0.0
                total_quantity = 0
                total_discount = 0.0
                categories = []
                products_info = []

                for _ in range(num_items):
                    current_month_categories = monthly_params['top_categories']
                    suitable_products = active_products[
                        active_products['category'].isin([cat for cat in self.categories.keys()
                                                          if any(
                                c.lower() in cat.lower() for c in current_month_categories)])
                    ]

                    if len(suitable_products) > 0:
                        product = suitable_products.sample(1).iloc[0]
                    else:
                        product = active_products.sample(1).iloc[0]

                    quantity = max(1, np.random.poisson(lam=1.8))  # Increased quantity
                    discount = round(random.uniform(0, 0.3), 2)

                    item_total = product['price'] * quantity * (1 - discount)
                    order_total += item_total
                    total_quantity += quantity
                    total_discount += discount

                    categories.append(product['category'])
                    products_info.append(f"{product['product_id']}(x{quantity})")

                    order_data.update({
                        'total_amount': round(order_total, 2),
                        'item_count': num_items,
                        'total_quantity': total_quantity,
                        'avg_discount': round(total_discount / num_items, 2) if num_items > 0 else 0.0,
                        'product_categories': ','.join(sorted(set(categories))),
                        'products_list': '|'.join(products_info)
                    })

                orders.append(order_data)
                order_id += 1

        return pd.DataFrame(orders)

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df):,} records to {filepath}")

    def generate_all_data(self):
        print("Generating products...")
        products_df = self.generate_products(num_products=200)

        print("\nGenerating customers...")
        customers_df = self.generate_customers()

        print("\nGenerating orders 2018-2024...")
        orders_df = self.generate_orders(customers_df, products_df)

        return products_df, customers_df, orders_df


def main():
    print("Starting fixed e-commerce data generation...")
    print("Target: 15,000 customers, realistic order distribution")

    generator = EcommerceDataGenerator()

    products_df, customers_df, orders_df = generator.generate_all_data()

    # Calculate metrics
    completed_orders = orders_df[orders_df['status'] == 'completed']
    total_revenue = completed_orders['total_amount'].sum()
    total_orders = len(orders_df)
    total_customers = len(customers_df)

    print(f"\n=== RESULTS ===")
    print(f"Total Customers: {total_customers:,}")
    print(f"Total Orders: {total_orders:,}")
    print(f"Orders per Customer: {total_orders / total_customers:.2f}")
    print(f"Total Revenue: €{total_revenue:,.2f}")

    # Yearly analysis
    orders_df['year'] = pd.to_datetime(orders_df['order_date']).dt.year
    yearly_stats = orders_df.groupby('year').size()
    print(f"\nYearly Order Distribution:")
    print(yearly_stats)

    # Save files
    print(f"\nSaving data...")
    generator.save_to_csv(products_df, "products.csv")
    generator.save_to_csv(customers_df, "customers.csv")
    generator.save_to_csv(orders_df, "orders.csv")

    print(f"\nData generation complete!")


if __name__ == "__main__":
    main()
# File: gen_faker/gen_ecom_faker0.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Dict, List, Tuple
from tqdm import tqdm


class EcommerceDataGenerator:
    def __init__(self, target_customers: int = 15000, missing_email_rate: float = 0.2,
                 num_duplicates: int = 500):
        self.target_customers = target_customers
        self.missing_email_rate = missing_email_rate
        self.num_duplicates = num_duplicates
        self.fake = Faker()
        self.data_dir = Path(__file__).parent.parent / "data/processed"
        self.data_dir.mkdir(exist_ok=True)

        # Load Shopify data for scaling
        self.shopify_data = self._load_shopify_data()
        self.base_total_customers = self.shopify_data['Active stores (mln)'].iloc[-1] * 1e6  # Last month active stores
        self.scaling_factor = self.target_customers / self.base_total_customers

        self.categories = self._load_category_distribution()

    def _load_shopify_data(self) -> pd.DataFrame:
        """Load and process Shopify data"""
        shopify_file = Path(__file__).parent.parent / "data/synthetic/shopify_monthly_reports_2018-2024.csv"
        df = pd.read_csv(shopify_file)

        # Convert to datetime and extract year, month
        df['date'] = pd.to_datetime(df['Month'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        return df

    def _load_category_distribution(self) -> Dict[str, Dict]:
        """Load and analyze category distribution from Shopify data"""
        # Extract top categories and their monthly weights
        category_data = {}

        for _, row in self.shopify_data.iterrows():
            month = row['month']
            year = row['year']
            sales_weight = row['Sales_Weight']
            categories = [cat.strip() for cat in str(row['Top sales categories']).split(',')]

            for category in categories:
                if category not in category_data:
                    category_data[category] = {'monthly_weights': {}, 'total_sales': 0}

                category_data[category]['monthly_weights'][(year, month)] = sales_weight
                category_data[category]['total_sales'] += row['Total sales (USD mln)'] * 1e6 * sales_weight

        # Calculate peak month and distribution parameters for each category
        category_stats = {}
        for category, data in category_data.items():
            monthly_weights = list(data['monthly_weights'].values())

            if len(monthly_weights) >= 2:
                mean_weight = np.mean(monthly_weights)
                var_weight = np.var(monthly_weights)
                shape = (mean_weight ** 2) / var_weight if var_weight > 0 else 2.0
                scale = var_weight / mean_weight if mean_weight > 0 else 0.5
            else:
                shape, scale = 2.0, 0.5

            # Find peak month (month with highest average weight)
            month_weights = {}
            for (year, month), weight in data['monthly_weights'].items():
                month_weights[month] = month_weights.get(month, 0) + weight

            peak_month = max(month_weights.items(), key=lambda x: x[1])[0] if month_weights else 6

            category_stats[category] = {
                'shape': shape,
                'scale': scale,
                'peak_month': peak_month,
                'total_sales': data['total_sales'],
                'avg_order_value': self.shopify_data['Avg. order value (USD)'].mean()
            }

        return category_stats

    def _get_monthly_scaling_factor(self, year: int, month: int) -> float:
        """Get scaling factor based on monthly sales data"""
        month_data = self.shopify_data[
            (self.shopify_data['year'] == year) &
            (self.shopify_data['month'] == month)
            ]

        if len(month_data) == 0:
            return 1.0

        # Use sales weight and active stores to determine scaling
        sales_weight = month_data['Sales_Weight'].values[0]
        active_stores = month_data['Active stores (mln)'].values[0] * 1e6

        # Base scaling on active stores and sales intensity
        return (active_stores / self.base_total_customers) * sales_weight * 1.5

    def _get_category_for_month(self, month: int) -> str:
        """Select a category based on the month using Shopify weights"""
        categories = list(self.categories.keys())
        weights = []

        for category in categories:
            stats = self.categories[category]
            month_diff = min(abs(month - stats['peak_month']),
                             12 - abs(month - stats['peak_month']))

            # Use gamma distribution based on category statistics
            scale = max(stats['scale'], 0.1)
            weight = stats['shape'] * scale * np.exp(-month_diff / scale)
            weights.append(weight)

        # Normalize and select
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1 / len(weights)] * len(weights)

        return random.choices(categories, weights=weights, k=1)[0]

    def generate_products(self, num_products: int = 200) -> pd.DataFrame:
        """Generate product catalog with pricing based on Shopify AOV"""
        products = []
        avg_aov = self.shopify_data['Avg. order value (USD)'].mean()

        for i in range(1, num_products + 1):
            current_month = datetime.now().month
            category = self._get_category_for_month(current_month)

            # Price based on category's average order value
            category_stats = self.categories.get(category, {})
            target_aov = category_stats.get('avg_order_value', avg_aov)

            # Generate price around 0.3-0.7 of AOV (typical basket has 2-3 items)
            base_price = np.random.uniform(target_aov * 0.3, target_aov * 0.7)

            products.append({
                'product_id': f"P{i:05d}",
                'name': f"Product {i} - {self.fake.word().capitalize()}",
                'category': category,
                'subcategory': self.fake.word(),
                'price': round(base_price, 2),
                'cost': round(base_price * random.uniform(0.4, 0.7), 2),
                'weight_kg': round(random.uniform(0.1, 5.0), 2),
                'created_at': self.fake.date_time_between(start_date='-3y', end_date='-6m'),
                'is_active': random.choices([True, False], weights=[0.9, 0.1])[0],
                'seasonal_peak': self.categories[category]['peak_month']
            })

        return pd.DataFrame(products)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer data scaled to Shopify growth"""
        customers = []
        used_emails = set()

        # Calculate customer distribution based on Shopify growth
        yearly_growth = []
        for year in range(2018, 2025):
            year_data = self.shopify_data[self.shopify_data['year'] == year]
            if len(year_data) > 0:
                avg_stores = year_data['Active stores (mln)'].mean() * 1e6
                yearly_growth.append((year, avg_stores))

        # Distribute customers across years based on growth
        total_base_customers = sum(stores for year, stores in yearly_growth)
        yearly_targets = {}

        for year, stores in yearly_growth:
            yearly_targets[year] = int(self.target_customers * (stores / total_base_customers))

        # Generate customers for each year
        customer_id = 1
        for year, target in yearly_targets.items():
            for i in tqdm(range(target), desc=f"Generating {year} customers"):
                email = self.fake.unique.email() if random.random() > self.missing_email_rate else None
                if email:
                    used_emails.add(email)

                # Create datetime objects for date_between
                start_dt = datetime(year, 1, 1)
                end_dt = datetime(year, 12, 31)

                customers.append({
                    'customer_id': f"C{customer_id:05d}",
                    'name': self.fake.name(),
                    'email': email,
                    'join_date': self.fake.date_between_dates(date_start=start_dt, date_end=end_dt),
                    'region': self.fake.state(),
                    'loyalty_score': np.random.beta(2, 5),
                    'join_year': year
                })
                customer_id += 1

        # Add duplicates
        for i in range(self.num_duplicates):
            if not customers:
                break
            dup_customer = random.choice(customers).copy()
            dup_customer['customer_id'] = f"D{len(customers) + i + 1:05d}"
            customers.append(dup_customer)

        return pd.DataFrame(customers)

    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame,
                        start_date: str = '2018-01-01', end_date: str = '2024-12-31') -> tuple:
        """Generate order data based on Shopify monthly patterns"""
        orders = []
        order_items = []
        order_id = 1

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        customers_by_year = customers_df.groupby('join_year').size().to_dict()

        for date in tqdm(date_range, desc="Generating orders"):
            year, month = date.year, date.month

            # Get Shopify scaling factors
            scaling_factor = self._get_monthly_scaling_factor(year, month)
            month_data = self.shopify_data[
                (self.shopify_data['year'] == year) &
                (self.shopify_data['month'] == month)
                ]

            if len(month_data) > 0:
                repeat_rate = month_data['Est. Customer repeat rate (orders/customer)'].values[0]
                aov_target = month_data['Avg. order value (USD)'].values[0]
            else:
                repeat_rate = 2.0
                aov_target = 50.0

            # Calculate daily order volume
            customers_this_year = customers_by_year.get(year, 0)
            daily_orders = int((customers_this_year * repeat_rate / 365) * scaling_factor)
            daily_orders = max(1, daily_orders)  # At least 1 order per day

            for _ in range(daily_orders):
                # Select customer who joined before order date
                eligible_customers = customers_df[customers_df['join_date'] <= date.date()]
                if len(eligible_customers) == 0:
                    continue

                customer = eligible_customers.sample(1).iloc[0]
                order_date = date + timedelta(hours=random.randint(9, 20))

                order = {
                    'order_id': f"O{order_id:07d}",
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'total_amount': 0,
                    'status': random.choices(
                        ['completed', 'processing', 'shipped', 'delivered', 'cancelled'],
                        weights=[0.6, 0.1, 0.1, 0.15, 0.05]
                    )[0],
                    'payment_method': random.choices(
                        ['credit_card', 'paypal', 'bank_transfer'],
                        weights=[0.7, 0.2, 0.1]
                    )[0]
                }

                # Generate order items targeting the AOV
                num_items = min(np.random.geometric(p=0.3), 10)
                order_total = 0
                target_item_value = aov_target / max(num_items, 1)

                for _ in range(num_items):
                    # Select product with price around target
                    suitable_products = products_df[
                        (products_df['price'] >= target_item_value * 0.5) &
                        (products_df['price'] <= target_item_value * 1.5)
                        ]

                    if len(suitable_products) == 0:
                        product = products_df.sample(1).iloc[0]
                    else:
                        product = suitable_products.sample(1).iloc[0]

                    quantity = np.random.poisson(lam=1.5) or 1
                    discount = round(random.uniform(0, 0.3), 2)
                    item_total = product['price'] * quantity * (1 - discount)
                    order_total += item_total

                    order_items.append({
                        'order_id': order['order_id'],
                        'product_id': product['product_id'],
                        'quantity': quantity,
                        'unit_price': product['price'],
                        'discount': discount,
                        'item_total': item_total
                    })

                order['total_amount'] = round(order_total, 2)
                orders.append(order)
                order_id += 1

        orders_df = pd.DataFrame(orders)
        order_items_df = pd.DataFrame(order_items)

        return orders_df, order_items_df

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV with progress bar"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df):,} records to {filepath}")


def main():
    print("Starting Shopify-based e-commerce data generation...")
    generator = EcommerceDataGenerator()

    print(f"Scaling factor: {generator.scaling_factor:.4f}")
    print(f"Base total customers: {generator.base_total_customers:,.0f}")

    # Generate data
    print("\nGenerating products...")
    products_df = generator.generate_products(num_products=200)

    print("\nGenerating customers...")
    customers_df = generator.generate_customers()

    print(f"\nTotal customers generated: {len(customers_df):,}")
    print("Customer distribution by year:")
    print(customers_df['join_year'].value_counts().sort_index())

    print("\nGenerating orders...")
    orders_df, order_items_df = generator.generate_orders(customers_df, products_df)

    # Calculate metrics for validation
    avg_aov = orders_df['total_amount'].mean()
    total_orders = len(orders_df)
    total_revenue = orders_df['total_amount'].sum()

    print(f"\nValidation Metrics:")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Average Order Value: ${avg_aov:.2f}")
    print(f"Orders per Customer: {total_orders / len(customers_df):.2f}")

    # Save data
    print("\nSaving data...")
    generator.save_to_csv(products_df, "products.csv")
    generator.save_to_csv(customers_df, "customers.csv")
    generator.save_to_csv(orders_df, "orders.csv")
    generator.save_to_csv(order_items_df, "order_items.csv")

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
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
        self.categories = self._load_category_distribution()

    # Add these methods to the EcommerceDataGenerator class

    def _load_category_distribution(self) -> Dict[str, Dict]:
        """Load and analyze category distribution from Shopify data"""
        shopify_file = Path(__file__).parent.parent / "data/synthetic/shopify_monthly_reports_2018-2024.csv"
        df = pd.read_csv(shopify_file)

        # Debug: Print available columns to find the correct one
        print("Available columns in the CSV file:")
        print(df.columns.tolist())

        # Try to find the categories column (case insensitive)
        category_columns = [col for col in df.columns if 'categor' in col.lower()]
        if not category_columns:
            raise ValueError("No category columns found in the CSV file. Available columns: " + ", ".join(df.columns))

        # Use the first matching column
        category_col = category_columns[0]
        print(f"\nUsing category column: '{category_col}'")

        # Extract and clean top categories
        df['top_categories'] = df[category_col].str.split(',').apply(
            lambda x: [cat.strip() for cat in x] if isinstance(x, list) else []
        )

        # Count monthly occurrences of each category in top positions
        category_months = {}
        for _, row in df.iterrows():
            month = row['Month']
            for i, category in enumerate(row['top_categories']):
                if not category:  # Skip empty categories
                    continue
                if category not in category_months:
                    category_months[category] = []
                # Position is 1-based (1 = top category)
                category_months[category].append((month, i + 1))

        # Calculate gamma distribution parameters for each category
        category_stats = {}
        for category, months in category_months.items():
            positions = [pos for _, pos in months]
            if len(positions) < 2:
                # Default values for categories with insufficient data
                category_stats[category] = {
                    'shape': 2.0,
                    'scale': 0.5,
                    'peak_month': 6  # Default to mid-year
                }
            else:
                # Fit gamma distribution to positions
                mean_pos = np.mean(positions)
                var_pos = np.var(positions)
                shape = (mean_pos ** 2) / var_pos if var_pos > 0 else 2.0
                scale = var_pos / mean_pos if mean_pos > 0 else 0.5

                # Find peak month (most common month for this category)
                month_counts = {}
                for month, _ in months:
                    month_counts[month] = month_counts.get(month, 0) + 1
                peak_month = max(month_counts.items(), key=lambda x: x[1])[0]

                category_stats[category] = {
                    'shape': shape,
                    'scale': scale,
                    'peak_month': pd.to_datetime(peak_month).month
                }

        print("\nCategory distribution parameters:")
        for cat, params in category_stats.items():
            print(
                f"- {cat}: shape={params['shape']:.2f}, scale={params['scale']:.2f}, peak_month={params['peak_month']}")

        return category_stats

    def _get_category_for_month(self, month: int) -> str:
        """Select a category based on the month using gamma distribution"""
        weights = []
        categories = list(self.categories.keys())

        for category in categories:
            stats = self.categories[category]
            # Calculate distance from peak month (handling year wrap-around)
            month_diff = min(abs(month - stats['peak_month']),
                            12 - abs(month - stats['peak_month']))
            # Add small epsilon to prevent division by zero
            scale = max(stats['scale'], 0.1)  # Ensure scale is never zero
            # Use gamma distribution to calculate weight
            weight = stats['shape'] * scale * np.exp(-month_diff / scale)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
        else:
            weights = [1/len(weights)] * len(weights)

        return random.choices(categories, weights=weights, k=1)[0]

    def generate_products(self, num_products: int = 200) -> pd.DataFrame:
        """Generate product catalog with seasonal categories"""
        products = []
        current_month = datetime.now().month

        for i in range(1, num_products + 1):
            # Select category based on current month's distribution
            category = self._get_category_for_month(current_month)

            base_price = np.random.lognormal(mean=3.5, sigma=0.8)

            products.append({
                'product_id': f"P{i:05d}",
                'name': f"Product {i} - {self.fake.word().capitalize()}",
                'category': category,
                'subcategory': self.fake.word(),
                'price': round(base_price * 10) * 5,
                'cost': round(base_price * 7 * random.uniform(0.4, 0.7), 2),
                'weight_kg': round(random.uniform(0.1, 5.0), 2),
                'created_at': self.fake.date_time_between(start_date='-3y', end_date='-6m'),
                'is_active': random.choices([True, False], weights=[0.9, 0.1])[0],
                'seasonal_peak': self.categories[category]['peak_month']
            })

        return pd.DataFrame(products)

    # [Previous methods remain the same until generate_products...]

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer data with realistic attributes"""
        customers = []
        used_emails = set()

        # Generate unique customers
        for i in tqdm(range(1, self.target_customers - self.num_duplicates + 1),
                      desc="Generating customers"):
            email = self.fake.unique.email() if random.random() > self.missing_email_rate else None
            if email:
                used_emails.add(email)

            customers.append({
                'customer_id': f"C{i:05d}",
                'name': self.fake.name(),
                'email': email,
                'join_date': self.fake.date_between(start_date='-5y', end_date='today'),
                'region': self.fake.state(),
                'loyalty_score': np.random.beta(2, 5)  # Most customers have lower scores
            })

        # Add duplicates
        for i in range(self.num_duplicates):
            if not customers:  # In case we have more duplicates than customers
                break
            dup_customer = random.choice(customers).copy()
            dup_customer['customer_id'] = f"D{len(customers) + i + 1:05d}"
            customers.append(dup_customer)

        return pd.DataFrame(customers)

    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame,
                        start_date: str = '2018-01-01', end_date: str = '2024-12-31') -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """Generate order data based on customer behavior"""
        orders = []
        order_items = []  # Store order items separately
        order_id = 1
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        for date in tqdm(date_range, desc="Generating orders"):
            # Adjust order volume by day of week and month
            day_factor = 1.5 if date.weekday() < 5 else 1.0  # Weekdays have more orders
            month_factor = 1.2 if date.month in [11, 12] else 1.0  # Holiday season boost

            # Generate orders for this date
            daily_orders = np.random.poisson(lam=50 * day_factor * month_factor)

            for _ in range(daily_orders):
                customer = customers_df.sample(1).iloc[0]
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

                # Add order items
                num_items = min(np.random.geometric(p=0.3), 10)  # 1-10 items per order
                order_total = 0

                for _ in range(num_items):
                    product = products_df.sample(1).iloc[0]
                    quantity = np.random.poisson(lam=1.5) or 1  # At least 1 item
                    discount = round(random.uniform(0, 0.3), 2)  # 0-30% discount
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

                # Update order total
                order['total_amount'] = round(order_total, 2)
                orders.append(order)
                order_id += 1

        # Convert to DataFrames
        orders_df = pd.DataFrame(orders)
        order_items_df = pd.DataFrame(order_items)  # Already in the correct format

        return orders_df, order_items_df

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV with progress bar"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df):,} records to {filepath}")


def main():
    print("Starting e-commerce data generation...")
    generator = EcommerceDataGenerator()

    # Generate data
    print("\nGenerating products...")
    products_df = generator.generate_products(num_products=200)

    print("\nGenerating customers...")
    customers_df = generator.generate_customers()

    print("\nGenerating orders...")
    orders_df, order_items_df = generator.generate_orders(customers_df, products_df)

    # Save data
    print("\nSaving data...")
    generator.save_to_csv(products_df, "products.csv")
    generator.save_to_csv(customers_df, "customers.csv")
    generator.save_to_csv(orders_df, "orders.csv")
    generator.save_to_csv(order_items_df, "order_items.csv")

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
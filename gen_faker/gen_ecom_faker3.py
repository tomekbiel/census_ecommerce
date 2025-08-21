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
        shopify_file = Path(__file__).parent.parent / "data/synthetic/shopify_with_customers.csv"
        df = pd.read_csv(shopify_file)
        
        # Convert date-related columns
        df['date'] = pd.to_datetime(df['Month'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Ensure Estimated_customers is treated as float (it might be read as string)
        if 'Estimated_customers' in df.columns:
            df['Estimated_customers'] = pd.to_numeric(df['Estimated_customers'], errors='coerce')
        
        return df

    def _load_monthly_stats(self) -> Dict:
        """Load monthly statistics including category weights based on top categories."""
        monthly_stats = {}
        
        # Handle case where self.categories is a list (not a dict)
        if isinstance(self.categories, list):
            all_cats = self.categories
        else:
            all_cats = list(self.categories.keys())
        
        for _, row in self.shopify_data.iterrows():
            key = (row['year'], row['month'])
            raw_top = [cat.strip() for cat in str(row['Top sales categories']).split(',')]
            
            # Generate monthly category weights
            monthly_weights = self._generate_monthly_category_weights(raw_top, all_cats)
            
            monthly_stats[key] = {
                'sales_weight': row['Sales_Weight'],
                'repeat_rate': row['Est. Customer repeat rate (orders/customer)'],
                'avg_order_value': row['Avg. order value (USD)'] * 0.9,
                'top_categories': raw_top,
                'estimated_customers': row.get('Estimated_customers', row['Active stores (mln)'] * 1e6 * 1.5),
                'category_weights': monthly_weights
            }
        return monthly_stats

    def _generate_monthly_category_weights(
        self, 
        raw_top: List[str], 
        all_cats: List[str]
    ) -> Dict[str, float]:
        """
        Generate monthly weights for categories based on top categories.
        
        Args:
            raw_top: List of top categories for the month
            all_cats: List of all possible categories
            
        Returns:
            Dictionary mapping categories to their weights for the month
        """
        # Normalize input categories (title case, remove duplicates)
        norm_top = [cat.strip().title() for cat in raw_top]
        norm_top = list(dict.fromkeys(norm_top))  # Remove duplicates while preserving order
        
        # Filter to only include categories that exist in our global list
        valid_top = [cat for cat in norm_top if cat in all_cats]
        other_cats = [cat for cat in all_cats if cat not in valid_top]
        
        # If no valid top categories, return equal weights
        if not valid_top:
            weight = 1.0 / len(all_cats) if all_cats else 1.0
            return {cat: weight for cat in all_cats}
        
        # Calculate weights: higher for top categories, lower for others
        n_top = len(valid_top)
        n_other = len(other_cats)
        
        # Distribute 80% of weight to top categories, 20% to others
        top_weight = 0.8
        other_weight = 0.2
        
        # Calculate individual weights
        weights = {}
        
        # For top categories: distribute top_weight equally
        top_each = top_weight / n_top if n_top > 0 else 0
        for cat in valid_top:
            weights[cat] = top_each
        
        # For other categories: distribute other_weight equally
        if n_other > 0 and other_weight > 0:
            other_each = other_weight / n_other
            for cat in other_cats:
                weights[cat] = other_each
        
        # Ensure all categories are included with at least a small weight
        for cat in all_cats:
            if cat not in weights:
                weights[cat] = 0.01  # Small weight for any missing categories
        
        # Normalize to ensure the sum is exactly 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights

    def _load_category_distribution(self) -> List[str]:
        """
        Extract and return a list of unique categories from the Shopify data.
        Categories are standardized to title case and case-insensitive.
        
        Returns:
            List of unique category names in title case
        """
        # 1. Get all 'Top sales categories' entries and split into individual categories
        raw_lists = (
            self.shopify_data['Top sales categories']
            .dropna()
            .astype(str)
            .apply(lambda s: [c.strip() for c in s.split(',')])
        )

        # 2. Normalize to lowercase and get unique values
        unique_lower = {cat.lower() for sub in raw_lists for cat in sub}

        # 3. Convert to Title Case and sort alphabetically
        final_cats = [cat.title() for cat in sorted(unique_lower)]

        return final_cats

    def _get_category_based_on_month(self, year: int, month: int) -> str:
        """
        Get a category based on the year and month, using the pre-generated weights.
        
        Args:
            year: The year
            month: The month (1-12)
            
        Returns:
            A category name
        """
        key = (year, month)
        if key in self.monthly_stats:
            # Get the pre-calculated weights for this month
            category_weights = self.monthly_stats[key]['category_weights']
            
            # If we have valid weights, use them for selection
            if category_weights and len(category_weights) > 0:
                categories = list(category_weights.keys())
                weights = list(category_weights.values())
                return random.choices(categories, weights=weights, k=1)[0]
            
            # Fallback to old method if no weights available
            top_categories = self.monthly_stats[key]['top_categories']
            if top_categories:
                # Try to match categories case-insensitively
                available_categories = []
                weights = []
                
                for cat in self.categories:
                    if any(cat.lower() == tc.lower() for tc in top_categories):
                        available_categories.append(cat)
                        # Higher weight for categories earlier in the top_categories list
                        weight = len(top_categories) - [tc.lower() for tc in top_categories].index(cat.lower())
                        weights.append(weight)
                
                if available_categories:
                    return random.choices(available_categories, weights=weights, k=1)[0]
        
        # Fallback: If no monthly data or no matching categories, use global weights
        categories = self.categories
        weights = [1.0 / len(categories) for _ in categories]
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
        """Generate product catalog with realistic pricing and categories."""
        products = []
        
        for i in range(1, num_products + 1):
            # Random creation date between 2018 and 2024
            create_year = random.randint(2018, 2024)
            create_month = random.randint(1, 12)
            
            # Get category based on creation date
            category = self._get_category_based_on_month(create_year, create_month)
            
            # Base price multiplier based on category rank (lower rank = higher price)
            # Rank 1 (highest) will have the highest multiplier (1.0), rank 6 the lowest (0.4)
            base_multiplier = 0.8 + (0.4 * (7 - self.categories.index(category)) / 6)
            
            # Generate base price with some randomness
            base_price = np.random.uniform(30 * base_multiplier, 200 * base_multiplier)
            
            products.append({
                'product_id': f"P{i:05d}",
                'name': f"Product {i} - {self.fake.word().capitalize()}",
                'category': category,
                'price': round(base_price, 2),
                'cost': round(base_price * random.uniform(0.4, 0.7), 2),
                'is_active': random.choices([True, False], weights=[0.85, 0.15])[0]
            })
            
        return pd.DataFrame(products)

    def generate_customers(self) -> pd.DataFrame:
        # 1. Pobierz dane o klientach z pliku wejściowego
        monthly_customers = self.shopify_data.groupby('Year')['Estimated_customers'].first()
        
        # 2. Oblicz retention rate na podstawie danych historycznych
        retention_rates = {}
        years = sorted(monthly_customers.index)
        
        # Dla pierwszego roku nie mamy retention rate, ustawiamy na 1.0
        retention_rates[years[0]] = 1.0
        
        # Oblicz retention rate dla kolejnych lat
        for i in range(1, len(years)):
            current_year = years[i]
            prev_year = years[i-1]
            
            # Oblicz retention rate: ile % klientów z poprzedniego roku pozostało
            if monthly_customers[prev_year] > 0:
                retention = monthly_customers[current_year] / monthly_customers[prev_year]
                # Ogranicz retention do maksymalnie 1.0 (100%)
                retention_rates[current_year] = min(1.0, retention)
            else:
                retention_rates[current_year] = 1.0
        
        # 3. Oblicz liczbę nowych klientów dla każdego roku, uwzględniając retention
        new_customers = {}
        previous_year_customers = 0
        
        for year in years:
            current_total = monthly_customers[year]
            if year == years[0]:
                # Dla pierwszego roku wszyscy klienci są nowi
                new = current_total
            else:
                # Dla kolejnych lat: new = current_total - (previous_year_customers * retention)
                new = current_total - (previous_year_customers * retention_rates[year])
            
            new_customers[year] = max(0, new)  # Upewniamy się, że nie mamy ujemnych wartości
            previous_year_customers = current_total
        
        # 4. Oblicz całkowitą liczbę klientów w danych wejściowych
        total_input_customers = sum(new_customers.values())
        
        # 5. Oblicz współczynnik skalowania, aby uzyskać docelową liczbę klientów (15,000 - 500 duplikatów = 14,500)
        scale_factor = (self.target_customers - 500) / total_input_customers if total_input_customers > 0 else 1.0
        
        # 6. Przeskaluj liczbę nowych klientów dla każdego roku
        scaled_new_customers = {}
        remaining = self.target_customers - 500  # Zarezerwuj miejsce na duplikaty
        
        for year in years[:-1]:
            scaled = int(new_customers[year] * scale_factor)
            scaled_new_customers[year] = scaled
            remaining -= scaled
        
        # Przypisz pozostałych klientów do ostatniego roku, aby uzyskać dokładną sumę
        scaled_new_customers[years[-1]] = remaining
        
        # 7. Wygeneruj klientów z uwzględnieniem retention rate
        customer_id = 1
        customers = []
        active_customers = {}  # Śledzi aktywnych klientów z poprzedniego roku
        
        for year in sorted(scaled_new_customers.keys()):
            # Dodaj nowych klientów
            new_customers_count = scaled_new_customers[year]
            year_customers = []
            
            for _ in range(new_customers_count):
                join_date = self.fake.date_between(
                    start_date=datetime(year, 1, 1),
                    end_date=datetime(year, 12, 31)
                )
                
                name = self.fake.name()
                
                customer_data = {
                    'customer_id': f"C{customer_id:05d}",
                    'name': name,
                    'email': self._generate_email_from_name(name) if random.random() > self.missing_email_rate else None,
                    'join_date': join_date,
                    'region': self.fake.state(),
                    'loyalty_score': round(np.random.beta(2, 5), 3),
                    'cohort_year': year
                }
                
                customers.append(customer_data)
                year_customers.append(customer_data)
                customer_id += 1
            
            # Dodaj klientów, którzy pozostali z poprzedniego roku (zgodnie z retention rate)
            if year > min(active_customers.keys(), default=year) and year in retention_rates:
                retention = retention_rates[year]
                prev_year = year - 1
                if prev_year in active_customers:
                    retained_count = int(len(active_customers[prev_year]) * retention)
                    retained_customers = random.sample(active_customers[prev_year], min(retained_count, len(active_customers[prev_year])))
                    year_customers.extend(retained_customers)
            
            active_customers[year] = year_customers
        
        # Spłaszcz listę aktywnych klientów
        all_customers = [customer for year_customers in active_customers.values() for customer in year_customers]
        
        # 8. Dodaj 500 duplikatów
        if len(all_customers) > 0:
            for _ in range(500):
                original = random.choice(all_customers)
                duplicate = original.copy()
                duplicate['customer_id'] = f"D{customer_id:05d}"
                all_customers.append(duplicate)
                customer_id += 1
        
        return pd.DataFrame(all_customers)

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
                        active_products['category'].isin([cat for cat in self.categories
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
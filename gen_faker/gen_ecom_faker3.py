# File: gen_faker/gen_ecom_faker5.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import random
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm
import re


class EcommerceDataGenerator:
    """
    A class to generate synthetic e-commerce data including customers, products, and orders.
    """

    def __init__(self, target_customers: int = 15000,
                 missing_email_rate: float = 0.2, num_duplicates: int = 500):
        self.target_customers = target_customers

        self.missing_email_rate = missing_email_rate
        self.num_duplicates = num_duplicates
        self.fake = Faker()
        self.end_date_2024 = datetime(2024, 12, 31, 23, 59, 59)
        self.start_date_2018 = datetime(2018, 1, 1)

        self.data_dir = Path(__file__).parent.parent / "data" / "synthetic"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.state_distribution = self._get_state_distribution()

    def _get_state_distribution(self) -> Dict[str, float]:
        top_states = {
            'California': 0.125, 'Texas': 0.090, 'New York': 0.085, 'Florida': 0.080,
            'Illinois': 0.045, 'Pennsylvania': 0.040, 'Ohio': 0.035, 'Georgia': 0.033,
            'North Carolina': 0.032, 'Michigan': 0.030
        }

        mid_states = {
            'New Jersey': 0.028, 'Virginia': 0.027, 'Washington': 0.026, 'Arizona': 0.025,
            'Massachusetts': 0.024, 'Indiana': 0.023, 'Tennessee': 0.022, 'Missouri': 0.021,
            'Maryland': 0.020, 'Wisconsin': 0.019
        }

        other_states = [
            'Alabama', 'Alaska', 'Arkansas', 'Colorado', 'Connecticut', 'Delaware',
            'Hawaii', 'Idaho', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
            'Minnesota', 'Mississippi', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
            'New Mexico', 'North Dakota', 'Oklahoma', 'Oregon', 'Rhode Island',
            'South Carolina', 'South Dakota', 'Utah', 'Vermont', 'West Virginia', 'Wyoming'
        ]

        other_weight = 0.20 / len(other_states)
        other_states_dict = {state: other_weight for state in other_states}

        all_states = {}
        all_states.update(top_states)
        all_states.update(mid_states)
        all_states.update(other_states_dict)

        total_weight = sum(all_states.values())
        if abs(total_weight - 1.0) > 0.001:
            all_states = {k: v / total_weight for k, v in all_states.items()}

        return all_states

    def _load_shopify_data(self) -> pd.DataFrame:
        try:
            shopify_file = self.data_dir / "shopify_with_customers.csv"
            print(f"Loading Shopify data from: {shopify_file}")

            if not shopify_file.exists():
                raise FileNotFoundError(f"Shopify data file not found at: {shopify_file}")

            column_mapping = {
                'Month': 'date',
                'Est. Customer repeat rate (orders/customer)': 'repeat_rate',
                'Avg. order value (USD)': 'avg_order_value',
                'Sales_Weight': 'sales_weight',
                'Total sales (USD mln)': 'total_sales_usd',
                'Top sales categories': 'top_categories'
            }

            df = pd.read_csv(shopify_file, usecols=list(column_mapping.keys()), thousands=' ')
            df = df.rename(columns=column_mapping)

            df['date'] = pd.to_datetime(df['date'])

            numeric_cols = ['repeat_rate', 'avg_order_value', 'sales_weight', 'total_sales_usd']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['top_categories'] = df['top_categories'].str.strip()

            # KONWERSJA MILIONÓW - KLUCZOWE!
            df['total_sales_usd'] = df['total_sales_usd'] * 1_000_000

            df = self._apply_revenue_scaling(df)

            df = df[['date', 'repeat_rate', 'avg_order_value',
                     'sales_weight', 'total_sales_usd', 'top_categories']]

            print(f"✅ Loaded {len(df)} months of Shopify data")
            return df

        except Exception as e:
            print(f"Error loading Shopify data: {str(e)}")
            raise

    def _apply_revenue_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply revenue scaling with fixed factor to reach $2.3M annual revenue.
        """
        df = df.copy()

        target_annual_revenue = 2_300_000
        scale_factor = 0.66  # 🎯 DOKŁADNIE 0.66

        print(f"🎯 Applying fixed scale factor: {scale_factor}")

        # ✅ ZMIANA: SKALOWANIE DO WSZYSTKICH DANYCH
        df['total_sales_usd'] = df['total_sales_usd'] * scale_factor

        # 2. Plateau od 07/2023 - utrzymuj wartość + wahania ±10%
        plateau_start = pd.to_datetime('2023-07-01')
        plateau_mask = df['date'] >= plateau_start

        if plateau_mask.any():
            # Znajdź wartość z czerwca 2023 (już po skalowaniu)
            jun_2023_mask = (df['date'].dt.year == 2023) & (df['date'].dt.month == 6)
            if jun_2023_mask.any():
                plateau_base_value = df.loc[jun_2023_mask, 'total_sales_usd'].iloc[0]

                # Dodaj wahania ±10% dla każdego miesiąca plateau
                for idx in df[plateau_mask].index:
                    fluctuation = random.uniform(0.9, 1.1)  # ±10%
                    df.loc[idx, 'total_sales_usd'] = plateau_base_value * fluctuation

        # Walidacja
        annual_revenues = df.groupby(df['date'].dt.year)['total_sales_usd'].sum()

        print("Annual Revenue After Scaling:")
        for year, revenue in annual_revenues.items():
            status = "🎯 TARGET" if year == 2023 and abs(revenue - target_annual_revenue) < 50000 else ""
            print(f"  {year}: ${revenue:,.2f} {status}")

        return df

    def _generate_monthly_category_weights(self, top_categories_str: str, all_categories: List[str]) -> Dict[
        str, float]:
        if not top_categories_str or pd.isna(top_categories_str):
            weight = 1.0 / len(all_categories) if all_categories else 1.0
            return {cat: weight for cat in all_categories}

        top_cats = [cat.strip().title() for cat in str(top_categories_str).split(',') if cat.strip()]

        invalid_cats = set(top_cats) - set(all_categories)
        if invalid_cats:
            raise ValueError(f"Invalid top categories found: {invalid_cats}")

        n_top = len(top_cats)
        weights = {cat: 0.0 for cat in all_categories}

        top_total = 0.5
        if n_top == 1:
            weights[top_cats[0]] = top_total
        elif n_top == 2:
            split = random.uniform(0.26, 0.49)
            weights[top_cats[0]] = split
            weights[top_cats[1]] = top_total - split
        elif n_top >= 3:
            first = random.uniform(0.3, 0.45)
            rest = top_total - first
            second_share = random.uniform(0.3, 0.6)
            second = rest * second_share
            third = rest - second

            weights[top_cats[0]] = first
            weights[top_cats[1]] = second
            weights[top_cats[2]] = third

        other_cats = [cat for cat in all_categories if cat not in top_cats[:3]]
        if other_cats:
            other_total = 1.0 - sum(weights.values())
            splits = np.random.dirichlet([1] * len(other_cats)) * other_total
            for i, cat in enumerate(other_cats):
                weights[cat] = splits[i]

        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _load_monthly_stats(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if hasattr(self, 'shopify_data'):
            shopify_data = self.shopify_data
        else:
            shopify_data = self._load_shopify_data()

        monthly_stats = {}
        for _, row in shopify_data.iterrows():
            year_month = (row['date'].year, row['date'].month)
            monthly_stats[year_month] = {
                'date': row['date'],
                'repeat_rate': row['repeat_rate'],
                'avg_order_value': row['avg_order_value'],
                'total_sales': row['total_sales_usd'],
                'top_categories': row['top_categories']
            }

        return monthly_stats

    def _load_category_distribution(self) -> List[str]:
        return (self.shopify_data['top_categories']
                .dropna()
                .str.split(',')
                .explode()
                .str.strip()
                .str.title()
                .drop_duplicates()
                .sort_values()
                .tolist())

    def _generate_email_from_name(self, name: str) -> str:
        clean_name = re.sub(r'[^a-zA-Z\s]', '', name).lower().strip()
        parts = clean_name.split()

        if len(parts) >= 2:
            email_format = random.choice([
                f"{parts[0]}.{parts[-1]}",
                f"{parts[0][0]}{parts[-1]}",
                f"{parts[0]}_{parts[-1]}"
            ])
        else:
            email_format = parts[0] if parts else "customer"

        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']

        if email_format in self.email_cache:
            counter = self.email_cache[email_format] + 1
            email = f"{email_format}{counter}@{random.choice(domains)}"
        else:
            counter = 0
            email = f"{email_format}@{random.choice(domains)}"

        self.email_cache[email_format] = counter
        return email

    def _generate_phone_number(self) -> str:
        area_code = f"{random.randint(200, 999):03d}"
        exchange = f"{random.randint(200, 999):03d}"
        line = f"{random.randint(1000, 9999):04d}"

        format_choice = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]

        if format_choice == 1:
            return f"({area_code}) {exchange}-{line}"
        elif format_choice == 2:
            return f"{area_code}-{exchange}-{line}"
        else:
            return f"{area_code}{exchange}{line}"

    def _generate_zip_code(self, state: str) -> str:
        zip_ranges = {
            'ME': '039-049', 'NH': '030-039', 'VT': '050-059', 'MA': '010-027', 'RI': '028-029',
            'CT': '060-069', 'NY': '100-149', 'NJ': '070-089', 'PA': '150-196', 'DE': '197-199',
            'MD': '206-219', 'VA': '220-246', 'WV': '247-269', 'NC': '270-289', 'SC': '290-299',
            'GA': '300-319', 'FL': '320-349', 'OH': '430-459', 'IN': '460-479', 'IL': '600-629',
            'MI': '480-499', 'WI': '530-549', 'KY': '400-427', 'TN': '370-385', 'AL': '350-369',
            'MS': '386-397', 'AR': '716-729', 'LA': '700-714', 'OK': '730-749', 'TX': '750-799',
            'MN': '550-567', 'IA': '500-528', 'MO': '630-658', 'ND': '580-588', 'SD': '570-577',
            'NE': '680-693', 'KS': '660-679', 'MT': '590-599', 'WY': '820-831', 'CO': '800-816',
            'NM': '870-884', 'AZ': '850-865', 'UT': '840-847', 'NV': '889-898', 'ID': '832-838',
            'WA': '980-994', 'OR': '970-979', 'CA': '900-961', 'AK': '995-999', 'HI': '967-968'
        }

        state_abbr = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
            'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
            'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
            'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
            'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
            'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
            'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
            'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
            'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
        }

        state_abbr = state_abbr.get(state, 'NY')
        zip_range = zip_ranges.get(state_abbr, '100-999')
        start, end = map(int, zip_range.split('-'))

        zip_code = random.randint(start, end)
        return f"{zip_code:05d}"

    def _get_first_order_date(self, join_date: datetime) -> Optional[datetime]:
        if random.random() < 0.56:
            return join_date
        else:
            days_later = random.randint(1, 90)
            return join_date + timedelta(days=days_later)

    def generate_customers(self) -> pd.DataFrame:
        print("Generating customers...")

        customers = []
        states = list(self.state_distribution.keys())
        state_weights = list(self.state_distribution.values())

        total_to_generate = int(self.target_customers / 0.8)

        for i in range(total_to_generate):
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            full_name = f"{first_name} {last_name}"

            email = None if random.random() < self.missing_email_rate else self._generate_email_from_name(full_name)

            join_year = random.randint(2018, 2024)
            join_date = self.fake.date_time_between_dates(
                datetime_start=datetime(join_year, 1, 1),
                datetime_end=datetime(join_year, 12, 31, 23, 59, 59)
            )

            first_order_date = self._get_first_order_date(join_date)

            if first_order_date is None:
                continue

            state = random.choices(states, weights=state_weights, k=1)[0]

            years_since_join = 2024 - join_date.year
            base_loyalty = min(0.9, years_since_join * 0.15)
            loyalty_score = round(np.random.normal(base_loyalty, 0.1), 2)
            loyalty_score = max(0, min(1, loyalty_score))

            end_date = min(datetime.now(), self.end_date_2024)
            if first_order_date >= end_date:
                last_purchase_date = first_order_date
            else:
                last_purchase_date = self.fake.date_time_between_dates(
                    datetime_start=first_order_date,
                    datetime_end=end_date
                )

            street_address = self.fake.street_address()
            city = self.fake.city()
            zip_code = self._generate_zip_code(state)

            customers.append({
                'customer_id': f"C{100000 + i}",
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': self._generate_phone_number(),
                'street_address': street_address,
                'city': city,
                'state': state,
                'zip_code': zip_code,
                'join_date': join_date,
                'first_order_date': first_order_date,
                'loyalty_score': loyalty_score,
                'email_optin': random.random() > 0.3,
                'last_purchase_date': last_purchase_date
            })

            if len(customers) >= self.target_customers:
                break

        df = pd.DataFrame(customers)
        df['join_date'] = pd.to_datetime(df['join_date'])
        df['first_order_date'] = pd.to_datetime(df['first_order_date'])
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

        total_generated = len(df)
        immediate_orders = (df['first_order_date'] == df['join_date']).sum()
        later_orders = total_generated - immediate_orders

        print(f"✅ Generated {total_generated} customers WITH orders")
        print(f"📊 {immediate_orders} ({immediate_orders / total_generated * 100:.1f}%) ordered immediately")

        if self.num_duplicates > 0 and len(df) > 0:
            num_duplicates = min(self.num_duplicates, len(df))
            duplicates = df.sample(n=num_duplicates, random_state=42).copy()

            max_id = df['customer_id'].str[1:].astype(int).max()
            duplicates['customer_id'] = ['D' + str(max_id + i + 1).zfill(5) for i in range(len(duplicates))]

            df = pd.concat([df, duplicates], ignore_index=True)
            print(f"✅ Added {num_duplicates} duplicate customer records")

        return df

    def _get_category_pricing(self, category: str) -> Dict[str, float]:
        pricing = {
            'Electronics': {'min': 24.99, 'max': 149.99, 'cost_multiplier': 0.6},
            'Fashion': {'min': 12.99, 'max': 89.99, 'cost_multiplier': 0.5},
            'Apparel': {'min': 8.99, 'max': 69.99, 'cost_multiplier': 0.45},
            'Health': {'min': 5.99, 'max': 59.99, 'cost_multiplier': 0.4},
            'Home': {'min': 14.99, 'max': 99.99, 'cost_multiplier': 0.55},
            'Sports': {'min': 16.99, 'max': 129.99, 'cost_multiplier': 0.6},
            'Home furnishings': {'min': 19.99, 'max': 149.99, 'cost_multiplier': 0.5},
            'Luxury goods': {'min': 39.99, 'max': 199.99, 'cost_multiplier': 0.65}
        }

        return pricing.get(category, {'min': 9.99, 'max': 99.99, 'cost_multiplier': 0.5})

    def _generate_sku(self, category: str, index: int) -> str:
        prefix = ''.join([c for c in category[:3] if c.isalpha()]).upper()
        return f"{prefix}-{index:06d}"

    def generate_products(self, num_products: int = 200) -> pd.DataFrame:
        print("Generating products...")
        products = []

        all_categories = list(self.categories)

        for i in tqdm(range(num_products), desc="Products"):
            category = random.choice(all_categories)
            pricing = self._get_category_pricing(category)

            price = round(random.uniform(pricing['min'], pricing['max']), 2)
            cost_multiplier = pricing['cost_multiplier'] * random.uniform(0.8, 1.2)
            cost = round(price * cost_multiplier, 2)

            name = f"{self.fake.word().capitalize()} {category.split()[0].lower()} {self.fake.word()}"

            description = (
                f"High-quality {category.lower()} designed for {self.fake.word()} and {self.fake.word()}. "
                f"Perfect for {self.fake.word()} and {self.fake.word()} applications. "
                f"{random.choice(['Premium', 'Durable', 'Eco-friendly', 'Innovative', 'Stylish'])} "
                f"design with {self.fake.color_name()} finish."
            )

            subcategory = f"{category} {self.fake.word().capitalize()}"

            base_quantity = int(1000 / (price ** 0.5))
            stock_quantity = max(10, int(random.normalvariate(base_quantity, base_quantity * 0.3)))

            is_active = random.random() < 0.9

            created_at = self.fake.date_time_between(start_date=self.start_date_2018, end_date=self.end_date_2024)
            last_updated = self.fake.date_time_between(start_date=created_at, end_date=self.end_date_2024)

            products.append({
                'product_id': i + 1000,
                'sku': self._generate_sku(category, i + 1),
                'name': name,
                'description': description,
                'category': category,
                'subcategory': subcategory,
                'price': price,
                'cost': cost,
                'stock_quantity': stock_quantity,
                'is_active': is_active,
                'created_at': created_at,
                'last_updated': last_updated
            })

        df = pd.DataFrame(products)

        discount_mask = np.random.random(len(df)) < 0.2
        df.loc[discount_mask, 'original_price'] = df.loc[discount_mask, 'price']
        df.loc[discount_mask, 'price'] = df.loc[discount_mask, 'price'] * np.random.uniform(0.6, 0.9,
                                                                                            size=discount_mask.sum())
        df['price'] = df['price'].round(2)

        print(f"✅ Generated {len(df)} products across {df['category'].nunique()} categories")
        return df

    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        print("Generating order headers...")

        orders = []
        order_id_counter = 1

        for _, customer in tqdm(customers_df.iterrows(), total=len(customers_df), desc="Orders"):
            order_count = self._get_customer_order_count(customer['join_date'])

            for i in range(order_count):
                order_date = self._generate_order_date(customer['join_date'])
                orders.append({
                    'order_id': f"ORD{order_id_counter:08d}",
                    'customer_id': customer['customer_id'],
                    'order_date': order_date,
                    'status': 'completed',
                    'payment_method': random.choices(
                        ['credit_card', 'paypal', 'bank_transfer'],
                        weights=[0.7, 0.2, 0.1]
                    )[0],
                    'subtotal': 0.0,
                    'tax': 0.0,
                    'shipping': 0.0,
                    'discount': 0.0,
                    'total': 0.0
                })
                order_id_counter += 1

        return pd.DataFrame(orders)

    def _get_customer_order_count(self, join_date: datetime) -> int:
        days_since_join = (self.end_date_2024 - join_date).days
        months_since_join = max(1, days_since_join // 30)

        if months_since_join < 3:
            return random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
        elif months_since_join < 12:
            return random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        else:
            return random.choices([2, 3, 4, 5, 6], weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def _generate_order_date(self, join_date: datetime) -> datetime:
        days_since_join = (self.end_date_2024 - join_date).days
        if days_since_join <= 0:
            return join_date

        days_offset = int(random.expovariate(1.0 / (days_since_join / 3)))
        days_offset = min(days_offset, days_since_join)

        hour = random.choices(
            range(24),
            weights=[0.01] * 6 + [0.04] * 6 + [0.1] * 6 + [0.15] * 6
        )[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        return join_date + timedelta(days=days_offset, hours=hour, minutes=minute, seconds=second)

    def generate_order_items(self, orders_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        print("Generating order items...")

        order_items = []
        item_id_counter = 1

        active_products = products_df[products_df['is_active'] == True]
        if active_products.empty:
            raise ValueError("No active products available for order items")

        product_ids = active_products['product_id'].tolist()
        orders_updated = orders_df.copy()

        for order_idx, order in tqdm(orders_df.iterrows(), total=len(orders_df), desc="Order items"):
            order_id = order['order_id']

            num_items = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
            selected_products = random.sample(product_ids, min(num_items, len(product_ids)))

            order_subtotal = 0.0

            for product_id in selected_products:
                product = products_df[products_df['product_id'] == product_id].iloc[0]

                max_qty = 3 if product['price'] < 100 else 1
                quantity = random.randint(1, max_qty)

                discount_rate = random.choices(
                    [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
                    weights=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
                )[0]

                unit_price = float(product['price'])
                discount = unit_price * discount_rate * quantity
                total_price = (unit_price * quantity) - discount

                order_items.append({
                    'order_item_id': item_id_counter,
                    'order_id': order_id,
                    'product_id': product_id,
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'total_price': total_price,
                    'discount': discount
                })

                order_subtotal += total_price
                item_id_counter += 1

            tax_rate = 0.08
            shipping = 0.0 if order_subtotal > 50 else 4.99
            tax = order_subtotal * tax_rate

            order_discount = 0.0
            if random.random() < 0.1:
                order_discount = order_subtotal * 0.1

            orders_updated.loc[orders_updated['order_id'] == order_id, [
                'subtotal', 'tax', 'shipping', 'discount', 'total'
            ]] = [
                round(order_subtotal, 2),
                round(tax, 2),
                round(shipping, 2),
                round(order_discount, 2),
                round(order_subtotal + tax + shipping - order_discount, 2)
            ]

        order_items_df = pd.DataFrame(order_items)
        return order_items_df, orders_updated

    def save_to_csv(self, data_dict):
        for name, df in data_dict.items():
            filepath = self.data_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df):,} rows to {filepath}")

    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        print("Generating complete e-commerce dataset...")

        print("\n=== Generating Customers ===")
        customers_df = self.generate_customers()

        print("\n=== Generating Products ===")
        products_df = self.generate_products()

        print("\n=== Generating Order Headers ===")
        orders_df = self.generate_orders(customers_df, products_df)

        print("\n=== Generating Order Items ===")
        order_items_df, orders_updated_df = self.generate_order_items(orders_df, products_df)

        customers_df['customer_id'] = customers_df['customer_id'].astype(str)
        products_df['product_id'] = products_df['product_id'].astype(int)
        orders_updated_df['order_id'] = orders_updated_df['order_id'].astype(str)
        orders_updated_df['customer_id'] = orders_updated_df['customer_id'].astype(str)
        order_items_df['order_item_id'] = order_items_df['order_item_id'].astype(int)
        order_items_df['order_id'] = order_items_df['order_id'].astype(str)
        order_items_df['product_id'] = order_items_df['product_id'].astype(int)

        return {
            'customers': customers_df,
            'products': products_df,
            'orders': orders_updated_df,
            'order_items': order_items_df
        }


def main():
    generator = EcommerceDataGenerator()
    print("Starting data generation...")

    data = generator.generate_all_data()
    generator.save_to_csv(data)

    print("\nData Generation Summary:")
    print("-" * 30)
    for name, df in data.items():
        print(f"{name.capitalize()}: {len(df):,} rows")

    print("\nData generation completed successfully!")


if __name__ == "__main__":
    main()
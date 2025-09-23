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
    The data is generated to simulate realistic e-commerce patterns with growth and plateau phases.

    Key Features:
    - Generates realistic customer data with retention patterns
    - Creates product catalog with category-based pricing
    - Simulates order history with seasonal variations
    - Handles data quality aspects like missing values and duplicates
    - Supports data generation for specific time periods (2018-2024)
    """

    def __init__(self, target_customers: int = 15000, target_revenue: float = 2300000,
                 missing_email_rate: float = 0.2):
        """
        Initialize the data generator with configuration parameters.

        Args:
            target_customers: Desired number of unique customers in the final year (2024)
            target_revenue: Target revenue for the peak year (2022) in euros
            missing_email_rate: Probability of generating a missing email (0.0 to 1.0)
        """
        self.target_customers = target_customers
        self.target_revenue = target_revenue
        self.missing_email_rate = missing_email_rate
        self.fake = Faker()
        self.data_dir = Path("C:/python/census_ecommerce/data/synthetic")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data structures
        self.shopify_data = self._load_shopify_data()
        self.categories = self._load_category_distribution()
        self.email_cache = {}
        self.monthly_stats = {}
        self.yearly_revenue = {}

        # State distribution data - POPRAWIONE
        self.state_distribution = self._get_state_distribution()

    def _get_state_distribution(self) -> Dict[str, float]:
        """POPRAWIONE: Teraz suma = 1.0"""
        # Top 10 states (55% of sales)
        top_states = {
            'California': 0.125, 'Texas': 0.090, 'New York': 0.085, 'Florida': 0.080,
            'Illinois': 0.045, 'Pennsylvania': 0.040, 'Ohio': 0.035, 'Georgia': 0.033,
            'North Carolina': 0.032, 'Michigan': 0.030
        }
        
        # Next 10 states (25% of sales)
        mid_states = {
            'New Jersey': 0.028, 'Virginia': 0.027, 'Washington': 0.026, 'Arizona': 0.025,
            'Massachusetts': 0.024, 'Indiana': 0.023, 'Tennessee': 0.022, 'Missouri': 0.021,
            'Maryland': 0.020, 'Wisconsin': 0.019
        }
        
        # Remaining states (20% of sales)
        other_states = [
            'Alabama', 'Alaska', 'Arkansas', 'Colorado', 'Connecticut', 'Delaware', 
            'Hawaii', 'Idaho', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 
            'Minnesota', 'Mississippi', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
            'New Mexico', 'North Dakota', 'Oklahoma', 'Oregon', 'Rhode Island', 
            'South Carolina', 'South Dakota', 'Utah', 'Vermont', 'West Virginia', 'Wyoming'
        ]
        
        # Calculate weight for each remaining state (20% / 30 states)
        other_weight = 0.20 / len(other_states)
        other_states_dict = {state: other_weight for state in other_states}
        
        # Combine all states - POPRAWIONE: używamy update zamiast **
        all_states = {}
        all_states.update(top_states)      # 55%
        all_states.update(mid_states)      # 25% 
        all_states.update(other_states_dict) # 20%
        
        # Verify the sum is exactly 1.0
        total_weight = sum(all_states.values())
        if abs(total_weight - 1.0) > 0.001:
            # Normalize if needed
            all_states = {k: v/total_weight for k, v in all_states.items()}
            
        return all_states

    def _load_shopify_data(self) -> pd.DataFrame:
        """
        Load and preprocess essential Shopify sales data from CSV file.

        Returns:
            DataFrame with essential columns:
            - date: Datetime of the record
            - repeat_rate: Estimated customer repeat rate (orders/customer) with anomaly in last 18 months
            - avg_order_value: Average order value in USD
            - sales_weight: Relative sales weight for the month
            - total_sales_usd: Total sales in USD (scaled to target revenue)
            - top_categories: Comma-separated list of top sales categories for the month

        Raises:
            FileNotFoundError: If the Shopify data file is not found
            ValueError: If required columns are missing from the data
        """
        try:
            # Define the path to the Shopify data file
            shopify_file = self.data_dir / "shopify_with_customers.csv"
            print(f"Loading Shopify data from: {shopify_file}")

            # Check if file exists
            if not shopify_file.exists():
                raise FileNotFoundError(f"Shopify data file not found at: {shopify_file}")

            # Define required columns and their new names
            column_mapping = {
                'Month': 'date',
                'Est. Customer repeat rate (orders/customer)': 'repeat_rate',
                'Avg. order value (USD)': 'avg_order_value',
                'Sales_Weight': 'sales_weight',
                'Total sales (USD mln)': 'total_sales_usd',
                'Top sales categories': 'top_categories'
            }

            # Read only the columns we need and rename them
            df = pd.read_csv(shopify_file, usecols=list(column_mapping.keys()), thousands=' ')
            df = df.rename(columns=column_mapping)

            # Convert and validate data types
            df['date'] = pd.to_datetime(df['date'])

            # Convert numeric columns to appropriate types
            numeric_cols = ['repeat_rate', 'avg_order_value', 'sales_weight', 'total_sales_usd']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Clean up top categories
            df['top_categories'] = df['top_categories'].str.strip()

            # Convert sales from millions to actual USD
            df['total_sales_usd'] = df['total_sales_usd'] * 1_000_000

            # Apply revenue scaling with plateau effect
            df = self._apply_revenue_scaling(df)

            # Select and reorder columns
            df = df[['date', 'repeat_rate', 'avg_order_value',
                     'sales_weight', 'total_sales_usd', 'top_categories']]

            print(f"Successfully loaded {len(df)} months of Shopify data")
            print(f"Total scaled revenue: {df['total_sales_usd'].sum():,.2f} USD")
            return df

        except Exception as e:
            print(f"Error loading Shopify data: {str(e)}")
            raise

    def _generate_order_dates(self, num_orders: int, year: int, month: int) -> List[str]:
        """
        Generate realistic order dates with weekday distribution.

        Args:
            num_orders: Number of dates to generate
            year: Year of the dates
            month: Month of the dates (1-12)

        Returns:
            List of date strings in 'YYYY-MM-DD' format
        """
        # Weekday weights (Monday=0 to Sunday=6)
        weekday_weights = {
            0: 0.16,  # Monday
            1: 0.15,  # Tuesday
            2: 0.14,  # Wednesday
            3: 0.14,  # Thursday
            4: 0.13,  # Friday
            5: 0.15,  # Saturday
            6: 0.13  # Sunday
        }

        # Get all days in the month
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        days_in_month = (datetime(next_year, next_month, 1) -
                         datetime(year, month, 1)).days

        # Generate dates with weekday distribution
        dates = []
        for _ in range(num_orders):
            # Choose a day based on weekday weights
            day = np.random.choice(range(days_in_month),
                                   p=[weekday_weights[datetime(year, month, d + 1).weekday()]
                                      for d in range(days_in_month)])

            # Add some randomness to the hour (8 AM to 10 PM)
            hour = np.random.choice(range(8, 22))
            minute = np.random.randint(0, 60)

            # Create datetime object
            order_datetime = datetime(year, month, day + 1, hour, minute)
            dates.append(order_datetime.strftime('%Y-%m-%d %H:%M'))

        return sorted(dates)

    def _generate_monthly_orders(self, year: int, month: int, total_sales: float,
                                 avg_order_value: float) -> pd.DataFrame:
        """
        Generate orders for a specific month with realistic order values and dates.

        Args:
            year: Year of the month
            month: Month number (1-12)
            total_sales: Total sales amount for the month
            avg_order_value: Target average order value

        Returns:
            DataFrame with generated orders containing:
            - order_id: Unique order ID
            - order_date: Date and time of the order
            - order_value: Value of the order in USD
            - status: Order status (always 'completed' for now)
            - payment_method: Payment method used
            - day_of_week: Name of the weekday
        """
        # Calculate number of orders (at least 1)
        num_orders = max(1, round(total_sales / avg_order_value))

        # Generate order values with normal distribution
        # Using 20% of avg_order_value as standard deviation for realistic variation
        std_dev = avg_order_value * 0.2
        order_values = np.random.normal(avg_order_value, std_dev, num_orders)

        # Ensure no negative values and round to 2 decimal places
        order_values = np.round(np.maximum(10.0, order_values), 2)

        # Adjust total to match exactly the target sales
        total_generated = np.sum(order_values)
        if total_generated > 0:
            scaling_factor = total_sales / total_generated
            order_values = np.round(order_values * scaling_factor, 2)

        # Generate realistic order dates with weekday distribution
        order_dates = self._generate_order_dates(num_orders, year, month)

        # Create orders DataFrame
        orders = []
        for i, (value, order_date) in enumerate(zip(order_values, order_dates), 1):
            order_dt = datetime.strptime(order_date, '%Y-%m-%d %H:%M')
            orders.append({
                'order_id': f"O{year}{month:02d}{i:05d}",
                'order_date': order_date,
                'order_value': value,
                'status': 'completed',
                'payment_method': random.choices(
                    ['credit_card', 'paypal', 'bank_transfer'],
                    weights=[0.7, 0.2, 0.1]
                )[0],
                'day_of_week': order_dt.strftime('%A')
            })

        return pd.DataFrame(orders)

    def _apply_revenue_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply two-step revenue scaling with a linear transition period.

        Args:
            df: DataFrame with 'date' and 'total_sales_usd' columns

        Returns:
            DataFrame with scaled revenue values
        """
        df = df.copy()

        # Define the periods
        first_period_end = pd.to_datetime('2022-06-30')
        transition_start = pd.to_datetime('2022-07-01')
        transition_end = pd.to_datetime('2023-03-31')
        second_period_start = pd.to_datetime('2023-04-01')

        # Scale factors
        first_scale = 3438.0  # For Jan 2018 - Jun 2022
        second_scale = 3860.0  # For Apr 2023 - Sep 2024

        # Apply first scale factor to initial period (Jan 2018 - Jun 2022)
        mask_first = df['date'] <= first_period_end
        df.loc[mask_first, 'scaled_sales'] = df.loc[mask_first, 'total_sales_usd'] / first_scale

        # Apply second scale factor to plateau period (Apr 2023 - Sep 2024)
        mask_second = df['date'] >= second_period_start
        df.loc[mask_second, 'scaled_sales'] = df.loc[mask_second, 'total_sales_usd'] / second_scale

        # Linear transition between the two periods (Jul 2022 - Mar 2023)
        transition_months = (transition_end - transition_start).days / 30.44  # Average month length
        for i, (_, row) in enumerate(df[(df['date'] > first_period_end) &
                                        (df['date'] < second_period_start)].iterrows()):
            # Calculate position in transition (0.0 to 1.0)
            progress = min(i / transition_months, 1.0)
            # Linear interpolation between scales
            current_scale = first_scale + (second_scale - first_scale) * progress
            df.at[row.name, 'scaled_sales'] = row['total_sales_usd'] / current_scale

        # Update the total_sales_usd column with scaled values
        df['total_sales_usd'] = df['scaled_sales'].round(2)
        df = df.drop(columns=['scaled_sales'])

        # Print summary
        first_period_sales = df[df['date'] <= first_period_end]['total_sales_usd'].sum()
        second_period_sales = df[df['date'] >= second_period_start]['total_sales_usd'].sum()
        transition_sales = df[(df['date'] > first_period_end) &
                              (df['date'] < second_period_start)]['total_sales_usd'].sum()

        print("\n=== Revenue Scaling Summary ===")
        print(f"Period 1 (2018-01 - 2022-06): {first_period_sales:,.2f} USD (scale: 1/{first_scale:.0f})")
        print(f"Transition (2022-07 - 2023-03): {transition_sales:,.2f} USD")
        print(f"Period 2 (2023-04 - 2024-09): {second_period_sales:,.2f} USD (scale: 1/{second_scale:.0f})")
        print(f"Total Scaled Revenue: {df['total_sales_usd'].sum():,.2f} USD")

        return df

    def _generate_monthly_category_weights(self, top_categories_str: str, all_categories: List[str]) -> Dict[
        str, float]:
        """
        Generate monthly weights for categories based on their position in top categories.

        Args:
            top_categories_str: Comma-separated string of top categories for the month
            all_categories: List of all possible categories

        Returns:
            Dictionary mapping each category to its weight for the month

        Raises:
            ValueError: If any top category is not found in all_categories
        """
        if not top_categories_str or pd.isna(top_categories_str):
            # If no top categories specified, return equal weights
            weight = 1.0 / len(all_categories) if all_categories else 1.0
            return {cat: weight for cat in all_categories}

        # Rozbijanie stringa na listę kategorii i normalizacja
        top_cats = [cat.strip().title() for cat in str(top_categories_str).split(',') if cat.strip()]

        # Walidacja, czy wszystkie kategorie z top_cats są w all_categories
        invalid_cats = set(top_cats) - set(all_categories)
        if invalid_cats:
            raise ValueError(f"Invalid top categories found: {invalid_cats}. "
                             f"Valid categories are: {all_categories}")

        n_top = len(top_cats)
        weights = {cat: 0.0 for cat in all_categories}

        # ---- TOP kategorie ----
        top_total = 0.5
        if n_top == 1:
            weights[top_cats[0]] = top_total
        elif n_top == 2:
            # Pierwsza musi mieć więcej niż druga
            split = random.uniform(0.26, 0.49)  # np. 26%–49% dla pierwszej
            weights[top_cats[0]] = split
            weights[top_cats[1]] = top_total - split
        elif n_top >= 3:
            # Zapewnienie, że pierwsza kategoria ma największą wagę
            first = random.uniform(0.3, 0.45)  # Pierwsza kategoria ma 30-45%
            rest = top_total - first
            # Dzielimy resztę na dwie części, ale upewniamy się, że druga część nie jest większa niż pierwsza
            second_share = random.uniform(0.3, 0.6)  # 30-60% reszty dla drugiej kategorii
            second = rest * second_share
            third = rest - second

            weights[top_cats[0]] = first
            weights[top_cats[1]] = second
            weights[top_cats[2]] = third

        # ---- Reszta kategorii ----
        other_cats = [cat for cat in all_categories if cat not in top_cats[:3]]
        if other_cats:
            other_total = 1.0 - sum(weights.values())
            splits = np.random.dirichlet([1] * len(other_cats)) * other_total
            for i, cat in enumerate(other_cats):
                weights[cat] = splits[i]

        # ---- Normalizacja (żeby suma była dokładnie 1.0) ----
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _load_monthly_stats(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Calculate and organize monthly statistics from the Shopify data.

        Returns:
            Dictionary mapping (year, month) tuples to their respective statistics
            with the following structure:
            {
                (year, month): {
                    'date': datetime object,
                    'repeat_rate': float,
                    'avg_order_value': float,
                    'sales_weight': float,
                    'total_sales': float,
                    'top_categories': str,
                    'category_weights': Dict[str, float]
                },
                ...
            }
        """
        if hasattr(self, '_monthly_stats_cache'):
            return self._monthly_stats_cache

        shopify_data = self._load_shopify_data()
        monthly_stats = {}

        for _, row in shopify_data.iterrows():
            year = row['date'].year
            month = row['date'].month

            # Generate category weights for this month
            category_weights = self._generate_monthly_category_weights(
                row['top_categories'],
                self.categories
            )

            # Store all relevant data for this month
            monthly_stats[(year, month)] = {
                'date': row['date'],
                'repeat_rate': row['repeat_rate'],
                'avg_order_value': row['avg_order_value'],
                'sales_weight': row['sales_weight'],
                'total_sales': row['total_sales_usd'],
                'top_categories': row['top_categories'],
                'category_weights': category_weights
            }

        # Apply repeat rate anomaly for last 18 months (reduce to ~1.8)
        if len(shopify_data) > 18:
            last_date = shopify_data['date'].max()
            anomaly_date = last_date - pd.DateOffset(months=18)
            anomaly_mask = shopify_data['date'] >= anomaly_date
            shopify_data.loc[anomaly_mask, 'repeat_rate'] = 1.8

            print(f"\n=== REPEAT RATE ANOMALY APPLIED ===")
            print(f"Anomaly period: {anomaly_date.strftime('%Y-%m')} to {last_date.strftime('%Y-%m')}")
            print("=" * 40)

        # Cache the results for future calls
        self._monthly_stats_cache = monthly_stats
        return monthly_stats

    def _load_category_distribution(self) -> List[str]:
        """
        Extract unique product categories from the top_categories column.
        Returns a list of unique category names in Title Case.

        Optimized to use vectorized pandas operations and efficient string handling.
        """
        # Drop NA values first to avoid unnecessary processing
        all_categories = self.shopify_data['top_categories'].dropna()
        categories = set()

        # Process each string only once
        for cats_str in all_categories:
            if pd.notna(cats_str):
                categories.update(
                    cat.strip().title()
                    for cat in cats_str.split(',')
                    if cat.strip()
                )

        return sorted(categories)  # Convert set to sorted list

    def _generate_email_from_name(self, name: str) -> str:
        """
        Generate a unique email address based on a customer's name.
        Uses patterns like: first.last@domain, flast@domain, first_last@domain
        """
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

        # Ensure email uniqueness
        if email_format in self.email_cache:
            counter = self.email_cache[email_format] + 1
            email = f"{email_format}{counter}@{random.choice(domains)}"
        else:
            counter = 0
            email = f"{email_format}@{random.choice(domains)}"

        self.email_cache[email_format] = counter
        return email

    def _generate_phone_number(self) -> str:
        # Generate area code and exchange (NPA-NXX)
        area_code = f"{random.randint(200, 999):03d}"  # 200-999
        exchange = f"{random.randint(200, 999):03d}"   # 200-999
        line = f"{random.randint(1000, 9999):04d}"      # 1000-9999
        
        # Randomly choose a format
        format_choice = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
        
        if format_choice == 1:
            return f"({area_code}) {exchange}-{line}"
        elif format_choice == 2:
            return f"{area_code}-{exchange}-{line}"
        else:
            return f"{area_code}{exchange}{line}"
            
    def _generate_zip_code(self, state: str) -> str:
        """
        Generate a realistic ZIP code for the given US state.
        
        Args:
            state: US state name
            
        Returns:
            str: 5-digit ZIP code as a string
        """
        # First digit ranges by region (1-9)
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
        
        # Get state abbreviation
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
        
        state_abbr = state_abbr.get(state, 'NY')  # Default to NY if state not found
        
        # Get ZIP code range for the state
        zip_range = zip_ranges.get(state_abbr, '100-999')  # Default to NY range
        start, end = map(int, zip_range.split('-'))
        
        # Generate a random ZIP code in the range
        zip_code = random.randint(start, end)
        
        # Format with leading zeros if needed
        return f"{zip_code:05d}"

    # ============================================
    # Core Data Generation Methods (4-Table Schema)
    # ============================================

    def _get_first_order_date(self, join_date: datetime) -> Optional[datetime]:
        """POPRAWIONE: Teraz 56% natychmiast, 44% później"""
        # 56% order immediately, 44% order later
        if random.random() < 0.56:  # 56% order immediately (same day)
            return join_date
        else:  # 44% order later (within 90 days)
            days_later = random.randint(1, 90)
            return join_date + timedelta(days=days_later)

    def generate_customers(self) -> pd.DataFrame:
        """POPRAWIONE: Uwzględnia tylko klientów z zamówieniami"""
        print("Generating customers...")
        
        customers = []
        states = list(self.state_distribution.keys())
        state_weights = list(self.state_distribution.values())
        
        # Generate only customers who will place orders
        # Since we want 15,000 customers WITH orders, we need to generate more initially
        # to account for the 20% that never order
        total_to_generate = int(self.target_customers / 0.8)  # Generate 20% more
        
        for i in range(total_to_generate):
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            full_name = f"{first_name} {last_name}"
            
            email = None if random.random() < self.missing_email_rate else self._generate_email_from_name(full_name)
            
            # Generate join date (2018-2024)
            join_year = random.randint(2018, 2024)
            join_date = self.fake.date_between(
                start_date=f"{join_year}-01-01",
                end_date=f"{join_year}-12-31"
            )
            
            # Get first order date - POPRAWIONE logika
            first_order_date = self._get_first_order_date(join_date)
            
            # Only include customers who placed orders
            if first_order_date is None:
                continue  # Skip customers who never order
                
            state = random.choices(states, weights=state_weights, k=1)[0]
            
            # Calculate loyalty score based on order history
            years_since_join = 2024 - join_date.year
            base_loyalty = min(0.9, years_since_join * 0.15)
            loyalty_score = round(np.random.normal(base_loyalty, 0.1), 2)
            loyalty_score = max(0, min(1, loyalty_score))
            
            # Generate last purchase date (after first order)
            last_purchase_date = self.fake.date_between(
                start_date=first_order_date,
                end_date=min(datetime.now(), datetime(2024, 12, 31))
            )
            
            # Generate address components
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
            
            # Stop when we have enough customers with orders
            if len(customers) >= self.target_customers:
                break
        
        df = pd.DataFrame(customers)
        df['join_date'] = pd.to_datetime(df['join_date'])
        df['first_order_date'] = pd.to_datetime(df['first_order_date'])
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        
        # Analysis
        total_generated = len(df)
        immediate_orders = (df['first_order_date'] == df['join_date']).sum()
        later_orders = total_generated - immediate_orders
        
        print(f"✅ Generated {total_generated} customers WITH orders")
        print(f"📊 {immediate_orders} ({immediate_orders/total_generated*100:.1f}%) ordered immediately")
        print(f"📊 {later_orders} ({later_orders/total_generated*100:.1f}%) ordered later")
        
        return df

    def _get_category_pricing(self, category: str) -> Dict[str, float]:
        """
        Get pricing parameters for a product category based on Shopify data.
        Uses exact category names as they appear in the Shopify data.
        
        Args:
            category: Product category from Shopify data (case-sensitive)
            
        Returns:
            Dictionary with min_price, max_price, and cost_multiplier
        """
        # Exact category pricing from Shopify data
        pricing = {
            # Main categories
            'Electronics': {'min': 49.99, 'max': 2499.99, 'cost_multiplier': 0.6},
            'Fashion': {'min': 14.99, 'max': 299.99, 'cost_multiplier': 0.5},
            'Apparel': {'min': 9.99, 'max': 199.99, 'cost_multiplier': 0.45},  # More specific than Fashion
            'Health': {'min': 4.99, 'max': 199.99, 'cost_multiplier': 0.4},
            'Home': {'min': 19.99, 'max': 999.99, 'cost_multiplier': 0.55},
            'Sports': {'min': 24.99, 'max': 499.99, 'cost_multiplier': 0.65},
            'Home furnishings': {'min': 29.99, 'max': 1499.99, 'cost_multiplier': 0.5},
            'Luxury goods': {'min': 99.99, 'max': 9999.99, 'cost_multiplier': 0.7}
        }
        
        # Return exact match or default pricing
        return pricing.get(category, {'min': 9.99, 'max': 99.99, 'cost_multiplier': 0.5})

    def _generate_sku(self, category: str, index: int) -> str:
        """Generate a realistic SKU based on category and index."""
        # Get first 3 letters of category (uppercase)
        prefix = ''.join([c for c in category[:3] if c.isalpha()]).upper()
        # Add 6-digit number with leading zeros
        return f"{prefix}-{index:06d}"

    def generate_products(self, num_products: int = 200) -> pd.DataFrame:
        """
        Generate product catalog with realistic attributes and pricing.
        
        Args:
            num_products: Number of products to generate (default: 200)
            
        Returns:
            DataFrame with columns: 
            - product_id: Unique product identifier
            - sku: Stock Keeping Unit
            - name: Product name
            - description: Detailed product description
            - category: Product category
            - subcategory: Product subcategory
            - price: Selling price (USD)
            - cost: Product cost (USD)
            - stock_quantity: Available quantity in stock
            - is_active: Whether product is active
            - created_at: Timestamp when product was added
            - last_updated: Timestamp of last update
        """
        print("Generating products...")
        products = []
        
        # Get all available categories
        all_categories = list(self.categories)
        
        # Generate products
        for i in tqdm(range(num_products), desc="Products"):
            # Select category based on weights
            category = random.choices(
                population=all_categories,
                weights=[self.categories[cat] for cat in all_categories],
                k=1
            )[0]
            
            # Get pricing for this category
            pricing = self._get_category_pricing(category)
            
            # Generate realistic price within category range
            price = round(random.uniform(pricing['min'], pricing['max']), 2)
            
            # Calculate cost (40-70% of price, with some randomness)
            cost_multiplier = pricing['cost_multiplier'] * random.uniform(0.8, 1.2)
            cost = round(price * cost_multiplier, 2)
            
            # Generate product name based on category
            name = f"{self.fake.word().capitalize()} {category.split()[0].lower()} {self.fake.word()}"
            
            # Generate description
            description = (
                f"High-quality {category.lower()} designed for {self.fake.word()} and {self.fake.word()}. "
                f"Perfect for {self.fake.word()} and {self.fake.word()} applications. "
                f"{random.choice(['Premium', 'Durable', 'Eco-friendly', 'Innovative', 'Stylish'])} "
                f"design with {self.fake.color_name()} finish."
            )
            
            # Generate subcategory (simplified for now)
            subcategory = f"{category} {self.fake.word().capitalize()}"
            
            # Generate stock quantity (more for cheaper items)
            base_quantity = int(1000 / (price ** 0.5))
            stock_quantity = max(10, int(random.normalvariate(base_quantity, base_quantity * 0.3)))
            
            # 90% chance of being active
            is_active = random.random() < 0.9
            
            # Generate timestamps
            created_at = self.fake.date_time_between(start_date='-3y', end_date='now')
            last_updated = self.fake.date_time_between(start_date=created_at, end_date='now')
            
            products.append({
                'product_id': i + 1000,  # Start from 1000
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
        
        # Create DataFrame
        df = pd.DataFrame(products)
        
        # Add some products with discounts (sale items)
        discount_mask = np.random.random(len(df)) < 0.2  # 20% of products on sale
        df.loc[discount_mask, 'original_price'] = df.loc[discount_mask, 'price']
        df.loc[discount_mask, 'price'] = df.loc[discount_mask, 'price'] * np.random.uniform(0.6, 0.9, size=discount_mask.sum())
        df['price'] = df['price'].round(2)
        
        print(f"✅ Generated {len(df)} products across {df['category'].nunique()} categories")
        print("📊 Category distribution:")
        print(df['category'].value_counts().to_string())
        print(f"💰 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        return df

    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Generate order data with associated order items.

        Args:
            customers_df: DataFrame of customers from generate_customers()
            products_df: DataFrame of products from generate_products()

        Returns:
            Tuple of (orders_df, order_items_df) where:
            - orders_df contains order headers (one row per order)
            - order_items_df contains individual line items (multiple per order)

        Note:
            - Orders will be generated with realistic patterns over time
            - Order items will reference valid product_ids
            - Order totals will be calculated automatically
        """
        # TODO: Implement order and order items generation
        return pd.DataFrame(), pd.DataFrame()

    def generate_order_items(self, orders_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate order items data with associated order and product information.

        Args:
            orders_df: DataFrame of orders from generate_orders()
            products_df: DataFrame of products from generate_products()

        Returns:
            DataFrame with columns: order_item_id, order_id, product_id, quantity,
            unit_price, total_price, discount, created_at, last_updated
        """
        # TODO: Implement order items generation
        return pd.DataFrame()

    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete e-commerce dataset with all four tables.

        Returns:
            Dictionary containing four DataFrames:
            - 'customers': Customer information
            - 'products': Product catalog
            - 'orders': Order headers
            - 'order_items': Individual order line items

        Note:
            Tables will have proper foreign key relationships for Power BI
        """
        print("Generating customers...")
        customers_df = self.generate_customers()

        print("Generating products...")
        products_df = self.generate_products()

        print("Generating orders and order items...")
        orders_df, order_items_df = self.generate_orders(customers_df, products_df)

        print("Generating order items...")
        order_items_df = self.generate_order_items(orders_df, products_df)

        return {
            'customers': customers_df,
            'products': products_df,
            'orders': orders_df,
            'order_items': order_items_df
        }

    def save_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: Path = None) -> None:
        """
        Save generated data to CSV files.

        Args:
            data: Dictionary of DataFrames from generate_all_data()
            output_dir: Directory to save files (defaults to self.data_dir)
        """
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            file_path = output_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} {name} to {file_path}")


def main():
    """Main function to run the data generation process."""
    # Initialize the data generator
    generator = EcommerceDataGenerator()

    # Generate and save all data
    print("Starting data generation...")

    # Generate all data tables
    data = generator.generate_all_data()

    # Save to CSV files
    generator.save_to_csv(data)

    # Print summary
    print("\nData Generation Summary:")
    print("-" * 30)
    for name, df in data.items():
        print(f"{name.capitalize()}: {len(df):,} rows")

    print("\nData generation completed successfully!")


if __name__ == "__main__":
    main()

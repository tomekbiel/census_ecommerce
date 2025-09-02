    def generate_orders(self, customers_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
       
        orders = []
        order_id = 1
        # Initialize total_revenue

        # Filter active products only
        active_products = products_df[products_df['is_active'] == True].copy()
        date_range = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')

        # Pre-calculate customer counts by join date
        customers_df['join_year'] = pd.to_datetime(customers_df['join_date']).dt.year
        customers_by_year = customers_df.groupby('join_year').size().to_dict()

        # Initialize yearly tracking
        self.yearly_revenue = {year: 0 for year in range(2018, 2025)}  # Instance variable for external access
        yearly_orders = {year: 0 for year in range(2018, 2025)}  # Local variable for internal use only

        # Initialize order template

        for date in tqdm(date_range, desc="Generating orders 2018-2024"):
            year, month = date.year, date.month
            monthly_params = self._get_monthly_parameters(year, month)

            # Calculate base daily orders with growth/plateau pattern
            if year <= 2022:  # Growth phase
                # Scale orders based on year to reach target in 2022
                year_factor = {
                    2018: 0.15,  # Startup
                    2019: 0.35,  # Early growth
                    2020: 0.6,  # Gaining traction
                    2021: 0.85,  # Approaching peak
                    2022: 1.0  # Peak
                }.get(year, 1.0)
            else:  # Plateau phase (2023-2024)
                # Calculate how much we've already generated for this year
                year_start = pd.Timestamp(datetime(date.year, 1, 1))
                days_elapsed = (date - year_start).days
                year_progress = days_elapsed / 365.0

                target_yearly_revenue = self.yearly_revenue[2022]  # Target is 2022's revenue
                current_yearly_revenue = self.yearly_revenue[year]

                # If we've already hit our target for this year, don't generate more orders
                if current_yearly_revenue >= target_yearly_revenue * 1.05:  # Allow 5% over target
                    continue

                # Adjust factor down as we approach the target
                remaining_revenue = max(0, target_yearly_revenue * 1.05 - current_yearly_revenue)
                days_remaining = 365 - days_elapsed

                if days_remaining > 0:
                    # Calculate how much we can spend per remaining day to hit target
                    daily_budget = remaining_revenue / days_remaining

                    # Reduce the year factor to slow down order generation
                    year_factor = min(0.8, daily_budget / 1000)  # Adjust divisor based on your average order value
                else:
                    year_factor = 0.1  # Minimal orders if we're at the end of the year

            # Calculate eligible customers (who could have ordered by this date)
            eligible_years = [y for y in customers_by_year.keys() if y <= year]
            eligible_customers = sum(customers_by_year[y] for y in eligible_years)

            # Calculate daily order volume with seasonal adjustments
            base_daily_orders = (eligible_customers * monthly_params['repeat_rate'] / 365) * monthly_params[
                'sales_weight']

            # For plateau years, we need to be more aggressive about capping orders
            if year > 2022:
                # Reduce the base number of orders to slow down growth
                base_daily_orders *= 0.7  # 30% reduction in order volume

                # Calculate how much revenue we've already generated this year
                year_start = pd.Timestamp(datetime(date.year, 1, 1))
                year_progress = (date - year_start).days / 365.0
                target_yearly_revenue = self.yearly_revenue[2022]  # Target is 2022's revenue

                # If we're already at or above target, significantly reduce orders
                if self.yearly_revenue[year] >= target_yearly_revenue * 0.95:  # 95% of target
                    base_daily_orders *= 0.3  # 70% reduction if we're close to target

            daily_orders = int(base_daily_orders * year_factor)
            daily_orders = max(1, min(daily_orders, 300))  # Cap at 300 orders/day

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

                # Generate order items with realistic quantities and values
                # Adjust number of items based on year (slightly more items per order in later years)
                base_items = 1.2 if year <= 2020 else 1.5
                num_items = min(int(np.random.poisson(lam=base_items) + 1), 10)

                order_total = 0.0
                total_quantity = 0
                total_discount = 0.0
                categories = []
                products_info = []

                # Adjust discount rate based on year (higher discounts in later years to stimulate sales)
                base_discount = 0.1 if year <= 2020 else 0.15
                if year >= 2023:  # Even higher discounts during plateau
                    base_discount = 0.18

                for _ in range(num_items):
                    current_month_categories = monthly_params['top_categories']
                    suitable_products = active_products[
                        active_products['category'].isin([cat for cat in self.categories
                                                          if any(
                                c.lower() in cat.lower() for c in current_month_categories)])
                    ]

                    if len(suitable_products) > 0:
                        # In later years, slightly favor higher-priced items
                        if year >= 2021:
                            product = suitable_products.nlargest(10, 'price').sample(1).iloc[0]
                        else:
                            product = suitable_products.sample(1).iloc[0]
                    else:
                        product = active_products.sample(1).iloc[0]

                    quantity = max(1, np.random.poisson(lam=1.8))  # Increased quantity
                    discount = round(random.uniform(0, 0.3), 2)

                    # Adjust price based on year (slight inflation over time)
                    # Cap inflation at 2022 levels for plateau years
                    inflation_year = min(year, 2022)
                    price_multiplier = 1.0 + (inflation_year - 2018) * 0.03  # 3% annual price increase, capped at 2022
                    adjusted_price = product['price'] * price_multiplier

                    item_total = adjusted_price * quantity * (1 - discount)
                    order_total += item_total
                    total_quantity += quantity
                    total_discount += discount * quantity  # Track total discount amount

                    # Track revenue by year
                    self.yearly_revenue[year] += item_total

                    categories.append(product['category'])
                    products_info.append(f"{product['product_id']}(x{quantity})")

                    # Update order data with calculated values
                    order_data.update({
                        'total_amount': round(order_total, 2),
                        'item_count': num_items,
                        'total_quantity': total_quantity,
                        'avg_discount': round(total_discount / num_items, 2) if num_items > 0 else 0.0,
                        'product_categories': ','.join(sorted(set(categories))),
                        'products_list': '|'.join(products_info),
                        'status': 'completed'  # Explicitly set status for completed orders
                    })

                    # Add order to the list
                    orders.append(order_data)
                    order_id += 1
                    yearly_orders[year] += 1
                    total_revenue += order_total
                    continue

                # If we get here, it's a completed order, so ensure status is set
                order_data['status'] = 'completed'

                # Add the completed order to the list
                orders.append(order_data)
                order_id += 1
                yearly_orders[year] += 1
                total_revenue += order_total

        # Print yearly summary
        print("\n=== Yearly Order Summary ===")
        for year in range(2018, 2025):
            if yearly_orders[year] > 0:
                print(f"{year}: {yearly_orders[year]:,} orders (€{self.yearly_revenue[year]:,.2f})")

        # Convert to DataFrame and ensure proper types
        orders_df = pd.DataFrame(orders)

        # Ensure proper date format
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

        # Ensure numeric columns are float
        numeric_cols = ['total_amount', 'avg_discount']
        for col in numeric_cols:
            if col in orders_df.columns:
                orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce').fillna(0.0)

        # Ensure integer columns are int
        int_cols = ['item_count', 'total_quantity']
        for col in int_cols:
            if col in orders_df.columns:
                orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce').fillna(0).astype(int)

        print("\n=== Order Generation Summary ===")
        print(f"Total orders generated: {len(orders_df):,}")
        print(f"Total revenue: €{orders_df['total_amount'].sum():,.2f}")
        print(f"Average order value: €{orders_df['total_amount'].mean():.2f}")

        # Print status distribution
        if 'status' in orders_df.columns:
            status_counts = orders_df['status'].value_counts()
            print("\nOrder Status Distribution:")
            print(status_counts)

        return orders_df
import pandas as pd
from datetime import datetime
import numpy as np

# Configuration - update this path to match your local file location
INPUT_CSV_PATH = "C:/python/census_ecommerce/data/synthetic/shopify_reports_2018-2024.csv"
OUTPUT_CSV_PATH = "C:/python/census_ecommerce/data/synthetic/shopify_monthly_reports_2018-2024.csv"


def load_quarterly_data(filepath):
    """Load quarterly Shopify report from CSV file"""
    df = pd.read_csv(filepath)
    # Convert relevant columns to numeric (handling any percentage signs or other non-numeric chars)
    numeric_cols = ['Total sales (USD mln)', 'GMV (USD bln)', 'Gross profit (USD mln)',
                    'Operating profit (USD mln)', 'Avg. order value (USD)',
                    'Active stores (mln)', 'Est. Customer repeat rate (orders/customer)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    return df


def generate_monthly_distribution(row):
    """Generate monthly distribution for a quarterly record based on promotions"""
    quarter = row['Quarter']
    year = int(quarter.split()[0])
    q = int(quarter.split()[1][1:])
    promotion = row['Promotions (description)']

    # Base monthly distribution without promotions (default weights)
    base_dist = {
        1: {'weights': [0.3, 0.3, 0.4], 'peak_month': None},  # Q1
        2: {'weights': [0.33, 0.33, 0.34], 'peak_month': None},  # Q2
        3: {'weights': [0.4, 0.35, 0.25], 'peak_month': None},  # Q3
        4: {'weights': [0.25, 0.6, 0.15], 'peak_month': 2}  # Q4 (Nov=Black Friday)
    }[q]

    # Adjust for specific promotions
    if 'Black Friday' in promotion:
        base_dist['weights'] = [0.15, 0.7, 0.15]  # 70% in November
        base_dist['peak_month'] = 2
    elif 'Prime Day' in promotion:
        base_dist['weights'] = [0.6, 0.25, 0.15]  # 60% in July
        base_dist['peak_month'] = 1
    elif 'Back to School' in promotion:
        base_dist['weights'] = [0.2, 0.5, 0.3]  # 50% in August
        base_dist['peak_month'] = 2
    elif 'Summer sales' in promotion:
        base_dist['weights'] = [0.25, 0.4, 0.35]  # 40% in May
        base_dist['peak_month'] = 2
    elif 'Spring sales' in promotion:
        base_dist['weights'] = [0.25, 0.45, 0.3]  # 45% in February
        base_dist['peak_month'] = 2

    # Get actual month numbers for the quarter
    month_numbers = {
        1: [1, 2, 3],  # Q1: Jan, Feb, Mar
        2: [4, 5, 6],  # Q2: Apr, May, Jun
        3: [7, 8, 9],  # Q3: Jul, Aug, Sep
        4: [10, 11, 12]  # Q4: Oct, Nov, Dec
    }[q]

    return month_numbers, base_dist['weights'], base_dist['peak_month']


def create_monthly_records(quarterly_df):
    """Create monthly records from quarterly data"""
    monthly_records = []

    for _, row in quarterly_df.iterrows():
        month_numbers, weights, peak_month = generate_monthly_distribution(row)
        total_sales = row['Total sales (USD mln)']

        # Extract year from the Quarter column (assuming format "2020 Q1")
        year = int(row['Quarter'].split()[0])

        # Distribute all quarterly metrics proportionally
        for i, month_num in enumerate(month_numbers):
            weight = weights[i]
            month_year = f"{year}-{month_num:02d}"  # Format as YYYY-MM

            # Create new monthly record
            monthly_record = {
                'Month': month_year,
                'Month_Number': month_num,
                'Year': year,
                'Quarter': row['Quarter'],
                'Is_Peak_Month': 1 if i == peak_month else 0,
                'Promotion_Type': row['Promotions (description)'],
                'Sales_Weight': weight
            }

            # Distribute all financial metrics
            for col in ['Total sales (USD mln)', 'GMV (USD bln)', 'Gross profit (USD mln)',
                        'Operating profit (USD mln)', 'Avg. order value (USD)']:
                monthly_record[col] = row[col] * weight

            # These metrics don't get distributed (keep original value)
            for col in ['Active stores (mln)', 'Est. Customer repeat rate (orders/customer)',
                        'Top sales categories']:
                monthly_record[col] = row[col]

            monthly_records.append(monthly_record)

    return pd.DataFrame(monthly_records)


def main():
    # Step 1: Load the quarterly data
    print(f"Loading quarterly data from {INPUT_CSV_PATH}")
    quarterly_df = load_quarterly_data(INPUT_CSV_PATH)

    # Step 2: Generate monthly records
    print("Generating monthly distribution...")
    monthly_df = create_monthly_records(quarterly_df)

    # Step 3: Verify quarterly sums match original data
    print("Verifying data consistency...")
    for col in ['Total sales (USD mln)', 'GMV (USD bln)', 'Gross profit (USD mln)']:
        original_sum = quarterly_df[col].sum()
        monthly_sum = monthly_df.groupby('Quarter')[col].sum().sum()
        assert abs(original_sum - monthly_sum) < 0.01, f"Quarterly sums don't match for {col}"

    # Step 4: Save results
    print(f"Saving monthly data to {OUTPUT_CSV_PATH}")
    monthly_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Process completed successfully!")
    print(f"Generated {len(monthly_df)} monthly records from {len(quarterly_df)} quarterly records")


if __name__ == "__main__":
    main()
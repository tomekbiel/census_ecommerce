# File: gen_faker/analyze_customers.py
import pandas as pd
from pathlib import Path


def calculate_total_customers():
    # Load the data
    data_dir = Path(__file__).parent.parent / "data/synthetic"
    input_file = data_dir / "shopify_with_customers.csv"
    df = pd.read_csv(input_file)

    # Calculate metrics
    total_customers = int(df['Estimated_customers'].sum())
    avg_monthly_customers = int(df['Estimated_customers'].mean())
    max_monthly_customers = int(df['Estimated_customers'].max())
    min_monthly_customers = int(df['Estimated_customers'].min())

    print(f"Analiza klientów w okresie {df['Month'].min()} - {df['Month'].max()}:")
    print(f"Łączna szacowana liczba klientów: {total_customers:,}")
    print(f"Średnia miesięczna liczba klientów: {avg_monthly_customers:,}")
    print(f"Maksymalna miesięczna liczba klientów: {max_monthly_customers:,}")
    print(f"Minimalna miesięczna liczba klientów: {min_monthly_customers:,}")


if __name__ == "__main__":
    calculate_total_customers()
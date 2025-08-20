# File: gen_faker/estimate_customers.py
import pandas as pd
from pathlib import Path


def add_customer_estimation(input_file, output_file=None):
    """
    Dodaje kolumnę z szacowaną liczbą klientów do pliku CSV.

    Args:
        input_file (str): Ścieżka do pliku wejściowego
        output_file (str, optional): Ścieżka do pliku wyjściowego. Jeśli None, nadpisuje plik wejściowy.
    """
    # Wczytaj dane
    df = pd.read_csv(input_file)

    # Konwersja jednostek i obliczenia
    df['Estimated_orders'] = (df['Total sales (USD mln)'] * 1_000_000 / df['Avg. order value (USD)']).round(0)
    df['Estimated_customers'] = (df['Estimated_orders'] / df['Est. Customer repeat rate (orders/customer)']).round(0)

    # Usuń kolumnę pomocniczą
    df = df.drop('Estimated_orders', axis=1)

    # Zapisz dane
    output_path = output_file if output_file else input_file
    df.to_csv(output_path, index=False)

    print(f"Zaktualizowane dane zapisano w: {output_path}")
    print(f"Średnia miesięczna liczba klientów: {df['Estimated_customers'].mean():.0f}")


if __name__ == "__main__":
    # Ścieżki względne od lokalizacji skryptu
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data/synthetic"

    input_file = data_dir / "shopify_monthly_reports_2018-2024.csv"
    output_file = data_dir / "shopify_with_customers.csv"

    # Uruchom funkcję
    add_customer_estimation(input_file, output_file)
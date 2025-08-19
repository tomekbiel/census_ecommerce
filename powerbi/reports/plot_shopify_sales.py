import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')  # lub 'TkAgg' jeśli masz zainstalowany Tkinter

def load_shopify_data(filepath=None):
    """Load the Shopify monthly data"""
    if filepath is None:
        filepath = Path("C:/python/census_ecommerce/data/synthetic/shopify_monthly_reports_2018-2024.csv")
    return pd.read_csv(filepath)


def prepare_data(df):
    """Prepare data for visualization"""
    df['Month'] = pd.to_datetime(df['Month'])
    df['Year'] = df['Month'].dt.year
    df['Month_Name'] = df['Month'].dt.strftime('%b')
    return df


def create_sales_plots(df, output_dir=None):
    """Create and save sales visualization plots"""
    if output_dir is None:
        output_dir = Path("C:/python/census_ecommerce/reports")
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))

    # 1. Monthly Sales Trend
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x='Month', y='Total sales (USD mln)')
    plt.title('Monthly Sales Trend (2018-2024)')

    # 2. Yearly Comparison
    plt.subplot(2, 2, 3)
    yearly_sales = df.groupby('Year')['Total sales (USD mln)'].sum().reset_index()
    sns.barplot(data=yearly_sales, x='Year', y='Total sales (USD mln)', palette='viridis')
    plt.title('Yearly Sales Comparison')

    # 3. Monthly Pattern
    plt.subplot(2, 2, 4)
    monthly_avg = df.groupby('Month_Name')['Total sales (USD mln)'].mean().reset_index()
    sns.barplot(data=monthly_avg, x='Month_Name', y='Total sales (USD mln)',
                order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Average Monthly Sales Pattern')
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / 'shopify_sales_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save data for Power BI
    powerbi_path = output_dir / 'shopify_sales_powerbi.csv'
    df.to_csv(powerbi_path, index=False)

    print(f"Plots saved to: {output_path}")
    print(f"Power BI data saved to: {powerbi_path}")


def main():
    # Load and prepare data
    df = load_shopify_data()
    df = prepare_data(df)

    # Create visualizations
    create_sales_plots(df)


if __name__ == "__main__":
    main()

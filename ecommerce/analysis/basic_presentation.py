# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

def main():
    # Define file paths
    processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    macro_file = processed_dir / 'ecommerce_analysis_latest.csv'

    # Load the FRED economic data
    print("Loading economic data from FRED...")
    try:
        # Load the data
        df = pd.read_csv(macro_file)
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure proper time series plotting
        df = df.sort_values('Date')
        
        print("\n=== Data Loaded Successfully ===")
        print(f"Time period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
        print(f"Number of records: {len(df)}")
        
        # Basic information about the data
        print("\n=== Data Overview ===")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataFrame Info:")
        print(df.info())
        
        # Basic statistics for numerical columns
        print("\n=== Basic Statistics ===")
        print(df.describe())
        
        # Time series visualization
        plot_time_series(df)
        
        # Correlation analysis
        plot_correlation_heatmap(df)
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("\nPlease make sure you have run ecommerce_data_analysis.py first to generate the data.")

def plot_time_series(df):
    """Create time series plots for key economic indicators"""
    print("\n=== Time Series Analysis ===")
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Plot 1: E-commerce and Total Retail Sales
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Ecommerce_Retail_Sales_Millions'], 
             label='E-commerce Sales (Millions $)', color='blue')
    plt.plot(df['Date'], df['Retail_Sales_Total_Millions'], 
             label='Total Retail Sales (Millions $)', color='orange')
    plt.title('E-commerce vs Total Retail Sales Over Time')
    plt.ylabel('Sales (Millions $)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Unemployment Rate and Consumer Sentiment
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(df['Date'], df['Unemployment_Rate'], 
             label='Unemployment Rate (%)', color='red')
    ax2.plot(df['Date'], df['Consumer_Sentiment'], 
             label='Consumer Sentiment', color='green')
    
    ax1.set_ylabel('Unemployment Rate (%)', color='red')
    ax2.set_ylabel('Consumer Sentiment', color='green')
    ax1.set_title('Unemployment Rate and Consumer Sentiment')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Create a correlation heatmap for key economic indicators"""
    print("\n=== Correlation Analysis ===")
    
    # Select columns for correlation analysis
    corr_columns = [
        'Ecommerce_Retail_Sales_Millions',
        'Retail_Sales_Total_Millions',
        'Unemployment_Rate',
        'Consumer_Sentiment',
        'Disposable_Income',
        'Personal_Consumption_Expenditures'
    ]
    
    # Calculate correlation matrix
    corr = df[corr_columns].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
                linewidths=0.5, linecolor='white')
    plt.title('Correlation Matrix of Economic Indicators')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print strongest correlations
    print("\nStrongest Correlations:")
    corr_pairs = corr.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs != 1]  # Remove self-correlations
    print(corr_pairs.head(5))

if __name__ == "__main__":
    main()
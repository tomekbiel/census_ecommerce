"""
E-commerce Data Analysis Script
-------------------------------
This script fetches e-commerce and related economic data from FRED,
processes it, and exports to CSV for analysis in Power BI.
"""
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import pandas_datareader.data as web
from dotenv import load_dotenv

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# FRED API key (optional but recommended for higher rate limits)
FRED_API_KEY = os.getenv('FRED_API_KEY', None)

# FRED Series IDs for economic data - comprehensive e-commerce and retail data
FRED_SERIES = {
    # E-commerce and Total Retail
    'ECOMSA': 'Ecommerce_Retail_Sales_Millions',  # E-commerce Retail Sales (Millions of Dollars, Monthly)
    'RSXFS': 'Retail_Sales_Total_Millions',       # Retail Sales: Total (Millions of Dollars, Monthly)
    'ECOMPCT': 'Ecommerce_Percent_Of_Total_Retail',  # E-commerce as Percent of Total Retail Sales
    
    # Retail Categories (NAICS)
    'MRTSSM44X72USN': 'Retail_Food_Beverage_Stores',  # Retail Trade: Food and Beverage Stores
    'MRTSSM4451USN': 'Retail_Grocery_Stores',        # Retail Trade: Grocery Stores
    'MRTSSM448USN': 'Retail_Clothing_Stores',        # Retail Trade: Clothing and Clothing Accessory Stores
    'MRTSSM452USN': 'Retail_General_Merchandise',    # Retail Trade: General Merchandise Stores
    'MRTSSM453USN': 'Retail_Misc_Store_Retailers',   # Retail Trade: Miscellaneous Store Retailers
    'MRTSSM4541USN': 'Retail_Electronic_Shopping',   # Retail Trade: Electronic Shopping and Mail-Order Houses
    
    # E-commerce by Category (Quarterly, Seasonally Adjusted)
    'ECOM442XAY': 'Ecomm_Furniture_Home_Furnishings',  # E-commerce: Furniture and Home Furnishings
    'ECOM4422XAY': 'Ecomm_Home_Furnishings',          # E-commerce: Home Furnishings Stores
    'ECOM4431XAY': 'Ecomm_Electronics_Appliances',    # E-commerce: Electronics and Appliance Stores
    'ECOM4441XAY': 'Ecomm_Building_Materials',        # E-commerce: Building Material and Garden Equipment
    'ECOM4451XAY': 'Ecomm_Food_Beverage',             # E-commerce: Food and Beverage Stores
    'ECOM4481XAY': 'Ecomm_Clothing_Accessories',      # E-commerce: Clothing and Clothing Accessories
    'ECOM4511XAY': 'Ecomm_Sporting_Goods',            # E-commerce: Sporting Goods, Hobby, Book, and Music Stores
    'ECOM4532XAY': 'Ecomm_Office_Supplies',           # E-commerce: Office Equipment and Supplies
    'ECOM4541XAY': 'Ecomm_Electronic_Shopping',       # E-commerce: Electronic Shopping and Mail-Order Houses
    
    # Economic Indicators
    'UNRATE': 'Unemployment_Rate',                    # Unemployment Rate
    'CPIAUCSL': 'CPI_All_Items',                      # Consumer Price Index for All Urban Consumers
    'RSAFS': 'Retail_Sales_Excl_Auto',                # Retail Sales Excluding Autos
    'RETAILMPCSMSA': 'Retail_Sales_Index',            # Retail Sales Index (2012=100)
    
    # Consumer Confidence and Spending
    'UMCSENT': 'Consumer_Sentiment',                  # University of Michigan: Consumer Sentiment
    'DSPIC96': 'Disposable_Income',                   # Real Disposable Personal Income
    'PCE': 'Personal_Consumption_Expenditures',       # Personal Consumption Expenditures
    
    # E-commerce Growth Rates
    'ECOMYOY': 'Ecomm_Year_Over_Year_Growth',         # E-commerce Year-over-Year Growth Rate
    'ECOMQ': 'Ecomm_Quarterly_Growth',                # E-commerce Quarterly Growth Rate
}

def fetch_fred_data(series_dict, start_date, end_date):
    """
    Fetch multiple economic indicators from FRED.
    
    Args:
        series_dict (dict): Dictionary of {series_id: column_name} pairs
        start_date (datetime): Start date for the data
        end_date (datetime): End date for the data
        
    Returns:
        pd.DataFrame: Combined DataFrame with all series
    """
    all_data = []
    
    for series_id, col_name in series_dict.items():
        try:
            print(f"Fetching {col_name} ({series_id})...")
            data = web.DataReader(
                series_id, 
                'fred', 
                start_date, 
                end_date,
                api_key=FRED_API_KEY
            )
            
            if not data.empty:
                data = data.rename(columns={series_id: col_name})
                all_data.append(data)
            else:
                print(f"  No data for {series_id}")
                
        except Exception as e:
            print(f"  Error fetching {series_id}: {e}")
    
    # Combine all series into a single DataFrame
    if all_data:
        # Outer join on the index (date)
        df = pd.concat(all_data, axis=1)
        return df
    return pd.DataFrame()

def clean_data(df):
    """Clean and process the raw FRED data for Power BI analysis."""
    if df.empty:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create a date column from the index
    df['Date'] = df.index
    
    # Forward fill missing values for monthly data
    df = df.ffill()
    
    # Calculate additional metrics
    if all(col in df.columns for col in ['Ecommerce_Retail_Sales_Millions', 'Retail_Sales_Total_Millions']):
        df['Ecommerce_Percent_Of_Total'] = (
            df['Ecommerce_Retail_Sales_Millions'] / df['Retail_Sales_Total_Millions']
        ) * 100
    
    # Calculate year and month for time-based analysis in Power BI
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Month_Name'] = df['Date'].dt.month_name()
    df['Quarter_Year'] = df['Date'].dt.to_period('Q').astype(str)
    
    # Calculate year-over-year growth for key metrics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['Year', 'Month', 'Quarter']:
            # Calculate monthly change
            df[f'{col}_MoM_Change'] = df[col].pct_change() * 100
            
            # Calculate year-over-year change
            if len(df) > 12:  # Ensure we have enough data for YoY
                df[f'{col}_YoY_Change'] = df[col].pct_change(periods=12) * 100
    
    # Reorder columns to have Date and time-related columns first
    date_cols = ['Date', 'Year', 'Month', 'Month_Name', 'Quarter', 'Quarter_Year']
    other_cols = [col for col in df.columns if col not in date_cols]
    df = df[date_cols + other_cols]
    
    # Sort by date to ensure proper time series
    df = df.sort_values('Date')
    
    return df

def save_to_csv(df, filename):
    """Save DataFrame to CSV file."""
    if df is None or df.empty:
        print("No data to save.")
        return False
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        csv_path = output_dir / filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nData successfully saved to {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def main():
    print("E-commerce Data Analysis Tool")
    print("============================\n")
    
    if not FRED_API_KEY:
        print("Note: FRED_API_KEY not found in .env file. Using public access (rate limited).\n")
    
    # Set date range (from January 1, 2018 to current date)
    end_date = datetime.now()
    start_date = datetime(2018, 1, 1)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...\n")
    
    # Fetch the data
    df = fetch_fred_data(FRED_SERIES, start_date, end_date)
    
    if df.empty:
        print("\nNo data was fetched. Please check your internet connection or FRED API key.")
        return
    
    # Clean and process the data
    print("\nProcessing data...")
    df_clean = clean_data(df)
    
    # Check if we have any data
    if df_clean.empty:
        print("\nNo valid data available for the selected time period.")
        return
    
    # Display summary
    print("\nData Summary:")
    print("------------")
    
    # Check if we have date information
    if not df_clean.empty and 'Date' in df_clean.columns:
        print(f"Time Period: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')}")
    
    print(f"Number of records: {len(df_clean)}")
    
    # Show available columns
    if not df_clean.empty:
        print("\nAvailable columns:")
        for col in df_clean.columns:
            print(f"- {col}")
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ecommerce_analysis_{timestamp}.csv"
    
    if save_to_csv(df_clean, csv_filename):
        print("\nData processing completed successfully!")
        print(f"You can now import '{csv_filename}' into Power BI for visualization.")
        
        # Save a sample of the data for reference
        sample_file = "ecommerce_analysis_latest.csv"
        save_to_csv(df_clean, sample_file)
        print(f"A copy has been saved as '{sample_file}' for easy access.")
    else:
        print("\nFailed to save data to CSV.")

if __name__ == "__main__":
    main()

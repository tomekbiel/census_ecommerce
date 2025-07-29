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

# FRED Series IDs for economic data - focusing on most reliable series
FRED_SERIES = {
    'ECOMSA': 'Ecommerce_Retail_Sales_Millions',  # E-commerce Retail Sales (Millions of Dollars, Monthly)
    'RSXFS': 'Retail_Sales_Total_Millions',       # Retail Sales: Total (Millions of Dollars, Monthly)
    'ECOMPCT': 'Ecommerce_Percent_Of_Total_Retail',  # E-commerce as Percent of Total Retail Sales
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
    """Clean and process the raw FRED data."""
    if df.empty:
        return df
    
    # Convert index to datetime if it's not already
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Forward fill missing values for monthly data
    df = df.ffill()
    
    # Reset index to make Date a column
    df = df.reset_index().rename(columns={'index': 'Date'})
    
    # Calculate additional metrics if we have the required columns
    if 'Ecommerce_Retail_Sales_Millions' in df.columns and 'Retail_Sales_Total_Millions' in df.columns:
        df['Ecommerce_Percent_Of_Total'] = (
            df['Ecommerce_Retail_Sales_Millions'] / df['Retail_Sales_Total_Millions']
        ) * 100
    
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
    
    # Set date range (last 5 years)
    end_date = datetime.now()
    start_date = datetime(end_date.year - 5, 1, 1)
    
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

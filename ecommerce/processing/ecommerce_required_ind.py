"""
E-commerce Required Indicators Filter
-----------------------------------
This script filters the e-commerce data to include only the required indicators
and date range (January 2018 - December 2024).
"""
import os
import pandas as pd
from datetime import datetime
import glob

# Required indicators (column names in the CSV)
REQUIRED_INDICATORS = [
    'Ecommerce_Retail_Sales_Millions',
    'Retail_Sales_Total_Millions',
    'Ecommerce_Percent_Of_Total_Retail',
    'Unemployment_Rate',
    'Consumer_Sentiment',
    'Disposable_Income',
    'Personal_Consumption_Expenditures'  # Note: Fixed typo from your list (Expenditure -> Expenditures)
]

def find_latest_csv():
    """Find the most recent CSV file in the data/processed directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
    files = glob.glob(os.path.join(processed_dir, 'ecommerce_analysis_*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {processed_dir}")
    return max(files, key=os.path.getmtime)

def filter_and_save_data():
    """Filter the data and save the result."""
    try:
        # Find the latest CSV file
        input_file = find_latest_csv()
        print(f"Processing file: {input_file}")
        
        # Read the data
        df = pd.read_csv(input_file, parse_dates=['Date'])
        
        # Ensure we have the required columns
        required_cols = ['Date'] + [col for col in REQUIRED_INDICATORS if col in df.columns]
        missing_cols = [col for col in REQUIRED_INDICATORS if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: The following columns were not found: {', '.join(missing_cols)}")
        
        # Filter columns
        df_filtered = df[required_cols].copy()
        
        # Filter date range
        start_date = '2018-01-01'
        end_date = '2024-12-31'
        df_filtered = df_filtered[
            (df_filtered['Date'] >= start_date) & 
            (df_filtered['Date'] <= end_date)
        ]
        
        # Ensure monthly data (resample if needed)
        df_filtered = df_filtered.set_index('Date')
        df_monthly = df_filtered.resample('M').last()  # Takes last value of each month
        
        # Reset index to have Date as a column again
        df_monthly = df_monthly.reset_index()
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        processed_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        output_file = os.path.join(processed_dir, f"required_indicators_{timestamp}.csv")
        
        # Save the filtered data
        df_monthly.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully processed and saved to: {output_file}")
        print(f"Date range: {df_monthly['Date'].min().strftime('%Y-%m-%d')} to {df_monthly['Date'].max().strftime('%Y-%m-%d')}")
        print("\nColumns included:")
        for col in df_monthly.columns:
            if col != 'Date':
                print(f"- {col}")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        return None

if __name__ == "__main__":
    print("E-commerce Required Indicators Filter")
    print("===================================\n")
    
    output_file = filter_and_save_data()
    
    if output_file:
        print("\n✅ Processing completed successfully!")
        print(f"You can find the filtered data in: {output_file}")
    else:
        print("\n❌ Processing failed. Please check the error message above.")
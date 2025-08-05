"""
E-commerce & Retail Data Sources Catalog
---------------------------------------
This script provides a comprehensive overview of available data sources for e-commerce and retail analysis.
It includes both free/public APIs and commercial data sources.
"""

import os
from datetime import datetime
import pandas as pd
import requests
import json
from typing import Dict, List, Optional

class DataSource:
    """Base class for data sources."""
    def __init__(self, name: str, description: str, api_docs: str, 
                 data_availability: str, update_frequency: str):
        self.name = name
        self.description = description
        self.api_docs = api_docs
        self.data_availability = data_availability
        self.update_frequency = update_frequency
        self.series = []
    
    def add_series(self, series_id: str, name: str, description: str, 
                  frequency: str, start_year: int, notes: str = ""):
        """Add a data series to the source."""
        self.series.append({
            'series_id': series_id,
            'name': name,
            'description': description,
            'frequency': frequency,
            'start_year': start_year,
            'notes': notes
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the data source to a pandas DataFrame."""
        df = pd.DataFrame(self.series)
        df['source'] = self.name
        return df

# =============================================================================
# 1. FRED (Federal Reserve Economic Data)
# =============================================================================
print("Creating FRED data source catalog...")

fred = DataSource(
    name="FRED (Federal Reserve Economic Data)",
    description="Comprehensive economic data from the Federal Reserve Bank of St. Louis",
    api_docs="https://fred.stlouisfed.org/docs/api/fred/",
    data_availability="National and some state-level economic indicators",
    update_frequency="Daily"
)

# E-commerce and Retail
fred.add_series(
    "ECOMSA", 
    "E-commerce Retail Sales", 
    "Total e-commerce retail sales in millions of dollars", 
    "Monthly", 2019
)

fred.add_series(
    "ECOMPCTSA", 
    "E-commerce as % of Total Retail", 
    "E-commerce as a percentage of total retail sales", 
    "Monthly", 2019
)

# Retail Categories
fred.add_series(
    "MRTSSM44X72USN", 
    "Retail Trade: Food and Beverage Stores", 
    "Retail sales for food and beverage stores", 
    "Monthly", 1992
)

fred.add_series(
    "MRTSSM4541USN", 
    "Retail Trade: Electronic Shopping and Mail-Order Houses", 
    "Retail sales for electronic shopping and mail-order houses", 
    "Monthly", 1992
)

# =============================================================================
# 2. US Census Bureau
# =============================================================================
print("Creating US Census data source catalog...")

census = DataSource(
    name="US Census Bureau",
    description="Official demographic and economic data from the US government",
    api_docs="https://www.census.gov/data/developers/data-sets.html",
    data_availability="National, state, county, and metro area levels",
    update_frequency="Annual/Quarterly"
)

# E-commerce specific
census.add_series(
    "ECNS48", 
    "Quarterly Retail E-commerce Sales", 
    "Quarterly e-commerce retail sales estimates", 
    "Quarterly", 1999
)

# Demographics
census.add_series(
    "DP05", 
    "Demographic and Housing Estimates", 
    "Demographic and housing characteristics", 
    "Annual", 2009
)

# =============================================================================
# 3. Commercial Data Sources (Not directly accessible via public API)
# =============================================================================
commercial = DataSource(
    name="Commercial Data Providers",
    description="Premium data sources for detailed e-commerce metrics",
    api_docs="Varies by provider",
    data_availability="Varies",
    update_frequency="Varies"
)

# Comscore
commercial.add_series(
    "COMSCORE_ECS", 
    "Comscore E-commerce Measurement", 
    "Detailed e-commerce metrics including traffic, transactions, and conversion", 
    "Monthly", 2000,
    notes="Requires enterprise subscription"
)

# eMarketer
commercial.add_series(
    "EMARKETER_EST", 
    "eMarketer Estimates", 
    "E-commerce market size, users, and penetration rates", 
    "Monthly", 2000,
    notes="Subscription required"
)

# =============================================================================
# 4. Data Gaps and Limitations
# =========================================================================
print("Documenting data gaps and limitations...")

GAPS = {
    "State-level E-commerce": "Not available from public sources. FRED and Census only provide national-level e-commerce data.",
    "Micro-industry Breakdowns": "Detailed subcategory data (e.g., specific product categories) is limited in public sources.",
    "Real-time Data": "Most public sources have a lag of 1-3 months.",
    "Qualitative Metrics": "Metrics like customer satisfaction, cart abandonment rates require commercial sources."
}

# =============================================================================
# Export to CSV for Power BI
# =============================================================================
print("Exporting data catalog to CSV...")

# Combine all data sources
all_sources = [fred, census, commercial]
df_list = [source.to_dataframe() for source in all_sources]
full_catalog = pd.concat(df_list, ignore_index=True)

# Export to CSV with UTF-8 encoding
output_file = "ecommerce_data_catalog.csv"
full_catalog.to_csv(output_file, index=False, encoding='utf-8-sig')

# Export gaps to markdown
with open("data_gaps.md", "w", encoding='utf-8') as f:
    f.write("# E-commerce Data Gaps\n\n")
    f.write("The following data is not available through public APIs and may require commercial sources:\n\n")
    for gap, desc in GAPS.items():
        f.write(f"- **{gap}**: {desc}\n")

# Print completion message (without special characters to avoid encoding issues)
print("\n[SUCCESS] Data catalog exported to", output_file)
print("[SUCCESS] Data gaps documented in data_gaps.md")
print("\nNext steps:")
print("1. Review the data catalog in ecommerce_data_catalog.csv")
print("2. Check data_gaps.md for unavailable metrics")
print("3. Use the catalog to plan your Power BI data integration")

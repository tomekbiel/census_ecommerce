# Advanced E-commerce and Retail Data

This dataset contains comprehensive e-commerce and retail data from multiple sources, including:
- National-level e-commerce and retail sales
- State-level retail metrics
- Demographic and economic indicators

## Data Files

### 1. National E-commerce Data (`national_ecommerce_data_*.csv`)
- Time series of national e-commerce and retail metrics
- Includes economic indicators like unemployment and consumer sentiment
- Date range: Varies by series

### 2. State E-commerce Data (`state_ecommerce_data_*.csv`)
- State-level retail and e-commerce metrics
- Can be joined with national data on date
- Includes state codes for mapping in Power BI

### 3. Census Demographic Data (`census_demographic_data_*.csv`)
- Demographic and economic data by state
- Can be joined with state data on State_Code

## Power BI Integration

1. **Import all CSV files** into Power BI
2. **Create relationships** between tables:
   - Link `national_ecommerce_data` and `state_ecommerce_data` on `date`
   - Link `state_ecommerce_data` and `census_demographic_data` on `State_Code`

3. **Recommended visualizations**:
   - Time series of national e-commerce growth
   - Choropleth map of state-level metrics
   - Scatter plots of e-commerce vs. demographic factors
   - Small multiples by state or region

## Data Dictionary

### National Data Columns:

## Last Updated
Data last fetched on: 2025-08-01 08:52:36

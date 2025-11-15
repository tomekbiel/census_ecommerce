# E-commerce Data Generator

This directory contains a comprehensive toolkit for generating and processing synthetic e-commerce data, designed to create realistic datasets for testing and analysis.

## Complete Workflow

### 1. Input Data Preparation
```
shopify_reports_2018-2024.csv (quarterly data)
       ↓
[convert_shopify_to_monthly.py]
       ↓
shopify_monthly_reports_2018-2024.csv
       ↓
[add_customers_estimation.py]
       ↓
shopify_with_customers.csv (final input for data generation)
```

### 2. Main Data Generation
```
[gen_ecom_faker5.py or gen_ecom_faker0_5.py]
       ↓
- customers.csv     # Initial customer data
- products.csv      # Product catalog
- orders.csv        # Initial order data
- order_items.csv   # Individual order items
```

### 3. Data Scaling & Enhancement
```
orders.csv
       ↓
[apply_revenue_scaling.py]
       ↓
orders2.csv (scaled to target revenue)
       ↓
[update_customers_loyalty.py]
       ↓
customers2.csv (with updated loyalty metrics)
```

## File Structure

### Core Data Generation
- `gen_ecom_faker0_5.py` - Latest version with enhanced documentation and features
- `gen_ecom_faker0_5.ipynb` - Interactive Jupyter notebook version
- `gen_ecom_faker3.py` - Previous version (maintained for reference)
- `gen_ecom_faker5.py` - Production version used for current data generation

### Data Processing Scripts
- `all_categories.py` - Defines product categories and their attributes
- `apply_revenue_scaling.py` - Scales order values to match target revenue
- `update_customers_loyalty.py` - Updates customer loyalty metrics based on order history
- `convert_shopify_to_monthly.py` - Converts quarterly Shopify data to monthly format
- `add_customers_estimation.py` - Estimates customer counts from Shopify sales data

## Output Files

### Main Outputs (in `data/synthetic/`)
- `customers.csv` - Initial customer data
- `customers2.csv` - Updated with loyalty metrics
- `products.csv` - Product catalog
- `orders.csv` - Initial order data
- `orders2.csv` - Scaled order data
- `order_items.csv` - Initial order items
- `order_items2.csv` - Updated order items

### Input Files (in `data/synthetic/`)
- `shopify_reports_2018-2024.csv` - Source Shopify data (quarterly)
- `shopify_monthly_reports_2018-2024.csv` - Converted monthly data
- `shopify_with_customers.csv` - Processed input for data generation

## Usage Example

### 1. Prepare Input Data
```python
# Convert Shopify quarterly to monthly data
python convert_shopify_to_monthly.py

# Estimate customer counts
python add_customers_estimation.py
```

### 2. Generate Base Data
```python
from gen_ecom_faker5 import EcommerceDataGenerator

# Initialize with target metrics
generator = EcommerceDataGenerator(
    target_customers=15000,
    target_revenue=2300000  # $2.3M target
)

# Generate and save all data
generator.generate_all_data()
```

### 3. Apply Scaling and Updates
```python
# Apply revenue scaling
python apply_revenue_scaling.py

# Update customer loyalty metrics
python update_customers_loyalty.py
```

## Dependencies
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- faker >= 12.0.0
- tqdm >= 4.62.0
- matplotlib >= 3.4.0 (for visualizations in the notebook)

## Troubleshooting

### Common Issues
1. **Missing Input Files**
   - Ensure all required input files are in `data/synthetic/`
   - Run the data preparation steps in order

2. **Memory Issues**
   - For large datasets, increase Python's memory allocation
   - Consider processing data in smaller batches

3. **Data Consistency**
   - Always use the complete workflow to maintain referential integrity
   - When updating data, regenerate all dependent files

## Notes
- The `gen_faker_backup` directory is excluded from version control
- Analysis outputs are stored in `../green_space_analysis/outputs_HTML/`
- For optimal performance, close other memory-intensive applications during data generation

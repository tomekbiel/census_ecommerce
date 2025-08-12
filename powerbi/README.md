# Power BI Project

This directory contains all Power BI related files for the E-commerce Census Analysis project.

## Directory Structure

- `reports/` - Store your `.pbix` Power BI report files
- `data/` - Place your CSV data files that will be used by Power BI reports
- `transformations/` - Any data transformation scripts or queries

## Getting Started

1. Place your Power BI report files (`.pbix`) in the `reports/` directory
2. Your Power BI reports should reference CSV files from the main `data/` directory:
   - Main data files: `../data/processed/`
   - Example: `../data/processed/ecommerce_analysis_latest.csv`
   - Synthetic data: `../data/synthetic/` (if applicable)

## Best Practices

1. **Version Control**: 
   - Only commit small Power BI files or consider using `.pbit` (Power BI Template) files
   - Large `.pbix` files should be stored outside version control (see `.gitignore`)
   - Data files are stored in the main `data/` directory, not in this folder
   
2. **Data Management**:
   - Raw and processed data files are stored in the main `data/` directory:
     - Processed data: `../data/processed/`
     - Synthetic data: `../data/synthetic/`
   - Document any data transformations in the `transformations/` directory
   - Use relative paths when referencing data files in Power BI

3. **Documentation**:
   - Document your data sources and any specific setup requirements
   - Keep your Power BI queries and data transformations well-documented

## Notes

- The `.gitignore` file is configured to exclude large binary files and sensitive information
- The main data directory is shared with other parts of the project
- When creating new Power BI reports, always use relative paths to the `data/` directory
- Example data source path in Power BI: `..\data\processed\ecommerce_analysis_latest.csv`

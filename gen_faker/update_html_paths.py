import os
import re
from pathlib import Path

def update_file_paths(file_path, replacements):
    """Update file paths in the specified HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Apply all replacements
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        print(f"✅ Updated {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error updating {file_path}: {str(e)}")
        return False

def main():
    # Define the directory containing the HTML files
    base_dir = Path(r"C:\Users\User\PycharmProjects\census_ecommerce\gen_faker")
    
    # Define the files to update
    html_files = [
        base_dir / 'customer_analysis.html',
        base_dir / 'marketing_performance.html'
    ]
    
    # Define the replacements (old path -> new path)
    replacements = {
        'data/synthetic/orders.csv': 'data/synthetic/orders2.csv',
        'data/synthetic/order_items.csv': 'data/synthetic/order_items2.csv',
        'data/synthetic/customers.csv': 'data/synthetic/customers2.csv',
        'data/synthetic/products.csv': 'data/synthetic/products2.csv'
    }
    
    # Update each file
    for file_path in html_files:
        if file_path.exists():
            print(f"\n🔧 Updating {file_path.name}...")
            update_file_paths(file_path, replacements)
        else:
            print(f"\n⚠️  File not found: {file_path}")
    
    print("\n🎉 Update complete!")

if __name__ == "__main__":
    main()

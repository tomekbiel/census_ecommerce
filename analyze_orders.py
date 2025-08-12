import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up the style for the plots
try:
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid")
except:
    # Fallback to default style if seaborn is not available
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [12, 8]

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'synthetic', 'orders.csv')
df = pd.read_csv(data_path)

# Display basic info about the data
print("Data Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Function to plot distributions
def plot_distributions(df, columns, figsize=(15, 10)):
    num_plots = len(columns)
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(columns):
            ax = axes[i]
            # For numerical columns
            if df[col].dtype in ['int64', 'float64']:
                sns.histplot(df[col], kde=True, ax=ax, bins=30)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel('')
                ax.set_ylabel('Frequency')
            
            # For datetime columns
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    date_series = pd.to_datetime(df[col])
                    sns.histplot(date_series, ax=ax, bins=30)
                    ax.set_title(f'Distribution of {col}')
                    ax.tick_params(axis='x', rotation=45)
                except:
                    pass
    
    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

# Get numerical and date columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

# Plot numerical distributions
if numeric_cols:
    print("\nPlotting numerical distributions...")
    fig_num = plot_distributions(df, numeric_cols)
    plt.savefig(os.path.join(current_dir, 'order_numerical_distributions.png'))
    print("Saved numerical distributions as 'order_numerical_distributions.png'")

# Plot date distributions
if date_cols:
    print("\nPlotting date distributions...")
    fig_date = plot_distributions(df, date_cols, figsize=(15, 5*((len(date_cols)+1)//2)))
    plt.savefig(os.path.join(current_dir, 'order_date_distributions.png'))
    print("Saved date distributions as 'order_date_distributions.png'")

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Show correlation matrix for numerical columns
if len(numeric_cols) > 1:
    print("\nCorrelation Matrix:")
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'order_correlation_matrix.png'))
    print("Saved correlation matrix as 'order_correlation_matrix.png'")

print("\nAnalysis complete. Check the generated plots in the project directory.")

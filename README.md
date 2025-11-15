# E-commerce Data Analysis Project

Comprehensive toolkit for analyzing and generating synthetic e-commerce data, with a focus on customer behavior and sales trends.

## Project Structure

- `ecommerce/` - Core e-commerce data analysis modules
  - `analysis/` - Data analysis scripts and visualizations
  - `processing/` - Data processing and transformation utilities
  - `data_sources_catalog.py` - Catalog of data sources and schemas
  - `ecommerce_data_catalog.csv` - Metadata for e-commerce datasets

- `gen_faker/` - Synthetic data generation
  - See [gen_faker/README.md](gen_faker/README.md) for detailed documentation

- `green_space_analysis/` - Analysis of green space impact on e-commerce

- `data/` - Data storage
  - `synthetic/` - Generated synthetic datasets
  - `raw/` - Raw data sources
  - `processed/` - Processed and cleaned datasets

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tomekbiel/census_ecommerce.git
   cd census_ecommerce
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Generation
Generate synthetic e-commerce data:
```bash
cd gen_faker
python gen_ecom_faker5.py
```

### 2. Data Analysis
Run analysis scripts from the `ecommerce/analysis/` directory.

### 3. Green Space Analysis
Explore the impact of green spaces on e-commerce metrics in the `green_space_analysis/` directory.

## Documentation

- [Synthetic Data Generation](gen_faker/README.md) - Complete guide to generating and processing synthetic e-commerce data
- [Data Catalog](ecommerce/ecommerce_data_catalog.csv) - Documentation of available datasets and their schemas
- [Data Processing](ecommerce/processing/) - Documentation of data processing workflows
- `data/processed/ecommerce_analysis_YYYYMMDD_HHMMSS.csv` - Timestamped data export
- `data/processed/ecommerce_analysis_latest.csv` - Always points to the most recent export

## Project Structure

```
census_ecommerce/
├── data/                    # Data storage
│   └── processed/           # Processed data files
├── src/                     # Source code (if applicable)
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── ecommerce_data_analysis.py  # Main analysis script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Data Sources

- **FRED (Federal Reserve Economic Data)**: 
  - E-commerce Retail Sales (ECOMSA)
  - Total Retail Sales (RSXFS)
  - E-commerce as Percent of Total Retail (ECOMPCT)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Tomek Biel - [tomekbiel@example.com]

Project Link: [https://github.com/tomekbiel/census_ecommerce](https://github.com/tomekbiel/census_ecommerce)

## Szybki start

### Uruchomienie interaktywnego eksploratora

```bash
python -m src.data_explorer
```

### Przykładowe użycie z wiersza poleceń

Pobierz dane kwartalne o sprzedaży e-commerce:

```bash
python -m src.main --api-key twój_klucz_api --time-period "from+2020" --naics 44X72 --clean --plot-timeseries --format csv
```

## Integracja z Power BI

### Opcja 1: Import plików CSV/Excel
1. Wyeksportuj dane za pomocą eksploratora do formatu CSV lub Excel
2. W Power BI wybierz "Pobierz dane" > "Plik" > "Tekst/CSV" lub "Excel"
3. Wybierz wyeksportowany plik i postępuj zgodnie z kreatorem importu

### Opcja 2: Łączenie przez Python
1. W Power BI wybierz "Pobierz dane" > "Inne" > "Python script"
2. Wprowadź następujący kod, dostosowując parametry do swoich potrzeb:

```python
import pandas as pd
from src.api_client import CensusEcommerceAPI

# Inicjalizacja klienta API
api = CensusEcommerceAPI()  # Używa klucza z .env

# Pobieranie danych
df = api.get_quarterly_retail_ecommerce(
    time="from+2020",
    NAICS="44X72"
)

# Konwersja na DataFrame Power BI
dataset = df
```

## Dostępne zestawy danych

### Kwartalna sprzedaż e-commerce (QSS)
- Częstotliwość: kwartalna
- Dostępne lata: 1999 - obecnie
- Główne kategorie NAICS:
  - 44X72: Całkowita sprzedaż detaliczna
  - 4541: Elektroniczne zakupy i domy wysyłkowe
  - 454111: Elektroniczne zakupy

### Miesięczny przegląd handlu detalicznego (MARTS)
- Częstotliwość: miesięczna
- Dostępne lata: 1992 - obecnie
- Zawiera dane o sprzedaży, zapasach i wskaźnikach rotacji zapasów

## Licencja

MIT

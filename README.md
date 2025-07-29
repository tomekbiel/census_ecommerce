# Census E-commerce Data Explorer

Narzędzie do eksploracji i eksportu danych o handlu elektronicznym z amerykańskiego Biura Spisu Ludności (U.S. Census Bureau).

## Features

- **Data Collection**: Fetch up-to-date e-commerce and retail sales data from FRED
- **Data Processing**: Clean and transform raw data into analysis-ready format
- **Export Capabilities**: Save processed data to CSV for use in Power BI, Excel, or other tools
- **Customizable**: Easily modify the script to include additional economic indicators

## Getting Started

### Prerequisites

- Python 3.8 or higher
- FRED API key (free account available at [fred.stlouisfed.org](https://fred.stlouisfed.org))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tomekbiel/census_ecommerce.git
   cd census_ecommerce
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your FRED API key:
   ```
   FRED_API_KEY=your_fred_api_key_here
   ```

## Usage

Run the main analysis script:
```bash
python ecommerce_data_analysis.py
```

This will:
1. Fetch the latest e-commerce and retail sales data from FRED
2. Process and clean the data
3. Save the results to `data/processed/` with a timestamped filename
4. Create/update `ecommerce_analysis_latest.csv` for easy access

### Output Files
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

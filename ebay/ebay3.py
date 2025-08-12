headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

params = {
    "dimension": "MONTH",  # lub "QUARTER"
    "metric": "TRANSACTION_COUNT,GROSS_SALES",
    "filter": "marketplace_ids:EBAY_US"  # Tylko rynek USA
}

response = requests.get("https://api.ebay.com/sell/analytics/v1/sales_report", headers=headers, params=params)
data = response.json()

print("Dane sprzedaży:", data)
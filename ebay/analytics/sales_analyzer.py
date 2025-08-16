from datetime import datetime, timedelta
from ..api.client import EBayClient


class EBaySalesAnalyzer:
    def __init__(self, env='production'):
        self.client = EBayClient(env=env)

    def get_completed_sales(self, keywords, days_back=30, limit=100):
        """Pobiera zakończone aukcje dla podanych słów kluczowych"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        params = {
            'q': keywords,
            'limit': min(limit, 200),  # Maksymalnie 200 wyników
            'item_filter': [
                {'name': 'SoldItemsOnly', 'value': True},
                {'name': 'EndTimeFrom', 'value': start_date.isoformat() + 'Z'},
                {'name': 'EndTimeTo', 'value': end_date.isoformat() + 'Z'}
            ]
        }

        return self.client._make_request(
            'GET',
            '/buy/marketplace_insights/v1_beta/item_sales/search',
            params=params
        )


# Przykład użycia
if __name__ == "__main__":
    analyzer = EBaySalesAnalyzer(env='production')
    sales = analyzer.get_completed_sales("laptop", days_back=7, limit=10)
    print("Completed sales:", sales)
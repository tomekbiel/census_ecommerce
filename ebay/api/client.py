import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class EBayClient:
    def __init__(self, env=None):
        self.env = env or os.getenv('EBAY_ENV', 'production')
        self.base_url = (
            "https://api.ebay.com"
            if self.env == 'production'
            else "https://api.sandbox.ebay.com"
        )
        self.token = os.getenv(
            "EBAY_PRODUCTION_ACCESS_TOKEN"
            if self.env == 'production'
            else "EBAY_SANDBOX_ACCESS_TOKEN"
        )

    def _make_request(self, method, endpoint, **kwargs):
        """Wykonuje zapytanie do API eBay"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        headers.update(kwargs.pop('headers', {}))

        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=30,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text}"
            raise Exception(f"API request failed: {error_msg}")

    # Przykładowe metody
    def search_items(self, query, limit=10):
        """Wyszukuje przedmioty"""
        return self._make_request(
            'GET',
            '/buy/browse/v1/item_summary/search',
            params={'q': query, 'limit': limit}
        )

    def get_item(self, item_id):
        """Pobiera szczegóły przedmiotu"""
        return self._make_request(
            'GET',
            f'/buy/browse/v1/item/{item_id}'
        )


# Przykład użycia
if __name__ == "__main__":
    client = EBayClient(env='production')
    results = client.search_items("laptop", limit=3)
    print("Search results:", results)
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the ebay/.env file
env_path = Path(__file__).resolve().parent.parent / '.env'
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Debug: Print if token is loaded
print(f"EBAY_PRODUCTION_ACCESS_TOKEN exists: {'EBAY_PRODUCTION_ACCESS_TOKEN' in os.environ}")
print(f"EBAY_SANDBOX_ACCESS_TOKEN exists: {'EBAY_SANDBOX_ACCESS_TOKEN' in os.environ}")


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
        
        # Debug: Print token and headers before request
        print(f"Using token: {'*' * 10 + self.token[-4:] if self.token else 'NOT FOUND'}")
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        # Make sure we're not overriding the Authorization header
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
            del kwargs['headers']
            
        print(f"Request headers: {headers}")
        print(f"Making {method} request to: {url}")
        print(f"Params: {kwargs.get('params', {})}")

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
        
    def get_headers(self):
        """Returns the headers with the access token"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def get_item(self, item_id):
        """Pobiera szczegóły przedmiotu"""
        return self._make_request(
            'GET',
            f'/buy/browse/v1/item/{item_id}'
        )


# Przykład użycia
if __name__ == "__main__":
    # Use sandbox by default for testing
    client = EBayClient(env='sandbox')
    print(f"Using {client.env} environment")
    print(f"Base URL: {client.base_url}")
    
    try:
        results = client.search_items("laptop", limit=3)
        print("\nSearch successful! Results:")
        print(f"Found {len(results.get('itemSummaries', []))} items")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease check:")
        print("1. Your .env file contains valid eBay API credentials")
        print("2. The token is not expired")
        print("3. You have the necessary permissions for the API")
        print("4. You're using the correct environment (sandbox/production)")
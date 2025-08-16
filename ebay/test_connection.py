import os
import requests
from dotenv import load_dotenv

load_dotenv()


class EBayConnectionTester:
    def __init__(self, env='production'):
        self.base_url = (
            "https://api.ebay.com"
            if env == 'production'
            else "https://api.sandbox.ebay.com"
        )
        self.token = os.getenv(
            "EBAY_PRODUCTION_ACCESS_TOKEN"
            if env == 'production'
            else "EBAY_SANDBOX_ACCESS_TOKEN"
        )

    def test_connection(self):
        """Test połączenia z API eBay"""
        url = f"{self.base_url}/buy/browse/v1/item_summary/search"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY-US"
        }
        params = {"q": "iphone", "limit": 1}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return {
                "success": True,
                "status": response.status_code,
                "data": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }


if __name__ == "__main__":
    # Test dla środowiska produkcyjnego
    print("Testing Production Environment...")
    prod_tester = EBayConnectionTester('production')
    result = prod_tester.test_connection()
    print(f"Production: {'✅' if result['success'] else '❌'} {result.get('error', '')}")

    # Test dla środowiska testowego
    print("\nTesting Sandbox Environment...")
    sandbox_tester = EBayConnectionTester('sandbox')
    result = sandbox_tester.test_connection()
    print(f"Sandbox: {'✅' if result['success'] else '❌'} {result.get('error', '')}")
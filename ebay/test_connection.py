import os
import sys
import requests
from dotenv import load_dotenv
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load environment variables from the ebay/.env file
env_path = Path(__file__).resolve().parent / '.env'
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Debug: Print environment variables with values (masked)
print("\n=== Environment Variables ===")
for var in ['EBAY_SANDBOX_ACCESS_TOKEN', 'EBAY_PRODUCTION_ACCESS_TOKEN']:
    value = os.getenv(var)
    print(f"{var}: {'*' * 10 + value[-4:] if value else 'NOT SET'}")
print("=" * 30)


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
        print(f"\nTesting connection to: {self.base_url}")
        print(f"Token: {'*' * 10 + self.token[-4:] if self.token else 'NOT FOUND'}")
        
        url = f"{self.base_url}/buy/browse/v1/item_summary/search"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY-US"
        }
        params = {"q": "iphone", "limit": 1}
        
        print(f"\nRequest Details:")
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Params: {params}")

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
    # Test for sandbox environment first (safer for testing)
    print("\n=== Testing Sandbox Environment ===")
    sandbox_tester = EBayConnectionTester('sandbox')
    result = sandbox_tester.test_connection()
    print(f"\nSandbox Test Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    if not result['success']:
        print(f"Error: {result.get('error')}")
        print(f"Status Code: {result.get('status')}")
    
    # Only test production if sandbox works
    if result['success']:
        print("\n=== Testing Production Environment ===")
        prod_tester = EBayConnectionTester('production')
        result = prod_tester.test_connection()
        print(f"\nProduction Test Result: {'SUCCESS' if result['success'] else 'FAILED'}")
        if not result['success']:
            print(f"Error: {result.get('error')}")
            print(f"Status Code: {result.get('status')}")
    else:
        print("\nSkipping production test due to sandbox failure")
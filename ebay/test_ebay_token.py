import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the production token
access_token = os.getenv('EBAY_PRODUCTION_ACCESS_TOKEN')

if not access_token:
    print("Error: No access token found in .env file")
    exit(1)

# Test a known-working endpoint (Sell Account API)
url = "https://api.ebay.com/sell/account/v1/privilege"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "X-EBAY-C-MARKETPLACE-ID": "EBAY-US"
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise HTTP errors
    print("Token is valid! API Response:")
    print(response.json())
except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
    print("Response:", err.response.text if err.response else "No response")
except Exception as e:
    print(f"An error occurred: {e}")
from ebay.api.client import EBayClient

# Initialize client
client = EBayClient(env='sandbox')  # or 'production'

# Search for items
results = client.search_items("laptop", limit=5)
print(f"Found {len(results.get('itemSummaries', []))} items")
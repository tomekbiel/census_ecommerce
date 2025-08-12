import requests
import base64

client_id = "Twoj-App-ID"
client_secret = "Twoj-App-Secret"
redirect_uri = "https://localhost"  # Musi być zgodny z ustawieniami aplikacji

# Krok 1: Generuj URL autoryzacyjny
auth_url = (
    f"https://auth.ebay.com/oauth2/authorize?"
    f"client_id={client_id}&"
    f"redirect_uri={redirect_uri}&"
    f"response_type=code&"
    f"scope=https://api.ebay.com/oauth/api_scope/sell.analytics.readonly"
)

print("Zaloguj się przez ten URL:", auth_url)
# Po zalogowaniu przekieruje na localhost z kodem w URL
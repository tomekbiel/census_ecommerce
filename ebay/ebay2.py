auth_code = "kod_z_url"  # Znajdziesz w URL po przekierowaniu (parametr ?code=...)

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
}

data = {
    "grant_type": "authorization_code",
    "code": auth_code,
    "redirect_uri": redirect_uri
}

response = requests.post("https://api.ebay.com/identity/v1/oauth2/token", headers=headers, data=data)
token = response.json()["access_token"]
print("Twój token:", token)
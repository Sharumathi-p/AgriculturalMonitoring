import requests

# Your API key and city
API_KEY = "abf4caa4f4ab81bf8a668e84ac332d5a"
CITY = "Chennai"

# API endpoint
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# Make the request
response = requests.get(URL)
data = response.json()
# Check if the request was successful
if data["cod"] == 200:
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Feels like: {data['main']['feels_like']}°C")
    print(f"Weather: {data['weather'][0]['description']}")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Wind Speed: {data['wind']['speed']} m/s")
else:
    print("Error fetching weather data:", data.get("message", "Unknown error"))

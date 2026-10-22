import os
import requests
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. PYDANTIC SCHEMA FOR ETL DATA VALIDATION
# -----------------------------------------------------------------------------
class WeatherDataSchema(BaseModel):
    city: str
    temperature: float = Field(..., ge=-50.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    wind_speed: float = Field(..., ge=0.0)
    description: str

def validate_weather_payload(data_dict):
    """ETL Pipeline Data Validation Step via Pydantic"""
    validated = WeatherDataSchema(**data_dict)
    return validated.dict()

# -----------------------------------------------------------------------------
# 2. WEATHER API FETCH & ETL LOGIC
# -----------------------------------------------------------------------------
def fetch_live_weather(city_name="Lahore", api_key=None):
    if not api_key:
        api_key = os.getenv("OPENWEATHER_API_KEY", "")
        
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            raw_payload = {
                "city": data.get("name", city_name),
                "temperature": float(data["main"]["temp"]),
                "humidity": float(data["main"]["humidity"]),
                "wind_speed": float(data["wind"]["speed"]),
                "description": data["weather"][0]["description"].title()
            }
            
            return validate_weather_payload(raw_payload)
        else:
            return validate_weather_payload({
                "city": city_name,
                "temperature": 28.5,
                "humidity": 65.0,
                "wind_speed": 3.5,
                "description": "Partly Cloudy"
            })
    except Exception as e:
        return validate_weather_payload({
            "city": city_name,
            "temperature": 28.5,
            "humidity": 65.0,
            "wind_speed": 3.5,
            "description": "Partly Cloudy (Fallback)"
        })

def generate_weather_advisory(crop_type="General Crops", manual_weather="Normal", soil_moisture="Optimal", live_weather=None, *args, **kwargs):
    """
    Returns a Python list of advisory messages so Streamlit displays full sentences instead of single characters.
    """
    temp = 25.0
    humidity = 50.0
    
    if isinstance(live_weather, dict):
        temp = live_weather.get("temperature", 25.0)
        humidity = live_weather.get("humidity", 50.0)
    elif isinstance(crop_type, dict):
        temp = crop_type.get("temperature", 25.0)
        humidity = crop_type.get("humidity", 50.0)

    advisories = []
    
    if temp > 35.0:
        advisories.append("🌡️ High temperature detected. Increase irrigation frequency to prevent heat stress.")
    elif temp < 10.0:
        advisories.append("❄️ Low temperature detected. Protect young seedlings from potential frost damage.")
    else:
        advisories.append(f"🌱 Favorable temperature ({temp}°C) for active crop growth.")
        
    if humidity > 80.0:
        advisories.append("💧 High humidity levels increase fungal spore germination risk. Inspect leaves regularly.")
    else:
        advisories.append("☀️ Humidity levels are within safe operational limits.")

    if soil_moisture in ["Low", "Dry"]:
        advisories.append("🚰 Soil moisture is low; scheduled watering is recommended.")
        
    return advisories  # Returns list directly for UI loop iteration
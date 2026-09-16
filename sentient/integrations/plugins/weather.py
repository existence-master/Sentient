"""Weather. Keyless default: Open-Meteo. Alternative: AccuWeather (API key)."""

from __future__ import annotations

from sentient.integrations.base import IntegrationError, IntegrationPlugin, itool, manager_from
from sentient.integrations.common import http_client
from sentient.tools.base import ToolContext

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ACCU_BASE = "https://dataservice.accuweather.com"

WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Slight rain showers",
    81: "Rain showers", 82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _location(ctx: ToolContext, location: str | None) -> str:
    loc = (location or "").strip() or manager_from(ctx).app.config.assistant.location.strip()
    if not loc:
        raise IntegrationError("Which city? No location was given and no home location is set in Settings.")
    return loc


async def geocode(location: str) -> dict:
    parts = [p.strip() for p in location.split(",") if p.strip()]
    name = parts[0] if parts else location
    rest = [p.lower() for p in parts[1:]]
    async with http_client() as http:
        r = await http.get(GEOCODE_URL, params={"name": name, "count": 10, "language": "en", "format": "json"})
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        raise IntegrationError(f"Couldn't find a place called '{location}'.")

    def score(x: dict) -> int:
        fields = " ".join(str(x.get(k, "")) for k in ("country", "country_code", "admin1", "admin2")).lower()
        return sum(1 for p in rest if p in fields)

    best = max(results, key=lambda x: (score(x), x.get("population") or 0)) if rest else results[0]
    label = ", ".join(str(best[k]) for k in ("name", "admin1", "country") if best.get(k))
    return {"name": label, "latitude": best["latitude"], "longitude": best["longitude"],
            "timezone": best.get("timezone")}


async def open_meteo(location: str, days: int) -> dict:
    place = await geocode(location)
    params = {
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
        "forecast_days": max(1, min(days, 16)),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,"
                   "wind_speed_10m,is_day",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "precipitation_sum,sunrise,sunset",
    }
    async with http_client() as http:
        r = await http.get(FORECAST_URL, params=params)
    r.raise_for_status()
    data = r.json()
    cur = data.get("current") or {}
    current = {
        "time": cur.get("time"),
        "condition": WMO.get(cur.get("weather_code"), "Unknown"),
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_percent": cur.get("relative_humidity_2m"),
        "precipitation_mm": cur.get("precipitation"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "is_day": bool(cur.get("is_day")),
    }
    d = data.get("daily") or {}
    forecast = []
    for i, day in enumerate(d.get("time") or []):
        def at(key: str, i: int = i):
            vals = d.get(key) or []
            return vals[i] if i < len(vals) else None

        forecast.append({
            "date": day, "condition": WMO.get(at("weather_code"), "Unknown"),
            "min_c": at("temperature_2m_min"), "max_c": at("temperature_2m_max"),
            "rain_chance_percent": at("precipitation_probability_max"), "precipitation_mm": at("precipitation_sum"),
            "sunrise": at("sunrise"), "sunset": at("sunset"),
        })
    return {"location": place["name"], "timezone": data.get("timezone"), "provider": "open-meteo",
            "current": current, "forecast": forecast}


async def accuweather(key: str, location: str, days: int) -> dict:
    async with http_client() as http:
        r = await http.get(f"{ACCU_BASE}/locations/v1/cities/search", params={"apikey": key, "q": location})
        r.raise_for_status()
        found = r.json()
        if not found:
            raise IntegrationError(f"AccuWeather couldn't find '{location}'.")
        loc = found[0]
        lk = loc["Key"]
        cur_r = await http.get(f"{ACCU_BASE}/currentconditions/v1/{lk}", params={"apikey": key, "details": "true"})
        cur_r.raise_for_status()
        fc_r = await http.get(f"{ACCU_BASE}/forecasts/v1/daily/5day/{lk}", params={"apikey": key, "metric": "true"})
        fc_r.raise_for_status()
    c = (cur_r.json() or [{}])[0]
    current = {
        "time": c.get("LocalObservationDateTime"), "condition": c.get("WeatherText"),
        "temperature_c": ((c.get("Temperature") or {}).get("Metric") or {}).get("Value"),
        "feels_like_c": ((c.get("RealFeelTemperature") or {}).get("Metric") or {}).get("Value"),
        "humidity_percent": c.get("RelativeHumidity"),
        "wind_kmh": (((c.get("Wind") or {}).get("Speed") or {}).get("Metric") or {}).get("Value"),
        "uv_index": c.get("UVIndex"), "is_day": c.get("IsDayTime"),
    }
    forecast = [{
        "date": (f.get("Date") or "")[:10], "condition": (f.get("Day") or {}).get("IconPhrase"),
        "night": (f.get("Night") or {}).get("IconPhrase"),
        "min_c": ((f.get("Temperature") or {}).get("Minimum") or {}).get("Value"),
        "max_c": ((f.get("Temperature") or {}).get("Maximum") or {}).get("Value"),
    } for f in (fc_r.json().get("DailyForecasts") or [])[:days]]
    name = ", ".join(x for x in (loc.get("LocalizedName"), (loc.get("Country") or {}).get("LocalizedName")) if x)
    return {"location": name, "provider": "accuweather", "current": current, "forecast": forecast}


async def _weather(ctx: ToolContext, location: str | None, days: int) -> dict:
    mgr = manager_from(ctx)
    loc = _location(ctx, location)
    if mgr.app.config.integrations.weather_provider == "accuweather" and await mgr.is_connected("accuweather"):
        c = await mgr.get_credentials("accuweather") or {}
        if c.get("api_key"):
            return await accuweather(c["api_key"], loc, min(days, 5))
    return await open_meteo(loc, days)


@itool("weather", "weather_current")
async def weather_current(ctx: ToolContext, location: str | None = None) -> dict:
    """Get the weather right now (condition, temperature, feels-like, humidity, wind) plus today's forecast.
    `location` is a city like "Pune, India"; leave it empty for the user's home location."""
    data = await _weather(ctx, location, 1)
    return {"location": data["location"], "provider": data["provider"], "current": data["current"],
            "today": (data.get("forecast") or [None])[0]}


@itool("weather", "weather_forecast")
async def weather_forecast(ctx: ToolContext, location: str | None = None, days: int = 7) -> dict:
    """Get the daily weather forecast (up to 16 days: min/max temperature, condition, chance of rain).
    `location` is a city like "Pune, India"; leave it empty for the user's home location."""
    return await _weather(ctx, location, max(1, min(int(days or 7), 16)))


class WeatherPlugin(IntegrationPlugin):
    id = "weather"
    display_name = "Weather"
    description = (
        "Current conditions and a 7-day forecast for any place, powered by Open-Meteo with no account or key "
        "needed. Uses your home location from Settings when you don't name a city. You can switch to "
        "AccuWeather by connecting it."
    )
    category = "utilities"
    icon = "weather"
    auth_type = "builtin"
    selection_hint = "Use for the weather right now, temperature, rain or the forecast for any place."
    tools = [weather_current, weather_forecast]


PLUGIN = WeatherPlugin()

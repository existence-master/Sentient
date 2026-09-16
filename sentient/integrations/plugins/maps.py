"""Maps. Keyless default: OpenStreetMap Nominatim (places) + OSRM (directions). Google Maps when a key is set."""

from __future__ import annotations

from sentient.integrations.base import IntegrationError, IntegrationPlugin, itool, manager_from
from sentient.integrations.common import http_client
from sentient.tools.base import ToolContext

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM = {
    "driving": "https://router.project-osrm.org/route/v1/driving",
    "walking": "https://routing.openstreetmap.de/routed-foot/route/v1/driving",
    "bicycling": "https://routing.openstreetmap.de/routed-bike/route/v1/driving",
}
MODES = {"drive": "driving", "driving": "driving", "car": "driving", "walk": "walking", "walking": "walking",
         "foot": "walking", "bike": "bicycling", "bicycle": "bicycling", "bicycling": "bicycling",
         "cycling": "bicycling", "transit": "transit"}


async def _google_key(ctx: ToolContext) -> str | None:
    mgr = manager_from(ctx)
    if await mgr.is_connected("google_maps"):
        return (await mgr.get_credentials("google_maps") or {}).get("api_key")
    return None


async def nominatim(query: str, limit: int) -> list[dict]:
    async with http_client() as http:
        r = await http.get(NOMINATIM, params={"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 1})
    r.raise_for_status()
    return [{"name": x.get("name") or x.get("display_name", "").split(",")[0], "address": x.get("display_name"),
             "latitude": float(x["lat"]), "longitude": float(x["lon"]), "type": x.get("type"),
             "url": f"https://www.openstreetmap.org/{x.get('osm_type')}/{x.get('osm_id')}"} for x in r.json()]


def _fmt_distance(m: float) -> str:
    return f"{m / 1000:.1f} km" if m >= 1000 else f"{int(m)} m"


def _fmt_duration(s: float) -> str:
    mins = round(s / 60)
    return f"{mins // 60} h {mins % 60} min" if mins >= 60 else f"{mins} min"


@itool("maps", "maps_search_places")
async def maps_search_places(ctx: ToolContext, query: str, max_results: int = 5) -> dict:
    """Find places, addresses or points of interest (e.g. "coffee near Koregaon Park, Pune", "Eiffel Tower").
    Returns names, full addresses, coordinates and a map link."""
    n = max(1, min(int(max_results or 5), 10))
    key = await _google_key(ctx)
    if key:
        async with http_client(headers={"X-Goog-Api-Key": key,
                                        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,"
                                                            "places.location,places.rating,places.googleMapsUri"}) as http:
            r = await http.post("https://places.googleapis.com/v1/places:searchText",
                                json={"textQuery": query, "pageSize": n})
        r.raise_for_status()
        places = [{"name": (p.get("displayName") or {}).get("text"), "address": p.get("formattedAddress"),
                   "latitude": (p.get("location") or {}).get("latitude"),
                   "longitude": (p.get("location") or {}).get("longitude"), "rating": p.get("rating"),
                   "url": p.get("googleMapsUri")} for p in r.json().get("places") or []]
        return {"query": query, "provider": "google", "places": places}
    places = await nominatim(query, n)
    return {"query": query, "provider": "openstreetmap", "places": places,
            **({} if places else {"note": "No places found. Try adding the city or country."})}


@itool("maps", "maps_directions")
async def maps_directions(ctx: ToolContext, origin: str, destination: str, mode: str = "driving") -> dict:
    """Get a route between two places with distance, travel time and turn-by-turn steps.
    `mode`: driving, walking, bicycling (or transit when Google Maps is connected)."""
    travel = MODES.get((mode or "driving").lower().strip(), "driving")
    key = await _google_key(ctx)
    if key:
        async with http_client() as http:
            r = await http.get("https://maps.googleapis.com/maps/api/directions/json",
                               params={"origin": origin, "destination": destination, "mode": travel, "key": key})
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            raise IntegrationError(f"Google Maps couldn't find a route ({data.get('status')}).")
        leg = data["routes"][0]["legs"][0]
        import re

        steps = [re.sub(r"<[^>]+>", "", s.get("html_instructions", "")) + f" ({s['distance']['text']})"
                 for s in leg.get("steps", [])[:30]]
        return {"provider": "google", "mode": travel, "from": leg.get("start_address"), "to": leg.get("end_address"),
                "distance": leg["distance"]["text"], "duration": leg["duration"]["text"], "steps": steps}
    note = None
    if travel == "transit":
        travel, note = "driving", "Public transit routes need Google Maps; showing a driving route instead."
    a = await nominatim(origin, 1)
    b = await nominatim(destination, 1)
    if not a or not b:
        raise IntegrationError(f"Couldn't find {'the start' if not a else 'the destination'} on the map.")
    coords = f"{a[0]['longitude']},{a[0]['latitude']};{b[0]['longitude']},{b[0]['latitude']}"
    async with http_client() as http:
        r = await http.get(f"{OSRM[travel]}/{coords}", params={"overview": "false", "steps": "true"})
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise IntegrationError("No route found between those places.")
    route = data["routes"][0]
    steps = []
    for leg in route.get("legs", []):
        for s in leg.get("steps", []):
            man = s.get("maneuver") or {}
            action = " ".join(x for x in (man.get("type"), man.get("modifier")) if x)
            road = s.get("name") or ""
            if s.get("distance", 0) > 0 or man.get("type") == "arrive":
                steps.append(f"{action}{' onto ' + road if road else ''} ({_fmt_distance(s.get('distance', 0))})")
    out = {"provider": "openstreetmap", "mode": travel, "from": a[0]["address"], "to": b[0]["address"],
           "distance": _fmt_distance(route["distance"]), "duration": _fmt_duration(route["duration"]),
           "steps": steps[:30],
           "map_url": "https://www.openstreetmap.org/directions?route="
                      f"{a[0]['latitude']},{a[0]['longitude']};{b[0]['latitude']},{b[0]['longitude']}"}
    if note:
        out["note"] = note
    return out


class MapsPlugin(IntegrationPlugin):
    id = "maps"
    display_name = "Maps"
    description = (
        "Looks up places and addresses and plans routes for driving, walking or cycling using OpenStreetMap, "
        "with no account or key needed. Connect Google Maps for richer place details and public transit."
    )
    category = "utilities"
    icon = "map"
    auth_type = "builtin"
    selection_hint = "Use to find places or addresses, or get directions, distance and travel time."
    tools = [maps_search_places, maps_directions]


PLUGIN = MapsPlugin()

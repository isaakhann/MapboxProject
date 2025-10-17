import json

with open("solar.geojson") as f:
    osm = json.load(f)

fc = {
  "type": "FeatureCollection",
  "features": []
}

for el in osm["elements"]:
    if el.get("lon") and el.get("lat"):
        fc["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [el["lon"], el["lat"]]
            },
            "properties": el.get("tags", {})
        })

with open("solar_clean.geojson", "w") as out:
    json.dump(fc, out, indent=2)

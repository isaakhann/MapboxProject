import pandas as pd
import json

# 1) Update this path to wherever your Excel file really lives
excel_path = 'Tarla.xlsx'

# 2) Read the sheet (adjust sheet_name if needed)
df = pd.read_excel(excel_path, sheet_name='Data')

# 3) Keep only rows with valid coordinates
df = df[df['Latitude'].notna() & df['Longitude'].notna()]

# 4) Build GeoJSON features
features = []
for _, row in df.iterrows():
    lon, lat = row['Longitude'], row['Latitude']
    feat = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "Project Name":       row['Project Name'],
            "Phase Name":         row['Phase Name'],
            "Capacity (MW)":      row['Capacity (MW)'],
            "Installation Type":  row['Installation Type'],
            "Status":             row['Status'],
            "Start year":         row['Start year'],
            "State/Province":     row['State/Province'],
            "Latitude":           lat,
            "Longitude":          lon
        }
    }
    features.append(feat)

# 5) Assemble FeatureCollection and write out
geojson = {
    "type": "FeatureCollection",
    "features": features
}
with open('tarla_windmills_with_coords.json', 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print("✅ Written tarla_windmills_with_coords.json")

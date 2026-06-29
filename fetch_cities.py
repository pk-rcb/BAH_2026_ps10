import urllib.request
import urllib.parse
import json
import csv

query = """
SELECT ?cityLabel ?coord WHERE {
  ?city wdt:P31/wdt:P279* wd:Q515;
        wdt:P17 wd:Q668;
        wdt:P625 ?coord.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 500
"""
url = "https://query.wikidata.org/sparql?query=" + urllib.parse.quote(query) + "&format=json"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    cities = []
    for item in data['results']['bindings']:
        city = item['cityLabel']['value']
        coord = item['coord']['value'] # Format: Point(lon lat)
        if coord.startswith("Point("):
            coord = coord.replace("Point(", "").replace(")", "")
            lon, lat = coord.split(" ")
            cities.append((city, lat, lon))

    with open("cities.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["City", "Latitude", "Longitude"])
        writer.writerows(cities)
    print(f"Successfully wrote {len(cities)} cities to cities.csv")
except Exception as e:
    print(f"Error fetching data: {e}")

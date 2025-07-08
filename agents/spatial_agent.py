import folium
import pandas as pd
from folium.plugins import MarkerCluster
import re
import os

def parse_summary(summary: str) -> dict | None:
    """
    Extracts date, GPS coordinates, fatigue, delay probability, and risk classification from summary string.
    """
    match = re.search(
        r"on\s+(\d{4}-\d{2}-\d{2})[\s\d:]*.*?GPS\s*\(([\d\.-]+),\s*([\d\.-]+)\).*?"
        r"(?:fatigue.*?(high|moderate|low)).*?delay probability (?:was )?(?:marked as )?([\d\.]+).*?"
        r"risk classification (?:was )?(?:marked as )?(high|moderate|low)",
        summary,
        re.IGNORECASE,
    )
    if match:
        try:
            return {
                "date": match.group(1),
                "lat": float(match.group(2)),
                "lon": float(match.group(3)),
                "fatigue": match.group(4).capitalize(),
                "delay_prob": float(match.group(5).rstrip(".")),  # Removes trailing dot
                "risk": match.group(6).capitalize()
            }
        except ValueError:
            return None
    return None

def plot_delay_clusters(summaries: list[str], output_path: str = "static/map.html") -> bool:
    """
    Plots a Folium map of shipment delay clusters and saves to static folder for web use.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    for s in summaries:
        record = parse_summary(s)
        if record:
            records.append(record)

    if not records:
        print("⚠️ No valid geospatial records found.")
        print("⚠️ Map file could not be saved.")
        return False

    df = pd.DataFrame(records)

    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        popup_content = f"""
            📅 <b>Date:</b> {row['date']}<br>
            😴 <b>Fatigue:</b> {row['fatigue']}<br>
            🕒 <b>Delay Prob:</b> {row['delay_prob']}<br>
            ⚠️ <b>Risk:</b> {row['risk']}
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color="red" if row["delay_prob"] > 0.5 else "green")
        ).add_to(marker_cluster)

    try:
        m.save(output_path)
        print(f"🗺️  Map saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving map: {e}")
        return False

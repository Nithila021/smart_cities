import pandas as pd
from sklearn.cluster import KMeans
import folium

# Load data
df = pd.read_csv('NYPD_Complaint_Data_YTD.csv')

# Clean data
df = df.dropna(subset=['lat_lon.latitude', 'lat_lon.longitude', 'ofns_desc'])
crime_types = df['ofns_desc'].unique()

# Step 1: Divide NYC into geographic clusters
coords = df[['lat_lon.latitude', 'lat_lon.longitude']].values
kmeans = KMeans(n_clusters=20, random_state=42)  # 20 zones
df['zone'] = kmeans.fit_predict(coords)

# Step 2: Find dominant crime per zone
zone_crimes = df.groupby(['zone', 'ofns_desc']).size().unstack().fillna(0)
df['dominant_crime'] = zone_crimes.idxmax(axis=1)[df['zone']].values

# Step 3: Create a lookup function
def get_crime_zone(lat, lon):
    point = [[lat, lon]]
    zone = kmeans.predict(point)[0]
    return {
        'zone': zone,
        'dominant_crime': zone_crimes.idxmax(axis=1)[zone],
        'common_crimes': zone_crimes.loc[zone].nlargest(3).to_dict()
    }

# Test with Times Square
print(get_crime_zone(40.7580, -73.9855))
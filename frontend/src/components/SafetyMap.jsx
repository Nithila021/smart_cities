import { MapContainer, TileLayer, CircleMarker, LayersControl } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

const SafetyMap = ({ clusters, center, crimeData }) => {
  const clusterColors = {
    dbscan: '#ec4899',
    demographic: '#db2777',
    density: '#be185d'
  }

  return (
    <div className="map-container" style={{ height: '400px', width: '100%' }}>
      <MapContainer center={center} zoom={13} scrollWheelZoom={true}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        
        <LayersControl position="topright">
          {clusters.dbscan && (
            <LayersControl.Overlay name="DBSCAN Clusters" checked>
              {clusters.dbscan.map((cluster, idx) => (
                <CircleMarker
                  key={`dbscan-${idx}`}
                  center={[cluster.center_lat, cluster.center_lon]}
                  radius={5}
                  color={clusterColors.dbscan}
                  fillOpacity={0.5}
                />
              ))}
            </LayersControl.Overlay>
          )}

          {clusters.demographic && (
            <LayersControl.Overlay name="Demographic Zones">
              {clusters.demographic.map((zone, idx) => (
                <CircleMarker
                  key={`demo-${idx}`}
                  center={[zone.center_lat, zone.center_lon]}
                  radius={5}
                  color={clusterColors.demographic}
                  fillOpacity={0.5}
                />
              ))}
            </LayersControl.Overlay>
          )}

          {clusters.density && (
            <LayersControl.Overlay name="Density Zones">
              {clusters.density.map((density, idx) => (
                <CircleMarker
                  key={`density-${idx}`}
                  center={[density.latitude, density.longitude]}
                  radius={5}
                  color={clusterColors.density}
                  fillOpacity={0.2}
                />
              ))}
            </LayersControl.Overlay>
          )}
        </LayersControl>
      </MapContainer>
    </div>
  )
}

export default SafetyMap
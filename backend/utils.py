"""
Utility module for visualization and additional analysis functions
"""
import folium
from folium.plugins import HeatMap, MarkerCluster

def create_safety_map(lat, lon, nearby_crimes, analysis_result):
    """Create an interactive safety map with crime data"""
    # Create base map centered on location
    safety_map = folium.Map(location=[lat, lon], zoom_start=14)
    
    # Add marker for specified location
    folium.Marker(
        [lat, lon],
        popup=f"Safety Score: {analysis_result['safety_score']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(safety_map)
    
    # Add crime markers
    if not nearby_crimes.empty:
        # Create marker cluster for crimes
        marker_cluster = MarkerCluster().add_to(safety_map)
        
        # Add individual crime markers
        for _, crime in nearby_crimes.iterrows():
            folium.Marker(
                [crime['latitude'], crime['longitude']],
                popup=f"Type: {crime['crime_type']}<br>Distance: {crime['distance']:.2f} km",
                icon=folium.Icon(color='red', icon='warning-sign', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for _, row in nearby_crimes.iterrows()]
        HeatMap(heat_data, radius=15).add_to(safety_map)
    
    # Create a circle showing the safety score
    folium.Circle(
        location=[lat, lon],
        radius=500,  # meters
        popup=f"Safety Score: {analysis_result['safety_score']}",
        color='green' if analysis_result['safety_score'] > 70 else 'orange' if analysis_result['safety_score'] > 40 else 'red',
        fill=True,
        fill_opacity=0.2
    ).add_to(safety_map)
    
    return safety_map

def generate_safety_report(analysis_result, address_str=None):
    """Generate a detailed safety report in text format"""
    safety_score = analysis_result['safety_score']
    
    # Determine safety level
    if safety_score >= 80:
        safety_level = "Very Safe"
    elif safety_score >= 60:
        safety_level = "Safe"
    elif safety_score >= 40:
        safety_level = "Moderate"
    elif safety_score >= 20:
        safety_level = "Concerning"
    else:
        safety_level = "High Risk"
    
    # Build report
    report = []
    report.append(f"SAFETY ANALYSIS REPORT")
    if address_str:
        report.append(f"Location: {address_str}")
    
    report.append(f"\nSAFETY ASSESSMENT")
    report.append(f"Safety Score: {safety_score}/100 ({safety_level})")
    report.append(f"Crime Zone: {analysis_result['zone']}")
    
    report.append(f"\nCRIME PROFILE")
    report.append(f"Dominant Crime: {analysis_result['dominant_crime']}")
    report.append("Common Crimes:")
    for crime, count in analysis_result['common_crimes'].items():
        report.append(f"  - {crime}: {count} incidents")
    
    report.append(f"\nNEARBY ACTIVITY")
    report.append(f"Total Nearby Crimes: {analysis_result['nearby_crime_count']}")
    
    if 'time_analysis' in analysis_result:
        report.append(f"\nTIME PATTERN ANALYSIS")
        time_data = analysis_result['time_analysis']['time_of_day']
        total = sum(time_data.values())
        
        if total > 0:
            report.append("Crime distribution by time of day:")
            for period, count in time_data.items():
                percentage = (count / total) * 100
                report.append(f"  - {period.capitalize()}: {count} crimes ({percentage:.1f}%)")
            
            # Identify high-risk times
            high_risk = max(time_data.items(), key=lambda x: x[1])[0]
            report.append(f"Highest risk time period: {high_risk.capitalize()}")
    
    if 'amenities' in analysis_result:
        report.append(f"\nNEARBY AMENITIES")
        report.append(f"Total amenities within 1 km: {analysis_result['amenities']['nearby_count']}")
        
        if analysis_result['amenities']['type_counts']:
            report.append("Amenities by type:")
            for amenity_type, count in analysis_result['amenities']['type_counts'].items():
                report.append(f"  - {amenity_type.capitalize()}: {count}")
    
    report.append(f"\nSAFETY RECOMMENDATIONS")
    if safety_score < 40:
        report.append("- Exercise increased caution in this area")
        report.append("- Avoid walking alone during high-risk times")
        report.append("- Keep valuables secure and out of sight")
    elif safety_score < 70:
        report.append("- Be aware of your surroundings")
        report.append("- Take normal precautions, especially at night")
    else:
        report.append("- Area appears relatively safe")
        report.append("- Standard safety precautions recommended")
    
    return "\n".join(report)


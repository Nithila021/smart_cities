from flask import Flask, request, jsonify
import json
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Process the message (replace with your actual logic)
    comma_index=user_message.find(",")
    amenity=user_message[:comma_index]
    address=user_message[comma_index+1:].strip()


    def get_coordinates(address):
        geolocator = Nominatim(user_agent="crime_safety_app", timeout=10) # increase timeout
        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            else:
                return None
        except Exception as e:
            print(f"geocoding error: {e}")
            return None

    #user_address = input("Enter an address: ")
    #print(user_address)
    coordinates = get_coordinates(address)

    if coordinates:
        latitude, longitude = coordinates
        print(f"Latitude: {latitude}, Longitude: {longitude}")
        # Now, compare these coordinates with your dataset
    else:
        print("Could not find coordinates for that address.")
    
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

    common_crimes=get_crime_zone(coordinates[0],coordinates[1])
    
    def prompt_gemini(prompt, api_key, model="gemini-1.5-flash"):
        """
        Prompts the Gemini model with a given input.

        Args:
            prompt: The text prompt to send to Gemini.
            api_key: Your Google Generative AI API key.
            model: The Gemini model to use (default: "gemini-pro").

        Returns:
            The generated response from Gemini, or None if an error occurs.
        """
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel(model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error prompting Gemini: {e}")
            return None

    # Example usage (replace with your actual API key):
    api_key = "AIzaSyDqn0jS5wvlPv9DHTSTtAsVgi3t3wBuXHs" # Replace with your API key.
    user_prompt = f'''
Given this Amenity : {amenity}, consider the type of People's Race, Age Group and Gender that would be most likely to visit that Amenity.
Also given these common crimes : {common_crimes}, determine whether it is safe to build that particular amenity at that location. Depending on how relevant the crimes are to the demographic
of the people that would frequent the amenity. '''

    gemini_response = prompt_gemini(user_prompt, api_key)


    if "safety" in user_message.lower():
        response_text = "Based on our analysis, this area has a safety score of 82/100. The main concerns are nighttime lighting and pedestrian access."
        
        # Example graph data
        graph_data = {
            "type": "bar",
            "title": "Safety Metrics Comparison",
            "data": {
                "labels": ["Lighting", "Crime Rate", "Accessibility", "Emergency Services"],
                "datasets": [{
                    "label": "Safety Scores",
                    "data": [75, 88, 92, 85],
                    "backgroundColor": [
                        'rgba(54, 162, 235, 0.6)',
                        'rgba(255, 99, 132, 0.6)',
                        'rgba(75, 192, 192, 0.6)',
                        'rgba(153, 102, 255, 0.6)'
                    ],
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100
                    }
                }
            },
            "notes": "Higher scores indicate safer conditions for each factor."
        }
    else:
        response_text = "I can analyze amenity safety. Try asking about safety in a specific area or type of amenity."
        graph_data = None
    
    return jsonify({
        "text": gemini_response,
        "graph": graph_data
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
import sqlite3
from flask import Flask, request, redirect, url_for, render_template, g, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
'''nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')'''
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
import pandas as pd
import geocoder

import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


app = Flask(__name__)

DATABASE = 'flood_management.db'

# Connect to the SQLite database
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

# Close the database connection at the end of each request
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def create_table():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS forms (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            form_text TEXT NOT NULL,
                            location TEXT,
                            area TEXT,
                            num_people INTEGER,
                            help_needed TEXT,
                            status TEXT
                        )''')
        db.commit()
    print("Table 'forms' created successfully")
    



def generate_location_map():
    db = sqlite3.connect('flood_management.db')
    cursor = db.cursor()
    cursor.execute('''SELECT area FROM forms''')
    area_name = cursor.fetchall()
    db.close()
    df = pd.DataFrame(area_name, columns=['Area'])
    area_counts = df['Area'].value_counts()
    fig_location_map = plt.figure()  
    area_counts.plot(kind='bar', color = '#53b3dcc4' )
    plt.xlabel('Area')
    plt.ylabel('Count')
    plt.title('Locations of Flood Reports')
    plt.xticks(rotation=0)
    # plt.grid(True)
    buffer = BytesIO()
    fig_location_map.savefig(buffer, format='png')
    buffer.seek(0)
    chart_location = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return chart_location

    
# Function to generate the bar chart of help needed categories
def generate_help_needed_chart():
    # Fetch help needed data from the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''SELECT help_needed FROM forms''')
    help_needed_data = cursor.fetchall()

    # Process help needed data and generate the chart
    help_categories = {}  # Dictionary to store category counts

    for row in help_needed_data:
        categories = row[0].split(',')
        for category in categories:
            category = category.strip().capitalize()  # Capitalize category names
            help_categories[category] = help_categories.get(category, 0) + 1

    # Sort categories by frequency
    sorted_categories = sorted(help_categories.items(), key=lambda x: x[1], reverse=True)
    categories, num_requests = zip(*sorted_categories)

    fig_help_needed = plt.figure()
    plt.bar(categories, num_requests , color = '#53b3dcc4')
    plt.xlabel('Help Needed Category')
    plt.ylabel('Number of Requests')
    plt.title('Help Needed Distribution')

    buffer = BytesIO()
    fig_help_needed.savefig(buffer, format='png')
    buffer.seek(0)
    chart_help_needed = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return chart_help_needed

# Function to generate the time series chart of form submissions
def generate_time_series_chart():
    db = sqlite3.connect('flood_management.db')
    cursor = db.cursor()
    cursor.execute('''SELECT status, COUNT(*) FROM forms GROUP BY status''')
    data = cursor.fetchall()
    db.close()

    if not data:
        return None  # No data found

    statuses = [row[0] for row in data]
    counts = [row[1] for row in data]

    fig_time_series = plt.figure()
    plt.bar(statuses, counts , color = '#53b3dcc4')
    plt.xlabel('Status')
    plt.ylabel('Number of Forms')
    plt.title('Form Status Distribution')
    plt.grid(True)
    buffer = BytesIO()
    fig_time_series.savefig(buffer, format='png')
    buffer.seek(0)
    chart_time_series = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return chart_time_series

# Function to get address from latitude and longitude coordinates using Google Maps Geocoding API
def get_address_from_coordinates(latitude, longitude):
    # url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={api_key}"
    # response = requests.get(url)
    # data = response.json()
    # if data['status'] == 'OK' and data['results']:
    #     address = data['results'][0]['formatted_address']
    #     return address
    # else:
    #     return None
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
    response = requests.get(url)
    data = response.json()
    if 'error' in data:
        return None
    elif 'display_name' in data:
        return data['display_name']
    else:
        return None
    
def get_weather_details(lat,lon):
    """
    Function to fetch weather details for a given location from a weather API.
    """
    # Replace 'YOUR_API_KEY' with your actual weather API key
    api_key = '01c8404013bcc52100638c0004dda58f'
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'

    try:
        response = requests.get(url)
        data = response.json()

        # Extract relevant weather information
        weather_info = {
            'Location': data['name'],
            'Temperature': data['main']['temp'],
            'Condition': data['weather'][0]['description'],
            'WindSpeed': data['wind']['speed'],
            'Humidity': data['main']['humidity']
        }

        return weather_info
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# def get_area_name_from_geocode(latitude, longitude, api_key):
#     url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={api_key}"
#     response = requests.get(url)
#     data = response.json()
#     if data['status'] == 'OK':
#         results = data['results']
#         for result in results:
#             # Loop through address components to find sublocality
#             for component in result['address_components']:
#                 if 'sublocality_level_1' in component['types']:
#                     return component['long_name']
                   

def get_area_name_from_geocode(latitude, longitude):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
    response = requests.get(url)
    data = response.json()
    if 'error' in data:
        return None
    elif 'address' in data and 'suburb' in data['address']:
        return data['address']['suburb']
    elif 'address' in data and 'town' in data['address']:
        return data['address']['town']
    elif 'address' in data and 'city' in data['address']:
        return data['address']['city']
    else:
        return None


# Function to extract key-value pairs from a forum message
def extract_key_value_pairs(message):
    # Tokenize the message into words
    words = word_tokenize(message)

    # Tag parts of speech for each word
    tagged_words = pos_tag(words)

    # Perform named entity recognition (NER)
    named_entities = ne_chunk(tagged_words)

    # Initialize variables to store extracted information
    no_of_people = None
    help_needed = []

    '''# Traverse the named entities tree and extract locations
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree) and entity.label() == 'GPE':
            location = ' '.join([word for word, tag in entity.leaves()])
            break'''

    # Iterate over words in the message
    for i, word in enumerate(words):
        # Check for keywords indicating the number of people
        if word.lower() in ["approximately", "about", "around", "roughly", "nearly", "estimated", "almost"]:
            # Extract the number of people mentioned in the message
            try:
                no_of_people = words[i+1]
            except IndexError:
                pass

        # Check for keywords indicating help needed
        if word.lower() in ["food","elder","elders","differently abled","disabled", "shelter","shelters","boat","boats", "evacuation", "pregnant", "elders", "children","baby","babies","women","lady","ladies","medical","medical aid"]:
            help_needed.append(word)

    # Build key-value pairs
    key_value_pairs = {'No of People': no_of_people, 'Help Needed': help_needed}
    return key_value_pairs

def get_user_location():
    try:
        # Fetching the user's location based on IP address
        location = geocoder.ip('me')
        if location:
            latitude = location.latlng[0]
            longitude = location.latlng[1]
            return latitude, longitude
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None

# Route to handle form deletion
@app.route('/delete-form', methods=['POST'])
def delete_form():
    form_id = request.form.get('form_id')
    if form_id:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''DELETE FROM forms WHERE id=?''', (form_id,))
        db.commit()
    return redirect(url_for('admin_page'))

# Route to handle form deletion for user page
@app.route('/user-delete-form', methods=['POST'])
def user_delete_form():
    form_id = request.form.get('form_id')
    if form_id:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''DELETE FROM forms WHERE id=?''', (form_id,))
        db.commit()
    return redirect(url_for('user_page'))


@app.route('/', methods=['GET', 'POST'])
def landing_page():
    error_message = None
    if request.method == 'POST':
        if request.form['submit_button'] == 'SignUp / Register':
            username = request.form['username']
            password = request.form['password']
            user_type = request.form['user_type']
            if user_type == 'organization':
                verfcode = request.form['verfcode']
                if verfcode != '123456':  # Function to verify the verification code
                    error_message = "Invalid verification code."
                    return render_template('landing_page.html', error_message=error_message)
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''SELECT * FROM users WHERE username=?''', (username,))
            existing_user = cursor.fetchone()
            if existing_user:
                error_message = "Username already exists. Please choose a different username."
                # time.sleep(1)  # Pause execution for 10 seconds
            else:
                cursor.execute('''INSERT INTO users (username, password, user_type)
                                  VALUES (?, ?, ?)''', (username, password, user_type))
                db.commit()
                return redirect(url_for('landing_page'))
        elif request.form['submit_button'] == 'SignIn':
            username = request.form['username']
            password = request.form['password']
            user_type = request.form['user_type']
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''SELECT * FROM users WHERE username=? AND password=? AND user_type=?''', (username, password, user_type))
            user = cursor.fetchone()
            if user:
                if user[3] == 'organization':
                    return redirect(url_for('admin_page'))
                elif user[3] == 'user':
                    lat, lon = get_user_location()
                    weather_info = get_weather_details(lat, lon)
                    return redirect(url_for('user_page', weather_info = weather_info))
            else:
                error_message = "Invalid username or password."
                # time.sleep(10)  # Pause execution for 10 seconds
                # error_message = ""
    return render_template('landing_page.html', error_message=error_message)


# Admin Page - Display form entries table and allow status updates
@app.route('/admin', methods=['GET', 'POST'])
def admin_page():
    if request.method == 'POST':
        # Update the status of a form entry
        form_id = request.form['form_id']
        new_status = request.form['new_status']

        db = get_db()
        cursor = db.cursor()
        cursor.execute('''UPDATE forms SET status = ? WHERE id = ?''', (new_status, form_id))
        db.commit()

    # Fetch form entries from the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''SELECT id, form_text, location, status,help_needed, num_people FROM forms ORDER BY id DESC''')
    form_entries = cursor.fetchall()
    
    chart_location = generate_location_map()
    chart_help_needed = generate_help_needed_chart()
    chart_time_series = generate_time_series_chart()

    return render_template('admin_page.html', chart_location=chart_location, chart_help_needed=chart_help_needed, chart_time_series=chart_time_series, form_entries=form_entries)


# User Page - Display form text history and form for new entry
@app.route('/user', methods=['GET', 'POST'])
def user_page():
    lat, lon = get_user_location()
    weather_info = get_weather_details(lat, lon)
    create_table()  # Call create_table() to ensure the table exists
    
    if request.method == 'POST':
        form_text = request.form['form_text']
        location = request.form['location']
        key_value_pairs = extract_key_value_pairs(form_text)
        # api_key = "AIzaSyD-0bIHc2ZSvP1vBaauXfNUevp9FMDk-eE"
        
        # url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
        # response = requests.get(url)
        # data = response.json()
        # if data['status'] == 'OK' and data['results']:
        #     # latitude = data['results'][0]['geometry']['location']['lat']
        #     # longitude = data['results'][0]['geometry']['location']['lng']
        #     latitude = 11.08545
        #     longitude = 76.99616
        lat_str, lon_str = location.split(',')
        latitude = float(lat_str)
        longitude = float(lon_str) 
        # Get address from latitude and longitude
        address = get_address_from_coordinates(latitude, longitude)
        # address = '365 KGiSL Campus, Saravanampatti, Coimbatore, TamilNadu - 641035'
        area_name = get_area_name_from_geocode(latitude, longitude)
        # weather_info = get_weather_details(latitude,longitude)
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''INSERT INTO forms (form_text, location,area, status, help_needed, num_people) 
                          VALUES (?, ?, ?, ?, ?,?)''', (form_text, address,area_name, 'pending', ', '.join(key_value_pairs['Help Needed']), key_value_pairs['No of People']))
        db.commit()
        return redirect(url_for('user_page', weather_info= weather_info))
    else:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''SELECT form_text, location, status FROM forms ORDER BY id DESC''')
        form_entries = cursor.fetchall()
        return render_template('user_page.html', form_entries=form_entries, weather_info= weather_info)



# Intents and example phrases
intents = {
    "flood_warning": ["Is there a flood warning in my area?",
                      "Any alerts about flooding nearby?",
                      "Are there any flood warnings for Mumbai?"],

    "evacuation_information": ["Where can I find evacuation routes?",
                               "How do I evacuate during a flood?",
                               "When should I evacuate?"],

    "safe_locations": ["Where is the nearest safe location during a flood?",
                       "What are the safest areas during a flood?",
                       "Is Pune safe during floods?"],

    "emergency_contacts": ["What are the emergency contact numbers for flood-related issues?",
                           "Who should I contact for rescue during a flood?",
                           "How can I reach emergency services during a flood?"],

    "preparing_for_floods": ["What should I include in my flood emergency kit?",
                             "How can I prepare my home for a flood?",
                             "Any tips for flood preparedness?"],

    "assistance_with_relief": ["How can I help with flood relief efforts?",
                               "Where can I donate for flood victims?",
                               "Are there any volunteer opportunities for flood relief?"],

    "road_conditions": ["Are the roads flooded in Kolkata?",
                        "Can I travel by road during a flood?",
                        "How do I check road conditions during floods?"],

    "safety_precautions": ["What safety precautions should I take during a flood?",
                           "How can I stay safe during a flood?",
                           "Any tips for staying safe in flooded areas?"],

    "health_concerns": ["How can I avoid water-borne diseases during floods?",
                        "What should I do if I get injured during a flood?",
                        "Are there any health risks associated with floods?"],

    "shelter_information": ["Where can I find temporary shelters during a flood?",
                            "How do I find shelter if my home is flooded?",
                            "Are there any relief camps set up in Bangalore?"],

    "pets_and_livestock_safety": ["How can I ensure the safety of my pets during a flood?",
                                   "What should I do with my livestock during a flood?",
                                   "Any tips for protecting animals during floods?"],

    "water_supply": ["Is the water supply safe during a flood?",
                     "How can I ensure access to clean water during floods?",
                     "What should I do if my water source gets contaminated?"],

    "electricity_and_power_outages": ["What should I do if there's a power outage during a flood?",
                                       "How can I stay safe with electrical appliances during floods?",
                                       "Are there any precautions for electrical safety during floods?"],

    "communication_during_floods": ["How can I stay informed during a flood?",
                                     "What communication channels are available during floods?",
                                     "How do I keep in touch with family and friends during floods?"],

    "insurance_claims": ["How do I file an insurance claim for flood damage?",
                         "What does my insurance cover for flood damage?",
                         "Any tips for dealing with insurance companies after a flood?"],

    "recovery_and_rehabilitation": ["What steps should I take for recovery after a flood?",
                                     "How can I rebuild my home after flood damage?",
                                     "Are there any government assistance programs for flood victims?"],

    "weather_forecast": ["What is the weather forecast for the next few days in Chennai?",
                         "Are there predictions for heavy rainfall in my area?",
                         "Should I expect more flooding based on the weather forecast?"],

    "community_support": ["How can I connect with other flood-affected individuals in my community?",
                          "Are there support groups for flood victims?",
                          "Where can I find emotional support during and after floods?"],

    "public_transport_availability": ["Is public transportation available during floods?",
                                       "Are there changes to public transport schedules due to flooding?",
                                       "How can I commute if roads are flooded?"],

    "government_assistance_programs": ["What government assistance programs are available for flood victims?",
                                       "How do I apply for government aid after a flood?",
                                       "Are there any relief packages for flood-affected areas?"]
}

# Preprocess intents
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
spell = Speller()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation and token not in stopwords]
    return " ".join(tokens)

preprocessed_intents = {}
for intent, phrases in intents.items():
    preprocessed_phrases = [preprocess_text(phrase) for phrase in phrases]
    preprocessed_intents[intent] = preprocessed_phrases

# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = []
y = []
for intent, phrases in preprocessed_intents.items():
    X.extend(phrases)
    y.extend([intent] * len(phrases))

X = tfidf_vectorizer.fit_transform(X)

# Intent classification
classifier = LogisticRegression()
classifier.fit(X, y)

# Responses corresponding to intents
responses = {
    "flood_warning": ["Evacuate immediately. Move to higher ground if possible.",
                      "Stay tuned to local news and weather channels for updates.",
                      "Avoid driving through flooded areas. Seek higher ground."],

    "evacuation_information": ["Evacuation routes can be found on local government websites.",
                               "Follow instructions from local authorities for safe evacuation.",
                               "Evacuate to higher ground if you're in a flood-prone area. Stay safe."],

    "safe_locations": ["Move to higher ground and wait for rescue teams.",
                       "If possible, move to a higher floor or roof of a sturdy building.",
                       "Avoid contact with floodwater as it may be contaminated. Seek safety."],

    "emergency_contacts": ["Call emergency services at 112 or your local emergency number.",
                           "Contact local authorities for assistance.",
                           "Seek help from nearby shelters or relief camps."],

    "preparing_for_floods": ["Stay indoors if it's safe.",
                             "Listen to official advisories and follow instructions.",
                             "Prepare an emergency kit with essentials like water, food, and medications."],

    "assistance_with_relief": ["Donate to reputable organizations providing flood relief.",
                               "Volunteer with local NGOs involved in flood relief efforts.",
                               "Share information about relief efforts on social media to raise awareness."],

    "road_conditions": ["Avoid driving through flooded areas.",
                        "Check traffic updates from reliable sources before traveling.",
                        "Follow alternative routes recommended by authorities."],

    "safety_precautions": ["Avoid walking or swimming through floodwaters.",
                           "Stay away from power lines and electrical wires.",
                           "Turn off utilities if instructed to do so by authorities."],

    "health_concerns": ["Boil water before drinking if there's a risk of contamination.",
                        "Seek medical attention if you experience any health issues.",
                        "Use insect repellent to prevent mosquito-borne diseases."],

    "shelter_information": ["Find shelter in a sturdy building on higher ground.",
                            "Use community shelters or relief camps if available.",
                            "Take essential items like food, water, and clothing to the shelter."],

    "pets_and_livestock_safety": ["Bring pets indoors and ensure they have enough food and water.",
                                   "Move livestock to higher ground if possible.",
                                   "Keep pets on a leash or in carriers to prevent them from wandering."],

    "water_supply": ["Store clean water in containers for drinking and cooking.",
                     "Avoid using water from wells or boreholes if they may be contaminated.",
                     "Boil water before using it for drinking, cooking, or personal hygiene."],

    "electricity_and_power_outages": ["Use flashlights or battery-operated lanterns instead of candles.",
                                       "Turn off electrical appliances to prevent electrical fires.",
                                       "Report power outages to the utility company and follow their instructions."],

    "communication_during_floods": ["Stay updated with local news and weather reports.",
                                     "Use social media to check on friends and family.",
                                     "Keep a battery-powered radio for emergency updates if power goes out."],

    "insurance_claims": ["Document flood damage with photographs or videos.",
                         "Contact your insurance company to file a claim as soon as possible.",
                         "Keep records of all communications with your insurance company."],

    "recovery_and_rehabilitation": ["Seek assistance from government agencies or NGOs for rebuilding.",
                                     "Follow safety guidelines when cleaning up flood-damaged areas.",
                                     "Take care of your mental health and seek support if needed."],

    "weather_forecast": ["Check weather forecasts regularly for updates on rainfall.",
                         "Be prepared for possible evacuation if heavy rainfall is predicted.",
                         "Follow instructions from authorities based on weather forecasts."],

    "community_support": ["Join local community groups providing support to flood victims.",
                          "Share resources and information with neighbors and community members.",
                          "Offer help to those in need, such as elderly or disabled individuals."],

    "public_transport_availability": ["Use public transportation only if it's safe to do so.",
                                       "Check with transport authorities for updates on service disruptions.",
                                       "Follow instructions from transport officials during floods."],

    "government_assistance_programs": ["Apply for government aid programs for flood victims.",
                                       "Check eligibility criteria and documentation required for assistance.",
                                       "Seek help from local government offices or helplines for assistance."]
}

def get_intent(query):
    preprocessed_query = preprocess_text(query)
    query_vec = tfidf_vectorizer.transform([preprocessed_query])
    predicted_intent = classifier.predict(query_vec)[0]
    return predicted_intent

def generate_response(intent):
    if intent in responses:
        return random.choice(responses[intent])
    else:
        return "I'm sorry, I couldn't understand that. Can you please rephrase your query?"


@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/chatbot_interaction', methods=['POST'])
def chatbot_interaction():
    user_input = request.json.get('user_input')
    if user_input.lower() == 'exit' or user_input.lower() == 'bye' or user_input.lower() == 'thank you':
        return jsonify("Thank you for using the Flood Assistance Chatbot. Stay safe!")
    corrected_input = spell(user_input)
    intent = get_intent(corrected_input)
    response = generate_response(intent)
    return jsonify(response)
    
if __name__ == '__main__':
    app.run(debug=True, port = 5000)

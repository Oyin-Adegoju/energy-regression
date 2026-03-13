import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime

# --- Database setup ---
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lights REAL,
            tdewpoint REAL,
            rh_6 REAL,
            day_of_week INTEGER,
            windspeed REAL,
            hour INTEGER,
            month INTEGER,
            prediction REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(inputs, prediction):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, lights, tdewpoint, rh_6, day_of_week, 
                                  windspeed, hour, month, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        inputs[0], inputs[1], inputs[2], inputs[3],
        inputs[4], inputs[5], inputs[6],
        prediction
    ))
    conn.commit()
    conn.close()

def get_predictions():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

# Database aanmaken als die nog niet bestaat
init_db()

# Model en feature-namen laden
model = joblib.load('models/model.joblib')
features = joblib.load('models/features.joblib')

# --- Titel ---
st.title('Energieverbruik Voorspeller')
st.write('Voorspel het energieverbruik van huishoudelijke apparaten (in Wh) op basis van omgevingsfactoren.')

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(['Voorspelling', 'Voorspelgeschiedenis', 'Simulatie'])

# --- Tab 1: Handmatige voorspelling ---
with tab1:
    st.header('Voer de waarden in:')

    lights = st.number_input('Lights (energieverbruik verlichting in Wh)', min_value=0, max_value=80, value=0)
    tdewpoint = st.number_input('Tdewpoint (dauwpunttemperatuur in °C)', min_value=-10.0, max_value=16.0, value=3.0)
    rh_6 = st.number_input('RH_6 (luchtvochtigheid kamer 6 in %)', min_value=0.0, max_value=100.0, value=50.0)
    day_of_week = st.selectbox('Dag van de week', options=[0, 1, 2, 3, 4, 5, 6],
                               format_func=lambda x: ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 
                                                       'Vrijdag', 'Zaterdag', 'Zondag'][x])
    windspeed = st.number_input('Windspeed (windsnelheid in m/s)', min_value=0.0, max_value=15.0, value=4.0)
    hour = st.slider('Uur van de dag', min_value=0, max_value=23, value=12)
    month = st.selectbox('Maand', options=[1, 2, 3, 4, 5], 
                         format_func=lambda x: ['Januari', 'Februari', 'Maart', 'April', 'Mei'][x-1])

    if st.button('Voorspel energieverbruik'):
        input_values = [lights, tdewpoint, rh_6, day_of_week, windspeed, hour, month]
        input_data = np.array([input_values])
        prediction = model.predict(input_data)[0]
        
        # Opslaan in database
        save_prediction(input_values, prediction)
        
        st.success(f'Voorspeld energieverbruik: {prediction:.2f} Wh')
        st.info('Voorspelling opgeslagen in de database.')

# --- Tab 2: Geschiedenis bekijken ---
with tab2:
    st.header('Opgeslagen voorspellingen')
    
    history = get_predictions()
    
    if len(history) == 0:
        st.write('Nog geen voorspellingen opgeslagen.')
    else:
        st.dataframe(history, use_container_width=True)
        
        # Grafiek: voorspellingen over tijd
        st.subheader('Voorspellingen over tijd')
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        
        st.line_chart(data=history, x='timestamp', y='prediction')
        
        st.write(f'Totaal aantal voorspellingen: {len(history)}')

# --- Tab 3: Simulatie met synthetische data ---
with tab3:
    st.header('Simulatie met synthetische data')
    st.write('Laad de synthetische dataset en simuleer voorspellingen om app-gebruik na te bootsen.')
    
    if st.button('Start simulatie'):
        synthetic = pd.read_csv('data/synthetic_data.csv')
        
        predictions = model.predict(synthetic[features])
        
        # Elke voorspelling opslaan in de database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        
        for i in range(len(synthetic)):
            row = synthetic.iloc[i]
            c.execute('''
                INSERT INTO predictions (timestamp, lights, tdewpoint, rh_6, day_of_week, 
                                          windspeed, hour, month, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                row['lights'], row['Tdewpoint'], row['RH_6'], row['day_of_week'],
                row['Windspeed'], row['hour'], row['month'],
                predictions[i]
            ))
        
        conn.commit()
        conn.close()
        
        st.success(f'{len(synthetic)} voorspellingen gesimuleerd en opgeslagen!')
        st.write('Ga naar het tabblad "Voorspelgeschiedenis" om de resultaten te bekijken.')
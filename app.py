import streamlit as st
import joblib
import numpy as np

# Model en feature-namen laden
model = joblib.load('models/model.joblib')
features = joblib.load('models/features.joblib')

# Titel
st.title('Energieverbruik Voorspeller')
st.write('Voorspel het energieverbruik van huishoudelijke apparaten (in Wh) op basis van omgevingsfactoren.')

# Inputvelden
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

# Voorspelling
if st.button('Voorspel energieverbruik'):
    input_data = np.array([[lights, tdewpoint, rh_6, day_of_week, windspeed, hour, month]])
    prediction = model.predict(input_data)[0]
    st.success(f'Voorspeld energieverbruik: {prediction:.2f} Wh')
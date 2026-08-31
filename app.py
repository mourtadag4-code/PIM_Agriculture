import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import requests  # pour appeler l'API
import os        # pour les variables d'environnement (clé API)

today = datetime.now().strftime("%Y-%m-%d")
# --- Configuration de la page ---
st.set_page_config(page_title="Intelligent Météo & Agricole", page_icon="🌾", layout="wide")
st.title("🌾 Plateforme d'Intelligent Météo & Agricole")
st.markdown("*Prédisez la pluie, détectez les risques de maladies et optimisez l'irrigation*")

# --- Chargement des modèles et des features ---
@st.cache_resource
def load_models():
    models = {}

    # 1. Modèle pluie (XGBoost)
    try:
        models['rain'] = joblib.load('xgb_rain_classifier.pkl')
        models['scaler_rain'] = joblib.load('scaler.pkl')
        models['features_rain'] = joblib.load('features_rain.pkl')
        #st.success("✅ Modèle pluie chargé")
    except Exception as e:
        st.error(f"Erreur chargement modèle pluie : {e}")
        models['rain'] = None

    # 2. Modèle maladie (Random Forest) - avec extraction si dictionnaire
    try:
        disease_obj = joblib.load('modele_risque_maladie.joblib')
        # Si c'est un dictionnaire, on suppose que la clé 'model' contient le modèle
        if isinstance(disease_obj, dict):
            if 'model' in disease_obj:
                models['disease'] = disease_obj['model']
                st.info("ℹ️ Modèle maladie extrait d'un dictionnaire")
            else:
                # Sinon, on prend le premier objet qui a predict_proba
                for key, val in disease_obj.items():
                    if hasattr(val, 'predict_proba'):
                        models['disease'] = val
                        #st.info(f"ℹ️ Modèle maladie extrait de la clé '{key}'")
                        break
                else:
                    models['disease'] = None
                    st.error("❌ Aucun modèle valide trouvé dans le dictionnaire")
        else:
            models['disease'] = disease_obj
            #st.success("✅ Modèle maladie chargé")
        
        # Vérification que le modèle a bien predict_proba
        if models['disease'] is not None and not hasattr(models['disease'], 'predict_proba'):
            st.error("❌ L'objet chargé n'a pas de méthode predict_proba")
            models['disease'] = None

        # Chargement des features
        models['features_disease'] = joblib.load('features_disease.pkl')
        
    except Exception as e:
        st.error(f"Erreur chargement modèle maladie : {e}")
        models['disease'] = None
        models['features_disease'] = None

    # 3. Modèle sécheresse
    try:
        models['drought'] = joblib.load('modele_secheresse_7jours.pkl')
        models['scaler_drought'] = joblib.load('scaler_secheresse.pkl')
        models['features_drought'] = joblib.load('features_secheresse.pkl')
        #st.success("✅ Modèle sécheresse chargé")
    except Exception as e:
        st.error(f"Erreur chargement modèle sécheresse : {e}")
        models['drought'] = None

    return models

models = load_models()

@st.cache_data(ttl=600)  # cache pendant 10 minutes pour éviter trop d'appels
def get_weather(city, api_key):
    """Récupère les données météo actuelles pour une ville via OpenWeatherMap."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Erreur API : {response.status_code} - {response.json().get('message', '')}")
        return None
    data = response.json()
    
    # Extraire les données nécessaires (avec valeurs par défaut si absentes)
    weather_data = {
        'temp_day': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'pressure': data['main']['pressure'],
        'wind_speed': data['wind']['speed'],
        'pop_value': data.get('pop', 0.0),  # probabilité de pluie (disponible dans forecast, pas toujours dans current)
        'rain_mm': data.get('rain', {}).get('1h', 0.0)  # pluie des 1 dernière heure
    }
    return weather_data




# Récupérer la clé API (à placer dans .env ou dans les secrets Streamlit)
#API_KEY = os.getenv('OPENWEATHER_API_KEY', '')  # ou 
st.secrets["OPENWEATHER_API_KEY"] #en prod

# --- Sidebar : paramètres météo du jour ---
st.sidebar.header("📋 Données météo du jour")

# Initialisation des session_state avec des floats (si non définis)
if 'temp_day' not in st.session_state:
    st.session_state.temp_day = 30.0
    st.session_state.humidity = 60.0
    st.session_state.pressure = 1013.0
    st.session_state.wind_speed = 5.0
    st.session_state.pop_value = 0.3
    st.session_state.rain_mm = 0.0

# Saisie de la ville et bouton de récupération
city = st.sidebar.text_input("Ville (pour récupération auto)", value="Dakar")
if st.sidebar.button("🌐 Récupérer la météo"):
    if not API_KEY:
        st.sidebar.error("❌ Clé API manquante. Veuillez la définir dans .env ou les secrets.")
    else:
        weather = get_weather(city, API_KEY)
        if weather:
            st.sidebar.success("✅ Données récupérées !")
            # Mettre à jour session_state avec des floats
            st.session_state.temp_day = float(weather['temp_day'])
            st.session_state.humidity = float(weather['humidity'])
            st.session_state.pressure = float(weather['pressure'])
            st.session_state.wind_speed = float(weather['wind_speed'])
            st.session_state.pop_value = float(weather['pop_value'])
            st.session_state.rain_mm = float(weather['rain_mm'])

# Champs de saisie (avec conversion explicite en float)
col1, col2 = st.sidebar.columns(2)
with col1:
    temp_day = st.number_input(
        "Température jour (°C)",
        min_value=15.0,
        max_value=45.0,
        value=float(st.session_state.temp_day),
        step=0.5,
        key="temp_day_input"
    )
    humidity = st.number_input(
        "Humidité (%)",
        min_value=10.0,
        max_value=100.0,
        value=float(st.session_state.humidity),
        step=1.0,
        key="humidity_input"
    )
    wind_speed = st.number_input(
        "Vitesse du vent (m/s)",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state.wind_speed),
        step=0.5,
        key="wind_speed_input"
    )
with col2:
    pressure = st.number_input(
        "Pression (hPa)",
        min_value=1000.0,
        max_value=1030.0,
        value=float(st.session_state.pressure),
        step=1.0,
        key="pressure_input"
    )
    pop = st.number_input(
        "Probabilité de pluie (pop_value)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.pop_value),
        step=0.05,
        key="pop_input"
    )
    rain_mm = st.number_input(
        "Pluie aujourd'hui (mm)",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state.rain_mm),
        step=0.5,
        key="rain_mm_input"
    )

# Mettre à jour session_state avec les nouvelles valeurs (pour que le bouton "Récupérer" les modifie)
st.session_state.temp_day = temp_day
st.session_state.humidity = humidity
st.session_state.pressure = pressure
st.session_state.wind_speed = wind_speed
st.session_state.pop = pop
st.session_state.rain_mm = rain_mm

# --- Fonctions de construction des features ---
def build_features_from_values(feature_names, base_values, expected_n_features=None):
    """
    Construit un tableau de features à partir des valeurs de base.
    Si expected_n_features est donné, force le nombre de colonnes.
    """
    # Initialiser toutes les colonnes à 0
    data = {col: 0 for col in feature_names}
    
    # Remplir les colonnes de base
    for col in feature_names:
        if col in base_values:
            data[col] = base_values[col]
        elif '_lag' in col:
            base = col.split('_lag')[0]
            if base in base_values:
                data[col] = base_values[base]
        elif '_roll7' in col:
            base = col.split('_roll7')[0]
            if base in base_values:
                if base == 'rain':
                    data[col] = base_values['rain_mm'] * 7
                else:
                    data[col] = base_values[base]
        elif col in ['month_sin', 'month_cos', 'doy_sin', 'doy_cos']:
            month = datetime.now().month
            day_of_year = datetime.now().timetuple().tm_yday
            if col == 'month_sin':
                data[col] = np.sin(2 * np.pi * month / 12)
            elif col == 'month_cos':
                data[col] = np.cos(2 * np.pi * month / 12)
            elif col == 'doy_sin':
                data[col] = np.sin(2 * np.pi * day_of_year / 365.25)
            elif col == 'doy_cos':
                data[col] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Créer un DataFrame avec l'ordre des colonnes
    df = pd.DataFrame([data])[feature_names]
    
    # Ajuster le nombre de colonnes si attendu
    if expected_n_features is not None:
        current_n = df.shape[1]
        if current_n < expected_n_features:
            # Ajouter des colonnes factices avec 0
            for i in range(expected_n_features - current_n):
                df[f'dummy_{i}'] = 0
        elif current_n > expected_n_features:
            # Prendre les premières colonnes (ou supprimer les dernières)
            df = df.iloc[:, :expected_n_features]
    return df.values

# --- Fonctions de prédiction ---
def predict_rain():
    if models['rain'] is None:
        return None, None
    base = {
        'temp_day': temp_day,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'pop': pop,
        'rain_mm': rain_mm
    }
    features = build_features_from_values(
        models['features_rain'],
        base,
        expected_n_features=models['scaler_rain'].n_features_in_
    )
    features_scaled = models['scaler_rain'].transform(features)
    proba = models['rain'].predict_proba(features_scaled)[0][1]
    pred = 1 if proba >= 0.35 else 0
    return pred, proba

def predict_disease():
    if models['disease'] is None or models['features_disease'] is None:
        # Fallback règle heuristique
        if temp_day > 28 and humidity > 75:
            return "ÉLEVÉ", 0.9
        elif temp_day > 25 and humidity > 70:
            return "MODÉRÉ", 0.6
        else:
            return "FAIBLE", 0.1
    # Utiliser le modèle (pas de scaler)
    base = {
        'temp_day': temp_day,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'pop': pop,
        'rain_mm': rain_mm
    }
    # On pourrait ajouter d'autres colonnes si nécessaire
    features = build_features_from_values(models['features_disease'], base)
    try:
        proba = models['disease'].predict_proba(features)[0][1]
        risk = "ÉLEVÉ" if proba > 0.5 else "FAIBLE"
        return risk, proba
    except Exception as e:
        st.error(f"Erreur prédiction maladie : {e}")
        return "ERREUR", 0.0

def predict_drought():
    if models['drought'] is None:
        return None, None
    base = {
        'temp_day': temp_day,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'pop': pop,
        'rain_mm': rain_mm
    }
    features = build_features_from_values(
        models['features_drought'],
        base,
        expected_n_features=models['scaler_drought'].n_features_in_
    )
    features_scaled = models['scaler_drought'].transform(features)
    proba = models['drought'].predict_proba(features_scaled)[0][1]
    pred = 1 if proba > 0.5 else 0
    return pred, proba

# --- Interface utilisateur ---
tab1, tab2, tab3 = st.tabs(["🌧️ Prévision pluie", "🦠 Risque maladie", "💧 Recommandation irrigation"])

with tab1:
    st.subheader("Prédiction de pluie pour demain")
    if st.button("Prédire", key="rain_btn"):
        pred, proba = predict_rain()
        if pred is not None:
            col1, col2 = st.columns(2)
            col1.metric("Probabilité de pluie", f"{proba*100:.1f}%")
            if pred == 1:
                col2.success("🌧️ **Il devrait pleuvoir demain**")
                st.info("💡 **Conseil** : réduisez l'irrigation, économisez l'eau.")
            else:
                col2.warning("☀️ **Pas de pluie prévue demain**")
                st.info("💡 **Conseil** : programmez l'irrigation si nécessaire.")
        else:
            st.error("Modèle non disponible.")

with tab2:
    st.subheader("Détection de risques de maladies (champignons)")
    if st.button("Évaluer le risque", key="disease_btn"):
        risk, proba = predict_disease()
        st.metric("Probabilité de risque", f"{proba*100:.1f}%")
        if risk == "ÉLEVÉ":
            st.error("⚠️ **Risque ÉLEVÉ** : conditions chaudes et humides favorables aux champignons.")
            st.warning("💡 **Action** : traitez les cultures avec un fongicide préventif.")
        elif risk == "MODÉRÉ":
            st.warning("🔶 **Risque MODÉRÉ** : surveillez l'apparition de taches ou de moisissures.")
            st.info("💡 **Action** : inspectez régulièrement les cultures.")
        else:
            st.success("✅ **Risque FAIBLE** : conditions favorables, pas d'inquiétude.")

with tab3:
    st.subheader("Recommandation d'irrigation")
    if st.button("Obtenir une recommandation", key="irrig_btn"):
        pred_rain, proba_rain = predict_rain()
        risk_disease, proba_disease = predict_disease()
        pred_drought, proba_drought = predict_drought()

        st.write("### Résultats combinés")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pluie demain", f"{proba_rain*100:.1f}%" if proba_rain is not None else "N/A")
        col2.metric("Risque maladie", risk_disease if risk_disease else "N/A")
        col3.metric("Risque sécheresse", f"{proba_drought*100:.1f}%" if proba_drought is not None else "N/A")

        # Règle de recommandation
        if proba_rain is not None and proba_rain > 0.5:
            st.success("💧 **Arrosage NON recommandé** : pluie probable demain.")
        elif pred_drought is not None and pred_drought == 1:
            st.warning("💧 **Arrosage recommandé** : risque de sécheresse dans les jours à venir.")
        else:
            if risk_disease == "ÉLEVÉ":
                st.warning("💧 **Arrosage modéré** : pluie peu probable mais risque maladie élevé (évitez l'excès d'humidité).")
            else:
                st.info("💧 **Arrosage recommandé** : pas de pluie prévue, sol probablement sec.")

st.sidebar.markdown("---")
st.sidebar.caption(f"📅 Données du : {today}")
st.sidebar.caption("⚠️ Les prédictions sont basées sur des modèles entraînés sur des données historiques.")
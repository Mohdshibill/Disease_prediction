import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diagnox AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .symptom-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f4f8;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load trained model and symptom data"""
    try:
        model = joblib.load('disease_model.pkl')
        symptoms_list = joblib.load('symptoms.pkl')
        
        # Load remedy data
        try:
            remedies_df = pd.read_csv('symptom_precaution.csv')
        except:
            remedies_df = None
            
        # Load descriptions
        try:
            descriptions_df = pd.read_csv('symptom_Description.csv')
        except:
            descriptions_df = None
            
        return model, symptoms_list, remedies_df, descriptions_df
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def create_symptom_vector(selected_symptoms, severity_dict, all_symptoms):
    """Create feature vector from selected symptoms with severity"""
    vector = np.zeros(len(all_symptoms))
    
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            # Normalize severity (1-5) to (0.2-1.0)
            severity = severity_dict.get(symptom, 3)
            vector[idx] = severity / 5.0
    
    return vector.reshape(1, -1)

def predict_disease(model, symptom_vector):
    """Predict disease and return confidence"""
    prediction = model.predict(symptom_vector)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(symptom_vector)[0]
        confidence = np.max(probabilities)
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(model.classes_[i], probabilities[i]) for i in top_indices]
    else:
        confidence = 0.85  # Default confidence for models without probability
        top_diseases = [(prediction, confidence)]
    
    return prediction, confidence, top_diseases

def get_remedies(disease, remedies_df):
    """Get remedies for predicted disease"""
    if remedies_df is None:
        return [
            "Consult with a healthcare professional",
            "Get adequate rest and sleep",
            "Stay hydrated",
            "Follow prescribed medications"
        ]
    
    disease_remedies = remedies_df[remedies_df['Disease'] == disease]
    
    if len(disease_remedies) > 0:
        remedies = []
        for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
            if col in disease_remedies.columns:
                remedy = disease_remedies[col].values[0]
                if pd.notna(remedy):
                    remedies.append(remedy)
        return remedies if remedies else ["Consult with a healthcare professional"]
    
    return ["Consult with a healthcare professional for specific guidance"]

def get_description(disease, descriptions_df):
    """Get description of the disease"""
    if descriptions_df is None:
        return "Please consult with a healthcare professional for detailed information."
    
    disease_desc = descriptions_df[descriptions_df['Disease'] == disease]
    
    if len(disease_desc) > 0 and 'Description' in disease_desc.columns:
        return disease_desc['Description'].values[0]
    
    return "Please consult with a healthcare professional for detailed information."

def find_nearby_doctors(specialty, user_location):
    """Find nearby doctors based on location"""
    st.subheader("🗺️ Find Nearby Doctors")
    
    try:
        geolocator = Nominatim(user_agent="disease_prediction_app")
        location = geolocator.geocode(user_location)
        
        if location:
            # Create map centered on user location
            m = folium.Map(
                location=[location.latitude, location.longitude],
                zoom_start=13
            )
            
            # Add marker for user location
            folium.Marker(
                [location.latitude, location.longitude],
                popup="Your Location",
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
            
            # Simulate nearby doctors (in real app, use Google Places API or similar)
            np.random.seed(42)
            for i in range(5):
                lat_offset = np.random.uniform(-0.02, 0.02)
                lon_offset = np.random.uniform(-0.02, 0.02)
                
                folium.Marker(
                    [location.latitude + lat_offset, location.longitude + lon_offset],
                    popup=f"Dr. {chr(65+i)} - {specialty}",
                    icon=folium.Icon(color='blue', icon='plus-sign')
                ).add_to(m)
            
            folium_static(m)
            
            # Display doctor list
            st.subheader("Nearby Healthcare Providers")
            doctors_data = {
                'Doctor Name': [f'Dr. {chr(65+i)}' for i in range(5)],
                'Specialty': [specialty] * 5,
                'Distance': [f'{np.random.uniform(0.5, 5):.1f} km' for _ in range(5)],
                'Rating': [f'{np.random.uniform(3.5, 5):.1f} ⭐' for _ in range(5)]
            }
            st.dataframe(pd.DataFrame(doctors_data), use_container_width=True)
            
        else:
            st.warning("Location not found. Please enter a valid location.")
            
    except Exception as e:
        st.error(f"Error finding location: {str(e)}")
        st.info("💡 Tip: Use Google Maps API or similar service for production")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model, symptoms_list, remedies_df, descriptions_df = load_model_and_data()
    
    if model is None or symptoms_list is None:
        st.error("⚠️ Model not found! Please run train_model.py first.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info(
            "This AI-powered system predicts potential diseases based on symptoms "
            "and their severity. It provides confidence levels, recommendations, "
            "and helps locate nearby healthcare providers."
        )
        
        st.header("⚠️ Disclaimer")
        st.warning(
            "This tool is for informational purposes only and should not replace "
            "professional medical advice, diagnosis, or treatment."
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Enter Your Symptoms")
        
        # Multi-select for symptoms
        selected_symptoms = st.multiselect(
            "Select symptoms you're experiencing:",
            options=symptoms_list,
            help="You can select multiple symptoms"
        )
        
        # Severity selection
        if selected_symptoms:
            st.subheader("Rate Symptom Severity")
            severity_dict = {}
            
            for symptom in selected_symptoms:
                severity = st.slider(
                    f"{symptom.replace('_', ' ').title()}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1 = Mild, 5 = Severe"
                )
                severity_dict[symptom] = severity
        
        # Predict button
        if st.button("🔮 Predict Disease", type="primary", use_container_width=True):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    # Create symptom vector
                    symptom_vector = create_symptom_vector(
                        selected_symptoms, severity_dict, symptoms_list
                    )
                    
                    # Predict
                    disease, confidence, top_diseases = predict_disease(model, symptom_vector)
                    
                    # Store in session state
                    st.session_state.prediction = disease
                    st.session_state.confidence = confidence
                    st.session_state.top_diseases = top_diseases
                    st.session_state.selected_symptoms = selected_symptoms
                    st.session_state.severity_dict = severity_dict
    
    with col2:
        st.header("ℹ️ Quick Info")
        st.metric("Total Symptoms Available", len(symptoms_list))
        st.metric("Symptoms Selected", len(selected_symptoms) if selected_symptoms else 0)
        
        if 'confidence' in st.session_state:
            confidence_pct = st.session_state.confidence * 100
            st.metric("Prediction Confidence", f"{confidence_pct:.1f}%")
    
    # Display prediction results
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Main prediction
        disease = st.session_state.prediction
        confidence = st.session_state.confidence
        
        confidence_class = (
            'confidence-high' if confidence > 0.8 
            else 'confidence-medium' if confidence > 0.6 
            else 'confidence-low'
        )
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Condition: {disease}</h2>
            <p class="{confidence_class}">Confidence Level: {confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Alternative predictions
        if len(st.session_state.top_diseases) > 1:
            st.subheader("🔄 Alternative Possibilities")
            for alt_disease, alt_conf in st.session_state.top_diseases[1:]:
                st.write(f"- {alt_disease}: {alt_conf*100:.1f}% confidence")
        
        # Description
        description = get_description(disease, descriptions_df)
        st.subheader("📝 Description")
        st.info(description)
        
        # Remedies
        remedies = get_remedies(disease, remedies_df)
        st.subheader("💊 Recommended Actions")
        for i, remedy in enumerate(remedies, 1):
            st.markdown(f"{i}. {remedy}")
        
        # Determine specialty
        specialty_map = {
            'fever': 'General Physician',
            'cough': 'Pulmonologist',
            'stomach': 'Gastroenterologist',
            'skin': 'Dermatologist',
            'heart': 'Cardiologist'
        }
        
        specialty = 'General Physician'
        for key, value in specialty_map.items():
            if key in disease.lower():
                specialty = value
                break
        
        # Location input
        st.markdown("---")
        user_location = st.text_input(
            "📍 Enter your location to find nearby doctors",
            placeholder="e.g., New York, NY or zip code"
        )
        
        if user_location:
            find_nearby_doctors(specialty, user_location)

if __name__ == "__main__":
    main()
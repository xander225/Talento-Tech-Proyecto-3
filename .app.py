import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el escalador
model = joblib.load('modelo_forest.joblib')
scaler = joblib.load('scaler.joblib')

# Título de la aplicación
st.title('Predictor de Niveles de Ansiedad y Depresión')

st.write("""
Por favor, responde a las siguientes preguntas en una escala. 
Esto es una herramienta de orientación y no reemplaza un diagnóstico profesional.
""")

# Crear sliders para cada característica en el sidebar
st.sidebar.header('Parámetros de Entrada del Usuario')

def user_input_features():
    self_esteem = st.sidebar.slider('Autoestima (Self Esteem)', 0, 25, 10)
    mental_health_history = st.sidebar.slider('Historial de Salud Mental (0: No, 1: Sí)', 0, 1, 0)
    sleep_quality = st.sidebar.slider('Calidad del Sueño', 0, 5, 3)
    noise_level = st.sidebar.slider('Nivel de Ruido en el Entorno', 0, 5, 3)
    living_conditions = st.sidebar.slider('Condiciones de Vida', 0, 5, 3)
    safety = st.sidebar.slider('Percepción de Seguridad', 0, 5, 3)
    basic_needs = st.sidebar.slider('Necesidades Básicas Cubiertas', 0, 5, 3)
    academic_performance = st.sidebar.slider('Rendimiento Académico', 0, 5, 3)
    study_load = st.sidebar.slider('Carga de Estudio', 0, 5, 3)
    teacher_student_relationship = st.sidebar.slider('Relación Profesor-Estudiante', 0, 5, 3)
    future_career_concerns = st.sidebar.slider('Preocupación por la Carrera Futura', 0, 5, 3)
    social_support = st.sidebar.slider('Apoyo Social', 0, 3, 2)
    peer_pressure = st.sidebar.slider('Presión de Grupo', 0, 5, 3)
    bullying = st.sidebar.slider('Acoso (Bullying)', 0, 5, 2)
    stress_level = st.sidebar.slider('Nivel de Estrés', 0, 5, 3)

    data = {'self_esteem': self_esteem,
            'mental_health_history': mental_health_history,
            'sleep_quality': sleep_quality,
            'noise_level': noise_level,
            'living_conditions': living_conditions,
            'safety': safety,
            'basic_needs': basic_needs,
            'academic_performance': academic_performance,
            'study_load': study_load,
            'teacher_student_relationship': teacher_student_relationship,
            'future_career_concerns': future_career_concerns,
            'social_support': social_support,
            'peer_pressure': peer_pressure,
            'bullying': bullying,
            'stress_level': stress_level}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Obtener la entrada del usuario
input_df = user_input_features()

# Mostrar la entrada del usuario
st.subheader('Tus respuestas:')
st.write(input_df)

# Cuando el usuario presione el botón, hacer la predicción
if st.button('Predecir'):
    # Escalar los datos de entrada
    input_scaled = scaler.transform(input_df)
    
    # Hacer la predicción
    prediction = model.predict(input_scaled)
    
    # Mostrar el resultado
    st.subheader('Resultados de la Predicción')
    st.write(f"**Nivel de Ansiedad Predicho:** {prediction[0][0]:.2f}")
    st.write(f"**Nivel de Depresión Predicho:** {prediction[0][1]:.2f}")

    st.success('Predicción realizada con éxito!')
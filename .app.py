import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y el escalador
model = joblib.load('modelo_forest.joblib')
scaler = joblib.load('scaler.joblib')

# HARDCODEAR las medias y desviaciones estándar de tus datos de entrenamiento
# (Reemplaza con los valores que obtuviste de tu Colab)
GLOBAL_MEAN_ANXIETY = 11.06 # Ejemplo: Reemplaza con tu valor real
GLOBAL_STD_ANXIETY = 6.12  # Ejemplo: Reemplaza con tu valor real
GLOBAL_MEAN_DEPRESSION = 12.56 # Ejemplo: Reemplaza con tu valor real
GLOBAL_STD_DEPRESSION = 7.73   # Ejemplo: Reemplaza con tu valor real

# Título de la aplicación
st.title('Evaluación de Riesgo de Ansiedad y Depresión en Estudiantes') # Título un poco más descriptivo

st.write("""
Por favor, responde a las siguientes preguntas en una escala. 
Esta herramienta proporciona una orientación basada en datos y **no reemplaza un diagnóstico profesional**.
""")

# Crear sliders para cada característica en el sidebar
st.sidebar.header('Parámetros de Entrada del Usuario')

def user_input_features():
    # Asegúrate de que los rangos (min_value, max_value) y el valor por defecto (value)
    # sean apropiados para tus datos y la escala de cada característica.
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

# --- INICIO DE CAMBIOS ---

# st.subheader('Tus respuestas:') # Eliminado
# st.write(input_df) # Eliminado

# Función para describir la posición en la campana de Gauss
def describe_position(score, mean, std, condition_name):
    st.write(f"### Nivel de {condition_name} Predicho: {score:.2f}")

    if score < mean - 1.5 * std:
        st.info(f"Tu nivel de {condition_name} está **significativamente por debajo del promedio** de los estudiantes en nuestro estudio. Esto sugiere un bienestar notable en este aspecto.")
        st.image('https://i.imgur.com/k9b6G9r.png') # Puedes cambiar esto por una imagen local o URL de un indicador bajo
    elif score < mean - 0.5 * std:
        st.info(f"Tu nivel de {condition_name} está **ligeramente por debajo del promedio**. Indica un buen manejo en comparación con la mayoría.")
        st.image('https://i.imgur.com/k9b6G9r.png') # Puedes cambiar esto por una imagen local o URL de un indicador bajo
    elif score < mean + 0.5 * std:
        st.warning(f"Tu nivel de {condition_name} está **dentro del rango promedio** de los estudiantes en nuestro estudio. Es similar a la experiencia de la mayoría.")
        st.image('https://i.imgur.com/lJ4W7Pq.png') # Puedes cambiar esto por una imagen local o URL de un indicador medio
    elif score < mean + 1.5 * std:
        st.warning(f"Tu nivel de {condition_name} está **ligeramente por encima del promedio**. Podría indicar que experimentas más desafíos en esta área que la mayoría.")
        st.image('https://i.imgur.com/H6x56yB.png') # Puedes cambiar esto por una imagen local o URL de un indicador alto
    else:
        st.error(f"Tu nivel de {condition_name} está **significativamente por encima del promedio**. Esto sugiere un riesgo elevado o una necesidad de atención en este aspecto.")
        st.image('https://i.imgur.com/H6x56yB.png') # Puedes cambiar esto por una imagen local o URL de un indicador muy alto
    st.markdown("---")


# Cuando el usuario presione el botón, hacer la predicción
if st.button('Obtener Evaluación'): # Cambié el texto del botón
    # Escalar los datos de entrada
    input_scaled = scaler.transform(input_df)
    
    # Hacer la predicción
    prediction = model.predict(input_scaled)
    
    # Mostrar el resultado de forma descriptiva
    st.subheader('Resultados de tu Evaluación:')
    
    # Llamar a la función descriptiva para Ansiedad
    describe_position(prediction[0][0], GLOBAL_MEAN_ANXIETY, GLOBAL_STD_ANXIETY, "Ansiedad")
    
    # Llamar a la función descriptiva para Depresión
    describe_position(prediction[0][1], GLOBAL_MEAN_DEPRESSION, GLOBAL_STD_DEPRESSION, "Depresión")

    st.success('Evaluación completada. Recuerda que esto es una orientación.')

# --- FIN DE CAMBIOS ---
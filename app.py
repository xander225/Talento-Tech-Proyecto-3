import streamlit as st
import pandas as pd
import joblib

# --- Carga del Modelo y Escalador de Clasificación ---
try:
    model = joblib.load('ml/modelo_clasificador.joblib')
    scaler = joblib.load('ml/scaler_clasificador.joblib')
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo. Asegúrate de que 'modelo_clasificador.joblib' y 'scaler_clasificador.joblib' estén en la misma carpeta que app.py.")
    st.stop()


# --- Interfaz de Usuario ---
def st_info_centered(text, icon="ℹ️"): # Añadimos un parámetro 'icon' con un emoji por defecto
    """
    Muestra un cuadro de información con el texto centrado y un ícono (emoji).
    
    Args:
        text (str): El texto a mostrar.
        icon (str): El emoji Unicode a usar como ícono. Ejemplos: "⚠️", "💡", "🧠", "✅".
    """
    st.markdown(f"""
    <div style="background-color: #325A78; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0">
        <p style="text-align: center; margin-bottom: 0;">
            {icon} {text}  </p>
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Evaluador de Bienestar Estudiantil", layout="wide")

st.markdown("<h1 style='text-align: center;'>Evaluador de Bienestar Estudiantil 🧠</h1>", unsafe_allow_html=True)

st.write(" ")
st.write("""
Esta herramienta te ayuda a comprender tu posible nivel de riesgo de ansiedad y depresión en comparación con otros estudiantes. 
Por favor, responde a las preguntas en el panel de la izquierda, de la manera más honesta posible. Luego, haz clic en 'Obtener Evaluación' para obtener tu resultado.

""")
st_info_centered("Importante: Esta es una herramienta de orientación y no reemplaza un diagnóstico médico o psicológico profesional.",icon="⚠️")
st.markdown("---")

#st.markdown("<div style='text-align: center;'>Si estas en un dispositivo movil dale a las flechas de arriba a la izquierda para ver las preguntas<div>",unsafe_allow_html=True)
st_info_centered("Si estás en un dispositivo móvil dale a las flechas de arriba a la izquierda para ver las preguntas")

st.markdown("---")


# --- Entradas del Usuario en la Barra Lateral ---
st.sidebar.header('Responde a estas preguntas:')

def user_input_features():
    """
    Crea los sliders para que el usuario ingrese sus datos.
    """
    # Notas: Ajusta los valores por defecto (value) y rangos (min_value, max_value) si es necesario.
    self_esteem = st.sidebar.slider('Nivel de autoestima', 0, 25, 12)
    mental_health_history = st.sidebar.slider('¿Tienes un historial de salud mental diagnosticado? (0: No, 1: Sí)', 0, 1, 0)
    sleep_quality = st.sidebar.slider('Calidad general de tu sueño', 0, 5, 3)
    noise_level = st.sidebar.slider('Nivel de ruido en tu lugar de estudio/vida', 0, 5, 2)
    living_conditions = st.sidebar.slider('Calidad de tus condiciones de vida', 0, 5, 3)
    safety = st.sidebar.slider('¿Qué tan seguro/a te sientes en tu entorno?', 0, 5, 4)
    basic_needs = st.sidebar.slider('¿Qué tan bien cubiertas están tus necesidades básicas (comida, vivienda)?', 0, 5, 4)
    academic_performance = st.sidebar.slider('Tu rendimiento académico actual', 0, 5, 3)
    study_load = st.sidebar.slider('Tu carga de estudio actual', 0, 5, 3)
    teacher_student_relationship = st.sidebar.slider('Calidad de la relación con tus profesores', 0, 5, 3)
    future_career_concerns = st.sidebar.slider('Nivel de preocupación por tu futura carrera', 0, 5, 3)
    social_support = st.sidebar.slider('Nivel de apoyo social que percibes (familia, amigos)', 0, 3, 2)
    peer_pressure = st.sidebar.slider('Nivel de presión de grupo que sientes', 0, 5, 2)
    bullying = st.sidebar.slider('¿Has sido víctima de acoso (bullying)?', 0, 5, 1)
    stress_level = st.sidebar.slider('Tu nivel general de estrés', 0, 5, 3)

    data = {
        'self_esteem': self_esteem,
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
        'stress_level': stress_level
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Lógica de Predicción y Visualización ---

# Muestra el botón en el cuerpo principal de la página
if st.button('Obtener Evaluación', type="primary", use_container_width=True):
    
    # Escalar los datos de entrada
    input_scaled = scaler.transform(input_df)
    
    # Hacer la predicción
    prediction = model.predict(input_scaled)
    
    # Extraer las predicciones categóricas
    riesgo_ansiedad_predicho = prediction[0][0] 
    riesgo_depresion_predicho = prediction[0][1]
    
    st.markdown("---")
    st.subheader('Resultados de tu Evaluación:')

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Nivel de Riesgo de Ansiedad:**")
        if riesgo_ansiedad_predicho == 'Riesgo Muy Alto':
            st.error(f"**{riesgo_ansiedad_predicho}**")
            st.image("img/1.jpg", width=100) # Imagen para riesgo Muy Alto
            st.write("Tu perfil sugiere una probabilidad significativamente elevada de experimentar síntomas de ansiedad. Se recomienda encarecidamente buscar apoyo profesional.")
        elif riesgo_ansiedad_predicho == 'Riesgo Alto':
            st.warning(f"**{riesgo_ansiedad_predicho}**")
            st.image("img/2.jpg", width=100) # Imagen para riesgo Alto
            st.write("Tu perfil indica una probabilidad elevada de síntomas de ansiedad. Sería beneficioso explorar técnicas de manejo del estrés y considerar hablar con un consejero.")
        elif riesgo_ansiedad_predicho == 'Riesgo Moderado':
            st.info(f"**{riesgo_ansiedad_predicho}**")
            st.image("img/2.5.jpg", width=100) # Imagen para riesgo Moderado
            st.write("Tu perfil se encuentra en un rango común, con algunos indicadores de ansiedad. Es un buen momento para enfocarse en hábitos de bienestar emocional.")
        else: # Riesgo Bajo
            st.success(f"**{riesgo_ansiedad_predicho}**")
            st.image("img/3.jpg", width=100) # Imagen para riesgo Bajo (Aquí usaremos img/1 o img/2.5 como la "menos grave")
            st.write("Tu perfil sugiere un bajo riesgo de síntomas de ansiedad. ¡Sigue manteniendo tus hábitos saludables!")

    with col2:
        st.write(f"**Nivel de Riesgo de Depresión:**")
        if riesgo_depresion_predicho == 'Riesgo Muy Alto':
            st.error(f"**{riesgo_depresion_predicho}**")
            st.image("img/1.jpg", width=100) # Imagen para riesgo Muy Alto
            st.write("Tu perfil sugiere una probabilidad significativamente elevada de experimentar síntomas de depresión. Es crucial buscar apoyo profesional lo antes posible.")
        elif riesgo_depresion_predicho == 'Riesgo Alto':
            st.warning(f"**{riesgo_depresion_predicho}**")
            st.image("img/2.jpg", width=100) # Imagen para riesgo Alto
            st.write("Tu perfil indica una probabilidad elevada de síntomas de depresión. Hablar con un profesional de la salud mental podría ser de gran ayuda.")
        elif riesgo_depresion_predicho == 'Riesgo Moderado':
            st.info(f"**{riesgo_depresion_predicho}**")
            st.image("img/2.5.jpg", width=100) # Imagen para riesgo Moderado
            st.write("Tu perfil se encuentra en un rango común, pero con algunos indicadores de estado de ánimo bajo. Presta atención a tu autocuidado y busca apoyo si lo necesitas.")
        else: # Riesgo Bajo
            st.success(f"**{riesgo_depresion_predicho}**")
            st.image("img/5.jpg", width=100) # Imagen para riesgo Bajo
            st.write("Tu perfil sugiere un bajo riesgo de síntomas de depresión. ¡Continúa cuidando tu bienestar!")
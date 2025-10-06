# app.py - TiempoTuyo Web App
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# Configuración
st.set_page_config(page_title="TiempoTuyo", page_icon="📱")

st.title("⏳ TiempoTuyo")
st.write("Tu asistente personal de bienestar digital con IA.")

# Datos de entrenamiento (simulados)
data = {
    'minutos_uso': [30, 45, 120, 15, 200, 80, 140],
    'notificaciones': [10, 20, 35, 5, 50, 18, 40],
    'uso_excesivo': [0, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Entrenar modelo
X = df[['minutos_uso', 'notificaciones']]
y = df['uso_excesivo']
modelo = LogisticRegression()
modelo.fit(X, y)

# Inputs del usuario
st.header("📊 Tu actividad de hoy")

minutos = st.slider("¿Cuántos minutos usaste el celular hoy?", 0, 300, 120)
notis = st.slider("¿Cuántas notificaciones recibiste hoy?", 0, 100, 30)

if st.button("Analizar mi día"):
    pred = modelo.predict([[minutos, notis]])

    if pred[0] == 1:
        st.error("⛔ TiempoTuyo detecta un posible uso excesivo del celular.")
        
        # IA Coach
        st.subheader("🤖 Coach Digital IA")
        chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")
        mensaje = f"Hoy usé el celular {minutos} minutos y recibí {notis} notificaciones. ¿Qué me recomiendas?"
        respuesta = chatbot(mensaje, max_length=60, do_sample=True)[0]['generated_text']
        st.write("💬", respuesta)
    else:
        st.success("✅ ¡Buen trabajo! Estás usando tu celular de forma equilibrada.")
        st.balloons()

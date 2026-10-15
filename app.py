import io
import base64
import platform
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# 1. Configuración de página
st.set_page_config(
    page_title="Gestos AI | Interfaz Multimodal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS Avanzado + SVG Icons + Glassmorphism
st.markdown("""
<style>
    /* Estilos globales y fondo general */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #070a12 100%);
        color: #f3f4f6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Ocultar elementos por defecto */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Barra lateral estilo Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* Tarjetas estilo Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Encabezados y títulos con degradado neón */
    .gradient-text {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Subtítulos */
    .sub-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Contenedor de cámara */
    div[data-testid="stCameraInput"] {
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 10px;
        background: rgba(15, 23, 42, 0.3);
    }

    /* Estilo de íconos SVG Inline */
    .icon-svg {
        width: 22px;
        height: 22px;
        fill: none;
        stroke: #38bdf8;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        vertical-align: middle;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Carga diferida del modelo Keras en caché
@st.cache_resource
def load_keras_model():
    try:
        from keras.models import load_model
        model = load_model('keras_model.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar 'keras_model.h5': {str(e)}")
        st.info("Asegúrate de haber subido los archivos 'keras_model.h5' y 'labels.txt' a la raíz de tu repositorio de GitHub.")
        return None

# 4. Carga dinámica de etiquetas
def load_labels():
    try:
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception:
        # Etiquetas por defecto si no encuentra el txt
        return ["0 Izquierda", "1 Arriba", "2 Derecha"]

# 5. Barra Lateral - Navegación e Información del Sistema
with st.sidebar:
    st.markdown("""
        <h1 class='gradient-text' style='font-size: 1.6rem;'>
            <svg class="icon-svg" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
            Vision AI
        </h1>
        <p class='sub-text'>Reconocimiento de Gestos</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <h3 style='color: #38bdf8; font-size: 1rem; display: flex; align-items: center; gap: 8px;'>
            <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            Información del Entorno
        </h3>
    """, unsafe_allow_html=True)
    st.caption(f"🐍 **Python:** v{platform.python_version()}")
    st.caption("🧠 **Motor:** Keras / TensorFlow")
    st.caption("⚡ **Modelo:** Teachable Machine")

# 6. ENCABEZADO PRINCIPAL
st.markdown("""
    <div class="glass-card">
        <h1 class="gradient-text" style="margin:0;">
            <svg class="icon-svg" style="width: 32px; height: 32px;" viewBox="0 0 24 24"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v6"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 0 1 2 2v4a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
            Clasificación de Gestos e Imágenes
        </h1>
        <p class="sub-text" style="margin-top: 8px;">
            Captura una fotografía para inferir la clase y probabilidad usando tu red neuronal entrenada en <b>Teachable Machine</b>.
        </p>
    </div>
""", unsafe_allow_html=True)

# Cargar Modelo y Etiquetas
model = load_keras_model()
labels = load_labels()

if model:
    img_file_buffer = st.camera_input("Capturar imagen desde la cámara web", key="camera")

    if img_file_buffer is not None:
        # Procesamiento de la Imagen con Pillow
        img = Image.open(img_file_buffer).convert("RGB")
        img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized)

        # Normalización [-1, 1]
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1.0
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Inferencia con Keras
        with st.spinner("Procesando clasificación con IA..."):
            prediction = model.predict(data)
            class_index = np.argmax(prediction[0])
            confidence = float(prediction[0][class_index])

        # Layout en 2 Columnas
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color: #38bdf8; margin-top:0; display: flex; align-items: center; gap: 8px;">
                        <svg class="icon-svg" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                        Captura Procesada (224x224)
                    </h3>
            """, unsafe_allow_html=True)
            st.image(img_resized, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color: #38bdf8; margin-top:0; display: flex; align-items: center; gap: 8px;">
                        <svg class="icon-svg" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        Predicción Detectada
                    </h3>
            """, unsafe_allow_html=True)
            
            # Nombre de la clase detectada
            class_name = labels[class_index] if class_index < len(labels) else f"Clase {class_index}"
            
            st.markdown(f"<h2 style='color: #38bdf8; font-size: 1.8rem; margin-bottom: 5px;'>{class_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub-text'>Confianza: <b>{confidence * 100:.2f}%</b></p>", unsafe_allow_html=True)
            st.progress(confidence)

            st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.2); margin: 15px 0;'>", unsafe_allow_html=True)

            # Tabla comparativa de todas las probabilidades
            probs_data = [
                {
                    "Clase / Gesto": labels[i] if i < len(labels) else f"Clase {i}",
                    "Probabilidad": f"{float(prob) * 100:.2f}%"
                }
                for i, prob in enumerate(prediction[0])
            ]
            df_probs = pd.DataFrame(probs_data)
            st.dataframe(df_probs, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

# Footer informativo
st.markdown("---")
st.markdown("<p style='text-align: center;' class='sub-text'>Aplicaciones con Visión Artificial | Teachable Machine + Streamlit</p>", unsafe_allow_html=True)

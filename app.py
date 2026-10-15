import io
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

# 2. Inyección de CSS estilo Glassmorphism
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #070a12 100%);
        color: #f3f4f6;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

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

    .gradient-text {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .sub-text {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    div[data-testid="stCameraInput"] {
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 10px;
        background: rgba(15, 23, 42, 0.3);
    }

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

# 3. Función de carga compatible con Teachable Machine
@st.cache_resource
def load_keras_model():
    try:
        # Forzamos tf_keras si Keras 3 está presente en el entorno
        import os
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tensorflow as tf

        # Usar tf.keras o tf_keras directamente
        try:
            import tf_keras as keras_engine
            from tf_keras.layers import DepthwiseConv2D
        except ImportError:
            import tensorflow.keras as keras_engine
            from tensorflow.keras.layers import DepthwiseConv2D

        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        model = keras_engine.models.load_model('keras_model.h5', custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar 'keras_model.h5': {str(e)}")
        return None

# 4. Carga de etiquetas
def load_labels():
    try:
        with open('labels.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception:
        return ["0 Izquierda", "1 Arriba", "2 Derecha"]

# 5. Barra Lateral
with st.sidebar:
    st.markdown("""
        <h1 class='gradient-text' style='font-size: 1.6rem;'>
            <svg class="icon-svg" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
            Vision AI
        </h1>
        <p class='sub-text'>Reconocimiento de Gestos</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(f"🐍 **Python:** v{platform.python_version()}")
    st.caption("🧠 **Framework:** TensorFlow / Keras")

# 6. ENCABEZADO PRINCIPAL
st.markdown("""
    <div class="glass-card">
        <h1 class="gradient-text" style="margin:0;">
            <svg class="icon-svg" style="width: 32px; height: 32px;" viewBox="0 0 24 24"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v6"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 0 1 2 2v4a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
            Clasificación de Gestos e Imágenes
        </h1>
        <p class="sub-text" style="margin-top: 8px;">
            Captura una fotografía para inferir la clase usando el modelo de <b>Teachable Machine</b>.
        </p>
    </div>
""", unsafe_allow_html=True)

labels = load_labels()
model = load_keras_model()

if model is None:
    st.warning("⚠️ No se pudo cargar el archivo 'keras_model.h5'. Verifica que esté presente en la raíz del repositorio.")
else:
    img_file_buffer = st.camera_input("Capturar imagen desde la cámara web", key="camera")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer).convert("RGB")
        img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized)

        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1.0
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        with st.spinner("Procesando clasificación con IA..."):
            prediction = model.predict(data)
            class_index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_index])

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color: #38bdf8; margin-top:0;">📸 Fotograma Procesado</h3>
            """, unsafe_allow_html=True)
            st.image(img_resized, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="glass-card">
                    <h3 style="color: #38bdf8; margin-top:0;">📊 Resultado</h3>
            """, unsafe_allow_html=True)
            
            class_name = labels[class_index] if class_index < len(labels) else f"Clase {class_index}"
            
            st.markdown(f"<h2 style='color: #38bdf8; font-size: 1.8rem; margin-bottom: 5px;'>{class_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='sub-text'>Confianza: <b>{confidence * 100:.2f}%</b></p>", unsafe_allow_html=True)
            st.progress(confidence)

            st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.2); margin: 15px 0;'>", unsafe_allow_html=True)

            probs_data = [
                {
                    "Clase": labels[i] if i < len(labels) else f"Clase {i}",
                    "Probabilidad": f"{float(prob) * 100:.2f}%"
                }
                for i, prob in enumerate(prediction[0])
            ]
            df_probs = pd.DataFrame(probs_data)
            st.dataframe(df_probs, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center;' class='sub-text'>Aplicaciones con Visión Artificial | Teachable Machine + Streamlit</p>", unsafe_allow_html=True)

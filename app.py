import base64
import platform
import numpy as np
from PIL import Image
import streamlit as st

# 1. Configuración de página de Streamlit
st.set_page_config(
    page_title="Reconocimiento de Imágenes | Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección CSS Glassmorphism
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

# Encabezado con versión de Python
st.markdown(f"""
    <div style='background: rgba(15, 23, 42, 0.6); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.2); display: inline-block; margin-bottom: 15px;'>
        <p class='sub-text' style='margin:0;'>🐍 <b>Versión de Python:</b> {platform.python_version()}</p>
    </div>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown("""
    <div class="glass-card">
        <h1 class="gradient-text" style="margin:0;">
            <svg class="icon-svg" style="width: 32px; height: 32px;" viewBox="0 0 24 24"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v6"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 0 1 2 2v4a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
            Reconocimiento de Imágenes
        </h1>
        <p class="sub-text" style="margin-top: 8px;">
            Inferencia en tiempo real utilizando el modelo de Teachable Machine.
        </p>
    </div>
""", unsafe_allow_html=True)

# Intenta mostrar la imagen inicial en base64 solo si existe el archivo
try:
    with open("OIG5.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <img src="data:image/jpeg;base64,{b64}" width="350" style="border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">
        </div>
    """, unsafe_allow_html=True)
except Exception:
    pass

# Carga segura del modelo
@st.cache_resource
def load_tm_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import DepthwiseConv2D

        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        return load_model('keras_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False), None
    except Exception as e:
        return None, str(e)

model, model_error = load_tm_model()

with st.sidebar:
    st.markdown("""
        <h1 class='gradient-text' style='font-size: 1.6rem;'>
            <svg class="icon-svg" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
            Vision AI
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")

if model_error:
    st.error(f"❌ Error al cargar 'keras_model.h5': {model_error}")
    st.info("Verifica que el archivo 'keras_model.h5' esté presente en la raíz del repositorio.")
else:
    img_file_buffer = st.camera_input("Toma una Foto")

    if img_file_buffer is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(img_file_buffer)

        newsize = (224, 224)
        img = img.resize(newsize)
        img_array = np.array(img)

        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)

        st.markdown("""<div class="glass-card">""", unsafe_allow_html=True)
        if prediction[0][0] > 0.5:
            st.markdown(f"<h2 style='color: #38bdf8;'>Izquierda, con Probabilidad: {prediction[0][0]:.4f}</h2>", unsafe_allow_html=True)
        if prediction[0][1] > 0.5:
            st.markdown(f"<h2 style='color: #818cf8;'>Arriba, con Probabilidad: {prediction[0][1]:.4f}</h2>", unsafe_allow_html=True)
        st.markdown("""</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center;' class='sub-text'>Aplicaciones con Visión Artificial | Interfaz Multimodal</p>", unsafe_allow_html=True)

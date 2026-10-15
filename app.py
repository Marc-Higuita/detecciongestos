# 3. Función segura para cargar Keras manejando incompatibilidad de Teachable Machine
@st.cache_resource
def load_keras_model():
    try:
        import tensorflow as tf
        from keras.models import load_model
        from keras.layers import DepthwiseConv2D

        # Parche para omitir el argumento 'groups' incompatible en Keras/Teachable Machine
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        return load_model('keras_model.h5', custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Error al cargar 'keras_model.h5': {str(e)}")
        return None

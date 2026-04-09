import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Clasificador de Flores con IA", page_icon="🌻", layout="centered")

# --- CONSTANTES ---
# Basado en tu entrenamiento
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE = (180, 180) 

# --- FUNCIONES CORE ---

@st.cache_resource
def load_flower_model():
    """Carga el modelo .keras una sola vez."""
    try:
        # Cargamos el modelo completo (incluyendo la capa de normalización integrada)
        model = keras.models.load_model('flores.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar 'flores.keras': {e}")
        return None

def predict_flower(img, model):
    """Procesa la imagen y retorna las probabilidades."""
    # 1. Ajustar tamaño a 180x180
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    
    # 2. Convertir a array
    img_array = keras.utils.img_to_array(img)
    
    # 3. Expandir dimensiones para crear el batch (1, 180, 180, 3)
    # No normalizamos manualmente aquí porque ya está "dentro del modelo"
    img_array = tf.expand_dims(img_array, 0) 
    
    # 4. Predicción
    predictions = model.predict(img_array)
    
    # Si el modelo termina en una capa densa sin softmax, aplicamos softmax aquí.
    # Si ya tiene softmax, esto no afectará el orden del resultado.
    score = tf.nn.softmax(predictions[0])
    
    return score.numpy()

# --- DISEÑO DE LA INTERFAZ ---

def main():
    st.title("🌻 Clasificador de Flores con IA")
    st.write("Sube una foto de una flor y la IA identificará si es una margarita, diente de león, rosa, girasol o tulipán.")
    
    model = load_flower_model()
    if model is None: st.stop()

    # Sidebar para opciones de carga
    st.sidebar.header("Opciones de Imagen")
    option = st.sidebar.selectbox(
        "Método de entrada:",
        ("Subir archivo", "URL de imagen", "Tomar Foto")
    )

    img_raw = None

    if option == "Subir archivo":
        img_raw = st.file_uploader("Cargar imagen (.jpg, .png)", type=["jpg", "jpeg", "png"])
    elif option == "URL de imagen":
        url = st.text_input("URL de la imagen:")
        if url:
            try:
                response = requests.get(url)
                img_raw = BytesIO(response.content)
            except: st.warning("URL no válida.")
    else:
        img_raw = st.camera_input("Capturar flor")

    # Área de visualización y resultados
    if img_raw:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(img_raw)
        
        with col1:
            st.image(image, caption="Imagen seleccionada", use_container_width=True)
        
        with col2:
            with st.spinner('Clasificando...'):
                # Ejecutar predicción
                probs = predict_flower(image, model)
                idx = np.argmax(probs)
                res_class = CLASS_NAMES[idx]
                res_prob = probs[idx] * 100

                st.subheader(f"Resultado: {res_class.capitalize()}")
                st.progress(float(probs[idx]))
                st.write(f"**Confianza:** {res_prob:.2f}%")

        # Gráfico comparativo
        st.divider()
        st.subheader("Probabilidades por categoría")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#E6E6E6'] * len(CLASS_NAMES)
        colors[idx] = '#FF4B4B' # Resaltar la clase ganadora
        
        ax.bar(CLASS_NAMES, probs * 100, color=colors)
        ax.set_ylabel("Confianza (%)")
        st.pyplot(fig)

    # --- PIE DE PÁGINA ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p style="margin-bottom:0px;">Desarrollado por <b>Alfredo Díaz</b></p>
            <p style="font-size: 0.8em; color: gray;">UNAB © Derechos Reservados</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
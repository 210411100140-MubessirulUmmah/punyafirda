import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import os


# Load Model 
MODEL_PATH = "best_model_fixed.h5"
model = None
# st.write("File exists:", os.path.exists(MODEL_PATH))


if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error("Gagal memuat model:")
        st.exception(e)  # ⬅️ Tampilkan error lengkapnya di Streamlit
        model = None
else:
    st.error("File model tidak ditemukan.")
    model = None

# Preprocessing
def apply_clahe(cv_img):
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final_img

def preprocess_image(image):
    # Convert PIL to RGB numpy, lalu ke BGR untuk OpenCV
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize ke 224x224
    img_resized = cv2.resize(img_bgr, (224, 224))

    # Terapkan CLAHE seperti di notebook
    img_clahe = apply_clahe(img_resized)

    # Convert ke grayscale
    img_gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)

    # Normalisasi ke [-1, 1]
    img_scaled = img_gray.astype(np.float32)
    img_scaled = img_scaled / 127.5 - 1.0  # ← ini penting untuk MobileNetV2

    # Expand dimensi jadi (1, 224, 224, 1)
    img_input = np.expand_dims(img_scaled, axis=(0, -1))

    return img_input

st.title("Klasifikasi Penyakit Lumpy Skin pada Sapi")

uploaded_file = st.file_uploader("Upload Gambar Sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write(f"Nama file: {uploaded_file.name}")
    st.write(f"Ukuran file: {uploaded_file.size} bytes")
             
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    # Preprocessing
    processed_image = preprocess_image(image)

    if model is not None:
    # Prediksi
        pred_prob = model.predict(processed_image)[0][0]
        predicted_label = "Lumpy Skin" if pred_prob > 0.5 else "Normal Skin"
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

        # Output
        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** `{predicted_label}`")
        st.write(f"**Confidence:** `{confidence:.2f}`")
    else:
        st.warning("Model belum dimuat. Tidak dapat melakukan prediksi.")

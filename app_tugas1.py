import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(layout="wide")
st.title("🧠 Aplikasi Pemrosesan Citra Wajah")

DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image, np.array(image)


def _preprocess_gray_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blur


def _ensure_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# ==========================================
# SIDEBAR MENU
# ==========================================
menu = st.sidebar.selectbox("📌 Main Menu", [
    "Register Wajah",
    "Image Processing"
])

if menu == "Image Processing":
    sub_menu = st.sidebar.radio("🔧 Sub Menu", [
        "Data (Input)",
        "Technique",
        "Convolution",
        "Morphology",
        "Feature Detection & Matching",
        "Unsupervised Learning (K-Means)"
    ])

# ==========================================
# REGISTER WAJAH
# ==========================================
if menu == "Register Wajah":
    st.subheader("📸 Register Wajah Baru")

    name = st.text_input("Masukkan Nama")
    uploaded = st.file_uploader("Upload Foto Wajah", type=["jpg", "png", "jpeg"])

    if uploaded and name:
        person_folder = os.path.join(DATASET_PATH, name)

        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        image, img = _load_image(uploaded)

        file_count = len(os.listdir(person_folder))
        file_path = os.path.join(person_folder, f"{file_count + 1}.jpg")

        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        st.success(f"✅ Wajah {name} berhasil disimpan!")
        st.image(image, width=250)

# ==========================================
# IMAGE PROCESSING
# ==========================================
elif menu == "Image Processing":
    uploaded = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        image, img = _load_image(uploaded)

        st.image(image, caption="Input Image", width=300)

        # ==========================================
        # DATA
        # ==========================================
        if sub_menu == "Data (Input)":
            st.subheader("📂 Data Input")
            st.info("Gambar digunakan sebagai input untuk semua proses.")

        # ==========================================
        # TECHNIQUE
        # ==========================================
        elif sub_menu == "Technique":
            st.subheader("⚙️ Technique (Preprocessing)")

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            resized = cv2.resize(img, (300, 300))
            gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            median_blur = cv2.medianBlur(gray, 5)

            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Grayscale")
                st.image(equalized, caption="Histogram Equalization")
                st.image(gaussian_blur, caption="Gaussian Blur")
            with col2:
                st.image(resized, caption="Resize")
                st.image(median_blur, caption="Median Blur")

        # ==========================================
        # CONVOLUTION
        # ==========================================
        elif sub_menu == "Convolution":
            st.subheader("🧮 Convolution")

            gray, blur = _preprocess_gray_blur(img)

            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            sharpen = cv2.filter2D(blur, -1, kernel)

            sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, 3)
            sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, 3)

            sobel_x = cv2.convertScaleAbs(sobel_x)
            sobel_y = cv2.convertScaleAbs(sobel_y)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(sharpen, caption="Sharpen")
            with col2:
                st.image(sobel_x, caption="Sobel X")
            with col3:
                st.image(sobel_y, caption="Sobel Y")

        # ==========================================
        # MORPHOLOGY
        # ==========================================
        elif sub_menu == "Morphology":
            st.subheader("🔬 Morphology")

            gray, blur = _preprocess_gray_blur(img)

            threshold = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            kernel = np.ones((3, 3), np.uint8)

            erosion = cv2.erode(threshold, kernel, iterations=1)
            dilation = cv2.dilate(threshold, kernel, iterations=1)
            opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(threshold, caption="Threshold")
                st.image(erosion, caption="Erosion")
            with col2:
                st.image(dilation, caption="Dilation")
            with col3:
                st.image(opening, caption="Opening")
                st.image(closing, caption="Closing")

        # ==========================================
        # FACE RECOGNITION
        # ==========================================
        elif sub_menu == "Feature Detection & Matching":
            st.subheader("Face Recognition")

            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            try:
                result = DeepFace.find(
                    img_path=temp_path,
                    db_path=DATASET_PATH,
                    enforce_detection=False
                )

                if len(result) > 0 and len(result[0]) > 0:
                    matched_path = result[0].iloc[0]['identity']
                    name = os.path.basename(os.path.dirname(matched_path))

                    st.success(f"✅ Hasil: {name}")
                else:
                    st.error("❌ Wajah tidak dikenali")

            except Exception as e:
                st.error(f"Error: {str(e)}")

        # ==========================================
        # UNSUPERVISED LEARNING (K-MEANS)
        # ==========================================
        elif sub_menu == "Unsupervised Learning (K-Means)":
            st.subheader("🎨 Unsupervised Learning (K-Means Clustering)")

            st.info(
                "Mengelompokkan warna piksel menggunakan K-Means untuk "
                "melakukan segmentasi (memisahkan objek dari latar "
                "belakang berdasarkan warna)."
            )

            n_clusters = st.slider(
                "Pilih jumlah kelompok warna (K)",
                min_value=2, max_value=10, value=2
            )

            img_rgb = _ensure_rgb(img)
            pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(pixel_values)
            centers = np.uint8(kmeans.cluster_centers_)

            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(img_rgb.shape)

            st.image(
                segmented_image,
                caption=f"Hasil Segmentasi K-Means (K={n_clusters})",
                width=400
            )

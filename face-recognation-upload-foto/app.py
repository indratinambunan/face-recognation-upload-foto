import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from deepface import DeepFace
from sklearn.cluster import KMeans # <--- TAMBAHAN: Import KMeans

st.set_page_config(layout="wide")
st.title("🧠 Aplikasi Pemrosesan Citra Wajah")

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# =========================
# SIDEBAR MENU
# =========================
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
        "Unsupervised Learning (K-Means)" # <--- TAMBAHAN: Menu baru untuk K-Means
    ])

# =========================
# REGISTER WAJAH
# =========================
if menu == "Register Wajah":
    st.subheader("📸 Register Wajah Baru")

    name = st.text_input("Masukkan Nama")
    uploaded = st.file_uploader("Upload Foto Wajah", type=["jpg","png","jpeg"])

    if uploaded and name:
        person_folder = os.path.join(dataset_path, name)

        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        image = Image.open(uploaded)
        img = np.array(image)

        file_count = len(os.listdir(person_folder))
        file_path = os.path.join(person_folder, f"{file_count+1}.jpg")

        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        st.success(f"✅ Wajah {name} berhasil disimpan!")
        st.image(image, width=250)

# =========================
# IMAGE PROCESSING
# =========================
elif menu == "Image Processing":
    uploaded = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

    if uploaded is not None:
        image = Image.open(uploaded)
        img = np.array(image)

        st.image(image, caption="Input Image", width=300)

        # =========================
        # DATA
        # =========================
        if sub_menu == "Data (Input)":
            st.subheader("📂 Data Input")
            st.info("Gambar digunakan sebagai input untuk semua proses.")

        # =========================
        # TECHNIQUE
        # =========================
        elif sub_menu == "Technique":
            st.subheader("⚙️ Technique (Preprocessing)")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equal = cv2.equalizeHist(gray)
            resize = cv2.resize(img, (300,300))
            gaussian = cv2.GaussianBlur(gray, (5,5), 0)
            median = cv2.medianBlur(gray, 5)

            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Grayscale")
                st.image(equal, caption="Histogram Equalization")
                st.image(gaussian, caption="Gaussian Blur")
            with col2:
                st.image(resize, caption="Resize")
                st.image(median, caption="Median Blur")

        # =========================
        # CONVOLUTION
        # =========================
        elif sub_menu == "Convolution":
            st.subheader("🧮 Convolution")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)

            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])

            sharpen = cv2.filter2D(blur, -1, kernel)

            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, 3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, 3)

            sobelx = cv2.convertScaleAbs(sobelx)
            sobely = cv2.convertScaleAbs(sobely)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(sharpen, caption="Sharpen")
            with col2:
                st.image(sobelx, caption="Sobel X")
            with col3:
                st.image(sobely, caption="Sobel Y")

        # =========================
        # MORPHOLOGY 
        # =========================
        elif sub_menu == "Morphology":
            st.subheader("🔬 Morphology")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)

            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            kernel = np.ones((3,3), np.uint8)

            erosion = cv2.erode(thresh, kernel, iterations=1)
            dilation = cv2.dilate(thresh, kernel, iterations=1)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(thresh, caption="Threshold")
                st.image(erosion, caption="Erosion")
            with col2:
                st.image(dilation, caption="Dilation")
            with col3:
                st.image(opening, caption="Opening")
                st.image(closing, caption="Closing")

        # =========================
        # FACE RECOGNITION
        # =========================
        elif sub_menu == "Feature Detection & Matching":
            st.subheader("Face Recognition")

            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            try:
                result = DeepFace.find(
                    img_path=temp_path,
                    db_path=dataset_path,
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
        
        # =========================
        # UNSUPERVISED LEARNING (K-MEANS) <--- TAMBAHAN: Logika K-Means
        # =========================
        elif sub_menu == "Unsupervised Learning (K-Means)":
            st.subheader("🎨 Unsupervised Learning (K-Means Clustering)")
            st.info("Mengelompokkan warna piksel menggunakan K-Means untuk melakukan segmentasi (memisahkan objek dari latar belakang berdasarkan warna).")
            
            # Slider interaktif di Streamlit untuk menentukan nilai K (jumlah kelompok)
            k = st.slider("Pilih jumlah kelompok warna (K)", min_value=2, max_value=10, value=2)
            
            # Mengubah bentuk gambar menjadi daftar piksel
            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Menerapkan algoritma K-Means
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(pixel_values)
            centers = np.uint8(kmeans.cluster_centers_)
            
            # Rekonstruksi gambar hasil klastering
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(img.shape)
            
            st.image(segmented_image, caption=f"Hasil Segmentasi K-Means (K={k})", width=400)
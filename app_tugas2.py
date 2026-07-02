import os
import json

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(layout="wide")
st.title("🧠 Aplikasi Pemrosesan Citra Wajah")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model_dl")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

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


def _get_persons(base_path):
    return [
        entry for entry in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, entry))
    ]


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    return face


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
        "Face Recognition (Deep Learning CNN)",
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

        st.success(f"✅ Wajah {name} berhasil disimpan! (Total: {file_count + 1} foto)")
        st.image(image, width=250)

    # ==========================================
    # DATASET INFO
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Dataset Saat Ini")

    persons = _get_persons(DATASET_PATH)

    if persons:
        for person in persons:
            count = len(os.listdir(os.path.join(DATASET_PATH, person)))
            st.write(f"👤 **{person}** — {count} foto")
    else:
        st.info("Belum ada wajah yang diregister.")

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
            st.write(f"- **Ukuran:** {img.shape[1]} x {img.shape[0]} piksel")
            st.write(f"- **Channel:** {img.shape[2] if len(img.shape) == 3 else 1}")
            st.write(f"- **Tipe data:** {img.dtype}")

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
                st.image(resized, caption="Resize (300x300)")
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

            sobel_x = cv2.convertScaleAbs(
                cv2.Sobel(blur, cv2.CV_64F, 1, 0, 3)
            )
            sobel_y = cv2.convertScaleAbs(
                cv2.Sobel(blur, cv2.CV_64F, 0, 1, 3)
            )

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
                cv2.THRESH_BINARY_INV, 11, 2
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
        # DEEP LEARNING TRAINING & PREDICTION
        # ==========================================
        elif sub_menu == "Face Recognition (Deep Learning CNN)":
            st.subheader("🤖 Face Recognition dengan Deep Learning (CNN)")

            st.markdown("""
            **Teori CNN (Convolutional Neural Network):**
            - **MobileNetV2 Pretrained**: Model menggunakan arsitektur MobileNetV2 yang telah dilatih pada dataset ImageNet (jutaan gambar). Ini disebut *Transfer Learning* — memanfaatkan pengetahuan dari tugas sebelumnya untuk mempercepat pelatihan pada dataset wajah yang lebih kecil.
            - **Ekstraksi Fitur dengan Convolution Layer**: CNN bekerja dengan menggeser filter (kernel) kecil ke seluruh gambar untuk mendeteksi pola seperti tepi, sudut, tekstur, hingga pola wajah yang lebih kompleks. Semakin dalam layer, fitur yang diekstrak semakin abstrak dan spesifik terhadap wajah.
            - **Klasifikasi dengan Dense Layer**: Setelah fitur diekstrak oleh convolution layer, hasilnya diratakan dan diproses oleh *Dense Layer* (fully connected layer) yang bertugas memetakan fitur-fitur tersebut ke kelas identitas wajah tertentu menggunakan aktivasi softmax.
            - **Face Recognition Pipeline**: Input gambar → preprocessing (resize, normalisasi) → ekstraksi fitur oleh MobileNetV2 → klasifikasi oleh Dense Layer → output probabilitas setiap identitas.
            """)

            # ==========================================
            # TRAIN MODEL
            # ==========================================
            st.markdown("---")
            st.subheader("🏋️ Step 1: Latih Model Deep Learning")

            persons = _get_persons(DATASET_PATH)

            if len(persons) < 2:
                st.warning(
                    "⚠️ Minimal register 2 orang berbeda untuk melatih model."
                )
            else:
                st.info(
                    f"Dataset ditemukan: **{len(persons)} orang** — "
                    f"{', '.join(persons)}"
                )

                col_train1, col_train2 = st.columns(2)
                with col_train1:
                    epochs = st.slider(
                        "Jumlah Epoch",
                        min_value=5, max_value=50, value=15
                    )
                with col_train2:
                    img_size = st.selectbox(
                        "Ukuran Input Gambar", [96, 128, 160], index=1
                    )

                if st.button("🚀 Mulai Training Model"):
                    with st.spinner("⏳ Memuat dan memproses dataset..."):
                        X, y = [], []
                        label_map = {}

                        for idx, person in enumerate(sorted(persons)):
                            label_map[idx] = person
                            person_dir = os.path.join(DATASET_PATH, person)

                            for filename in os.listdir(person_dir):
                                filepath = os.path.join(person_dir, filename)
                                face_img = cv2.imread(filepath)

                                if face_img is not None:
                                    face_img = cv2.cvtColor(
                                        face_img, 
                                        cv2.COLOR_BGR2RGB
                                    )
                                    detected = detect_face(face_img)
                                if detected is not None:
                                    detected = cv2.resize(
                                        detected, 
                                        (img_size, img_size)
                                    )
                                    X.append(detected)
                                    y.append(idx)

                        X = np.array(X, dtype=np.float32) / 255.0
                        y = np.array(y)
                        num_classes = len(persons)

                        with open(
                            os.path.join(MODEL_PATH, "label_map.json"), "w"
                        ) as f:
                            json.dump(label_map, f)

                        with open(
                            os.path.join(MODEL_PATH, "config.json"), "w"
                        ) as f:
                            json.dump({"img_size": img_size}, f)

                        st.success(
                            f"✅ Dataset dimuat: {len(X)} gambar, "
                            f"{num_classes} kelas"
                        )

                    with st.spinner("🧠 Membangun arsitektur MobileNetV2..."):
                        base_model = MobileNetV2(
                            input_shape=(img_size, img_size, 3),
                            include_top=False,
                            weights='imagenet'
                        )

                        # Fine-tune: freeze early layers, train top layers
                        base_model.trainable = True
                        for layer in base_model.layers[:140]:
                            layer.trainable = False

                        x = base_model.output
                        x = GlobalAveragePooling2D()(x)
                        x = Dense(128, activation='relu', name='embedding_layer')(x)
                        x = Dropout(0.5)(x)
                        output = Dense(num_classes, activation='softmax')(x)

                        model = Model(
                            inputs=base_model.input, outputs=output
                        )
                        model.compile(
                            optimizer=Adam(learning_rate=0.0001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        st.success(
                            f"✅ Arsitektur siap — Total parameter: "
                            f"{model.count_params():,}"
                        )

                    with st.expander("📐 Lihat Arsitektur Model"):
                        st.code(f"""
=== ARSITEKTUR DEEP LEARNING ===

[INPUT]
  └─ Gambar {img_size}x{img_size}x3 (RGB)

[BASE MODEL — MobileNetV2 (pretrained ImageNet)]
  └─ 154 layer konvolusional
  └─ 100 layer awal: FROZEN
  └─ ~54 layer akhir: TRAINABLE (fine-tuned)
  └─ Fungsi: Ekstraksi fitur + adaptasi ke dataset wajah

[CUSTOM HEAD — Transfer Learning]
  ├─ GlobalAveragePooling2D
  ├─ Dense(128, activation='relu')  ← embedding_layer
  ├─ Dropout(0.5)
  └─ Dense({num_classes}, activation='softmax')

[OUTPUT]
  └─ Probabilitas untuk {num_classes} kelas: {list(label_map.values())}
                        """)

                    with st.spinner(f"🏋️ Training {epochs} epoch..."):
                        if len(X) > 10:
                            X_train, X_val, y_train, y_val = train_test_split(
                                X, y,
                                test_size=0.2,
                                random_state=42,
                                stratify=y
                            )
                            validation_data = (X_val, y_val)
                        else:
                            X_train, y_train = X, y
                            validation_data = None

                        # Data augmentation untuk stabilitas dan generalisasi
                        datagen = ImageDataGenerator(
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            fill_mode='nearest'
                        )
                        datagen.fit(X_train)

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        history_log = []

                        class StreamlitCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)

                                acc = logs.get('accuracy', 0)
                                val_acc = logs.get('val_accuracy', 0)
                                loss = logs.get('loss', 0)
                                lr_val = float(tf.keras.backend.get_value(
                                    model.optimizer.learning_rate
                                ))

                                status_text.text(
                                    f"Epoch {epoch + 1}/{epochs} — "
                                    f"Loss: {loss:.4f} | "
                                    f"Acc: {acc:.4f} | "
                                    f"Val Acc: {val_acc:.4f} | "
                                    f"LR: {lr_val:.2e}"
                                )

                                history_log.append({
                                    "epoch": epoch + 1,
                                    "accuracy": float(acc),
                                    "val_accuracy": float(val_acc),
                                    "loss": float(loss)
                                })

                        callbacks_list = [StreamlitCallback()]
                        if validation_data:
                            callbacks_list.append(
                                ReduceLROnPlateau(
                                    monitor='val_loss', factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=0
                                )
                            )

                        model.fit(
                            datagen.flow(
                                X_train, y_train,
                                batch_size=min(16, len(X_train))
                            ),
                            epochs=epochs,
                            validation_data=validation_data,
                            callbacks=callbacks_list,
                            verbose=0
                        )

                        model.save(os.path.join(MODEL_PATH, "face_model.h5"))

                        # Simpan embeddings untuk unknown face detection
                        extractor = Model(
                            inputs=model.input,
                            outputs=model.get_layer('embedding_layer').output
                        )
                        all_embeddings = extractor.predict(X, verbose=0)
                        np.save(
                            os.path.join(MODEL_PATH, "embeddings.npy"),
                            all_embeddings
                        )
                        np.save(
                            os.path.join(MODEL_PATH, "labels.npy"), y
                        )

                        progress_bar.progress(1.0)

                    st.success("✅ Model berhasil dilatih dan disimpan!")

                    final_metrics = history_log[-1]
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "🎯 Akurasi Training",
                        f"{final_metrics['accuracy'] * 100:.1f}%"
                    )
                    if validation_data:
                        col2.metric(
                            "🔍 Akurasi Validasi",
                            f"{final_metrics['val_accuracy'] * 100:.1f}%"
                        )
                    col3.metric(
                        "📉 Loss Akhir",
                        f"{final_metrics['loss']:.4f}"
                    )

            # ==========================================
            # PREDICT
            # ==========================================
            st.markdown("---")
            st.subheader("🔍 Step 2: Prediksi Wajah")

            model_file = os.path.join(MODEL_PATH, "face_model.h5")
            label_file = os.path.join(MODEL_PATH, "label_map.json")
            config_file = os.path.join(MODEL_PATH, "config.json")

            if not os.path.exists(model_file):
                st.warning(
                    "⚠️ Model belum dilatih. Lakukan training terlebih "
                    "dahulu di Step 1."
                )
            else:
                if st.button("🔮 Prediksi Siapa Orangnya"):
                    with st.spinner(
                        "Memuat model dan melakukan prediksi..."
                    ):
                        model = tf.keras.models.load_model(model_file)

                        with open(label_file) as f:
                            label_map = json.load(f)
                        with open(config_file) as f:
                            config = json.load(f)

                        img_size = config["img_size"]

                        face_crop = detect_face(img)

                        if face_crop is None:
                            st.error("❌ Wajah tidak ditemukan pada gambar")
                            st.stop()

                        face_input = cv2.resize(
                            face_crop,
                            (img_size, img_size)
                        )
                        face_input = np.expand_dims(
                            face_input.astype(np.float32) / 255.0, axis=0
                        )

                        predictions = model.predict(face_input, verbose=0)[0]
                        pred_idx = int(np.argmax(predictions))
                        confidence = float(predictions[pred_idx])

                        sorted_pred = np.sort(predictions)
                        top2 = sorted_pred[-2]
                        margin = confidence - top2

                    st.markdown("### 📊 Hasil Prediksi Deep Learning")

                    CONF_THRESH = 0.40
                    MARGIN_THRESH = 0.10

                    if confidence < CONF_THRESH or margin < MARGIN_THRESH:
                        st.error("❌ Wajah tidak dikenali / model tidak yakin")
                    else:
                        st.success(
                            f"✅ **{label_map[str(pred_idx)]}** — "
                            f"Keyakinan: {confidence:.1%}"
                        )

                        st.markdown("**Distribusi Probabilitas:**")
                        for i, prob in enumerate(predictions):
                            name_label = label_map[str(i)]
                            bar_color = "🟢" if i == pred_idx else "⬜"
                            st.write(
                                f"{bar_color} **{name_label}**: "
                                f"{prob * 100:.1f}%"
                            )
                            st.progress(float(prob))

        # ==========================================
        # K-MEANS
        # ==========================================
        elif sub_menu == "Unsupervised Learning (K-Means)":
            st.subheader("🎨 Unsupervised Learning (K-Means Clustering)")

            st.info(
                "Mengelompokkan warna piksel menggunakan K-Means "
                "untuk segmentasi warna."
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

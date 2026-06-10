import streamlit as st
import cv2
import numpy as np
import os
import json
import pickle
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("🧠 Aplikasi Pemrosesan Citra Wajah")

dataset_path = "dataset"
model_path = "model_dl"
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

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
        "Feature Detection & Matching (Deep Learning)",
        "Unsupervised Learning (K-Means)"
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

        st.success(f"✅ Wajah {name} berhasil disimpan! (Total: {file_count+1} foto)")
        st.image(image, width=250)

    # Info jumlah data per orang
    st.markdown("---")
    st.subheader("📊 Dataset Saat Ini")
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if persons:
        for p in persons:
            count = len(os.listdir(os.path.join(dataset_path, p)))
            st.write(f"👤 **{p}** — {count} foto")
    else:
        st.info("Belum ada wajah yang diregister.")

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
            st.write(f"- **Ukuran:** {img.shape[1]} x {img.shape[0]} piksel")
            st.write(f"- **Channel:** {img.shape[2] if len(img.shape) == 3 else 1}")
            st.write(f"- **Tipe data:** {img.dtype}")

        # =========================
        # TECHNIQUE
        # =========================
        elif sub_menu == "Technique":
            st.subheader("⚙️ Technique (Preprocessing)")

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            equal = cv2.equalizeHist(gray)
            resize = cv2.resize(img, (300, 300))
            gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
            median = cv2.medianBlur(gray, 5)

            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Grayscale")
                st.image(equal, caption="Histogram Equalization")
                st.image(gaussian, caption="Gaussian Blur")
            with col2:
                st.image(resize, caption="Resize (300x300)")
                st.image(median, caption="Median Blur")

        # =========================
        # CONVOLUTION
        # =========================
        elif sub_menu == "Convolution":
            st.subheader("🧮 Convolution")

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
            sharpen = cv2.filter2D(blur, -1, kernel)
            sobelx = cv2.convertScaleAbs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, 3))
            sobely = cv2.convertScaleAbs(cv2.Sobel(blur, cv2.CV_64F, 0, 1, 3))

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

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
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
        # FEATURE DETECTION - DEEP LEARNING
        # =========================
        elif sub_menu == "Feature Detection & Matching (Deep Learning)":
            st.subheader("🤖 Face Recognition dengan Deep Learning (MobileNetV2)")

            st.markdown("""
            **Cara Kerja:**
            1. Dataset wajah yang sudah diregister digunakan untuk melatih model CNN (MobileNetV2)
            2. MobileNetV2 adalah arsitektur deep learning yang sudah pretrained di ImageNet
            3. Model di-fine-tune khusus untuk mengenali wajah di dataset kamu (Transfer Learning)
            4. Prediksi dilakukan menggunakan model yang sudah dilatih
            """)

            # --------------------------------------------------
            # SECTION 1: TRAIN MODEL
            # --------------------------------------------------
            st.markdown("---")
            st.subheader("🏋️ Step 1: Latih Model Deep Learning")

            persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

            if len(persons) < 2:
                st.warning("⚠️ Minimal register 2 orang berbeda untuk melatih model.")
            else:
                st.info(f"Dataset ditemukan: **{len(persons)} orang** — {', '.join(persons)}")

                col_train1, col_train2 = st.columns(2)
                with col_train1:
                    epochs = st.slider("Jumlah Epoch", min_value=5, max_value=50, value=10)
                with col_train2:
                    img_size = st.selectbox("Ukuran Input Gambar", [96, 128, 160], index=1)

                if st.button("🚀 Mulai Training Model"):
                    import tensorflow as tf
                    from tensorflow.keras.applications import MobileNetV2
                    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
                    from tensorflow.keras.models import Model
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator
                    from tensorflow.keras.optimizers import Adam

                    with st.spinner("⏳ Memuat dan memproses dataset..."):
                        # Load dataset
                        X, y = [], []
                        label_map = {}

                        for idx, person in enumerate(sorted(persons)):
                            label_map[idx] = person
                            person_dir = os.path.join(dataset_path, person)
                            for fname in os.listdir(person_dir):
                                fpath = os.path.join(person_dir, fname)
                                face_img = cv2.imread(fpath)
                                if face_img is not None:
                                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                    face_img = cv2.resize(face_img, (img_size, img_size))
                                    X.append(face_img)
                                    y.append(idx)

                        X = np.array(X, dtype=np.float32) / 255.0
                        y = np.array(y)
                        num_classes = len(persons)

                        # Simpan label map
                        with open(os.path.join(model_path, "label_map.json"), "w") as f:
                            json.dump(label_map, f)
                        with open(os.path.join(model_path, "config.json"), "w") as f:
                            json.dump({"img_size": img_size}, f)

                        st.success(f"✅ Dataset dimuat: {len(X)} gambar, {num_classes} kelas")

                    with st.spinner("🧠 Membangun arsitektur MobileNetV2..."):
                        # ==============================
                        # ARSITEKTUR DEEP LEARNING
                        # ==============================
                        # Base model: MobileNetV2 (pretrained ImageNet)
                        base_model = MobileNetV2(
                            input_shape=(img_size, img_size, 3),
                            include_top=False,       # Hapus fully connected layer asli
                            weights='imagenet'       # Gunakan bobot pretrained
                        )

                        # Freeze layer base (tidak dilatih ulang)
                        base_model.trainable = False

                        # Tambahkan custom classification head
                        x = base_model.output
                        x = GlobalAveragePooling2D()(x)      # Pooling global
                        x = Dense(128, activation='relu')(x) # Fully connected
                        x = Dropout(0.7)(x)                  # Regularisasi
                        output = Dense(num_classes, activation='softmax')(x)  # Output

                        model = Model(inputs=base_model.input, outputs=output)
                        model.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )

                        st.success(f"✅ Arsitektur siap — Total parameter: {model.count_params():,}")

                    # Tampilkan ringkasan arsitektur
                    with st.expander("📐 Lihat Arsitektur Model"):
                        st.code(f"""
=== ARSITEKTUR DEEP LEARNING ===

[INPUT]
  └─ Gambar {img_size}x{img_size}x3 (RGB)

[BASE MODEL — MobileNetV2 (pretrained ImageNet)]
  └─ 154 layer konvolusional
  └─ Bobot: FROZEN (tidak dilatih ulang)
  └─ Fungsi: Ekstraksi fitur gambar

[CUSTOM HEAD — Transfer Learning]
  ├─ GlobalAveragePooling2D
  ├─ Dense(128, activation='relu')
  ├─ Dropout(0.7)
  └─ Dense({num_classes}, activation='softmax')

[OUTPUT]
  └─ Probabilitas untuk {num_classes} kelas: {list(label_map.values())}
                        """)

                    with st.spinner(f"🏋️ Training {epochs} epoch..."):
                        # Data augmentation untuk training
                        from sklearn.model_selection import train_test_split

                        if len(X) > 10:
                            X_train, X_val, y_train, y_val = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            validation_data = (X_val, y_val)
                        else:
                            X_train, y_train = X, y
                            validation_data = None

                        # Progress bar training
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
                                status_text.text(
                                    f"Epoch {epoch+1}/{epochs} — "
                                    f"Loss: {loss:.4f} | "
                                    f"Acc: {acc:.4f} | "
                                    f"Val Acc: {val_acc:.4f}"
                                )
                                history_log.append({
                                    "epoch": epoch+1,
                                    "accuracy": float(acc),
                                    "val_accuracy": float(val_acc),
                                    "loss": float(loss)
                                })

                        model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=min(16, len(X_train)),
                            validation_data=validation_data,
                            callbacks=[StreamlitCallback()],
                            verbose=0
                        )

                        # Simpan model
                        model.save(os.path.join(model_path, "face_model.h5"))
                        progress_bar.progress(1.0)

                    st.success("✅ Model berhasil dilatih dan disimpan!")

                    # Tampilkan hasil training
                    final = history_log[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🎯 Akurasi Training", f"{final['accuracy']*100:.1f}%")
                    if validation_data:
                        c2.metric("🔍 Akurasi Validasi", f"{final['val_accuracy']*100:.1f}%")
                    c3.metric("📉 Loss Akhir", f"{final['loss']:.4f}")

            # --------------------------------------------------
            # SECTION 2: PREDICT
            # --------------------------------------------------
            st.markdown("---")
            st.subheader("🔍 Step 2: Prediksi Wajah")

            model_file = os.path.join(model_path, "face_model.h5")
            label_file = os.path.join(model_path, "label_map.json")
            config_file = os.path.join(model_path, "config.json")

            if not os.path.exists(model_file):
                st.warning("⚠️ Model belum dilatih. Lakukan training terlebih dahulu di Step 1.")
            else:
                if st.button("🔮 Prediksi Siapa Orangnya"):
                    import tensorflow as tf

                    with st.spinner("Memuat model dan melakukan prediksi..."):
                        model = tf.keras.models.load_model(model_file)

                        with open(label_file) as f:
                            label_map = json.load(f)
                        with open(config_file) as f:
                            config = json.load(f)

                        img_size = config["img_size"]

                        # Preprocess input image
                        face_input = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                        face_input = cv2.resize(face_input, (img_size, img_size))
                        face_input = np.expand_dims(face_input.astype(np.float32) / 255.0, axis=0)

                        # Prediksi
                        predictions = model.predict(face_input)[0]
                        pred_idx = int(np.argmax(predictions))
                        confidence = float(predictions[pred_idx]) * 100
                        pred_name = label_map[str(pred_idx)]

                    st.markdown("### 📊 Hasil Prediksi Deep Learning")
                    if confidence >= 60:
                        st.success(f"✅ **{pred_name}** — Keyakinan: {confidence:.1f}%")
                    else:
                        st.warning(f"⚠️ **{pred_name}** — Keyakinan rendah: {confidence:.1f}% (wajah mungkin tidak dikenal)")

                    # Tampilkan probabilitas semua kelas
                    st.markdown("**Distribusi Probabilitas:**")
                    for i, prob in enumerate(predictions):
                        name_label = label_map[str(i)]
                        bar_color = "🟢" if i == pred_idx else "⬜"
                        st.write(f"{bar_color} **{name_label}**: {prob*100:.1f}%")
                        st.progress(float(prob))

        # =========================
        # K-MEANS
        # =========================
        elif sub_menu == "Unsupervised Learning (K-Means)":
            st.subheader("🎨 Unsupervised Learning (K-Means Clustering)")
            st.info("Mengelompokkan warna piksel menggunakan K-Means untuk segmentasi warna.")

            k = st.slider("Pilih jumlah kelompok warna (K)", min_value=2, max_value=10, value=2)

            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(pixel_values)
            centers = np.uint8(kmeans.cluster_centers_)

            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(img.shape)

            st.image(segmented_image, caption=f"Hasil Segmentasi K-Means (K={k})", width=400)
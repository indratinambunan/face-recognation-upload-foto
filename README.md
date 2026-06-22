# Face Recognition Upload Foto

Aplikasi berbasis **Streamlit** untuk pengolahan citra digital dan pengenalan wajah (*Face Recognition*) menggunakan teknik **Computer Vision** dan **Deep Learning**.

Project ini dibuat sebagai implementasi tugas mata kuliah **Computer Vision**, yang mencakup preprocessing citra, operasi konvolusi, morfologi, segmentasi menggunakan **K-Means**, serta pengenalan wajah menggunakan arsitektur **MobileNetV2 (Transfer Learning)**.

---

## ✨ Fitur Aplikasi

### 📌 Tugas 1 — Image Processing

Modul pengolahan citra digital dasar yang terdiri dari:

* Register dataset wajah
* Data Input (informasi gambar)
* Technique (Preprocessing)

  * Grayscale
  * Histogram Equalization
  * Gaussian Blur
  * Median Blur
  * Resize
* Convolution

  * Sharpen Filter
  * Sobel X
  * Sobel Y
* Morphology

  * Thresholding
  * Erosion
  * Dilation
  * Opening
  * Closing
* Unsupervised Learning

  * K-Means Clustering (segmentasi warna)

---

### 📌 Tugas 2 — Deep Learning Face Recognition

Implementasi pengenalan wajah berbasis Deep Learning:

* Dataset wajah multi-kelas
* Training model CNN menggunakan MobileNetV2
* Transfer Learning (Pretrained ImageNet)
* Fine Tuning model
* Face Prediction
* Confidence Score Prediction
* Visualisasi probabilitas hasil prediksi

---

## 📦 Library yang Dibutuhkan

Install dependencies berikut:

* streamlit
* opencv-python
* numpy
* pillow
* tensorflow
* scikit-learn

Install dengan command:

```bash
python -m pip install streamlit opencv-python numpy pillow tensorflow scikit-learn
```

---

## 🚀 Cara Menjalankan Program

Clone repository:

```bash
git clone https://github.com/indratinambunan/face-recognation-upload-foto.git
```

Masuk ke folder project:

```bash
cd face-recognation-upload-foto
```

Install dependencies:

```bash
python -m pip install streamlit opencv-python numpy pillow tensorflow scikit-learn
```

Menjalankan program tugas 1:

```bash
python -m streamlit run app_tugas1.py
```

Menjalankan program tugas 2:

```bash
python -m streamlit run app_tugas2.py
```

---

## 🧠 Teknologi yang Digunakan

* Python 3
* Streamlit
* OpenCV
* NumPy
* TensorFlow / Keras
* MobileNetV2
* Scikit-Learn (K-Means Clustering)

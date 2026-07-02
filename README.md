# Face Recognition Upload Foto

Aplikasi berbasis **Streamlit** untuk implementasi **Computer Vision** dan **Deep Learning** dalam pengolahan citra digital serta pengenalan wajah (*Face Recognition*).

Project ini dibuat sebagai tugas mata kuliah **Computer Vision** yang mencakup implementasi pengolahan citra digital, feature detection, segmentasi, serta pengenalan wajah berbasis Deep Learning menggunakan **MobileNetV2 (Transfer Learning)**.

---

## ✨ Fitur Program

### 📌 Tugas 1 — Image Processing
- Register dataset wajah  
- Grayscale  
- Histogram Equalization  
- Gaussian Blur  
- Median Blur  
- Resize  
- Convolution (Sharpen, Sobel X, Sobel Y)  
- Morphology (Thresholding, Erosion, Dilation, Opening, Closing)  
- Feature Detection menggunakan Hough Transform  
- K-Means Clustering  

### 📌 Tugas 2 — Deep Learning Face Recognition
- Dataset multi-kelas wajah  
- Training model menggunakan arsitektur CNN MobileNetV2  
- Transfer Learning menggunakan MobileNetV2  
- Fine Tuning model  
- Face Prediction  
- Confidence Score Prediction  
- Visualisasi probabilitas hasil prediksi  

---

## 📦 Instalasi dan Menjalankan Program

Clone repository:

```bash
git clone https://github.com/indratinambunan/face-recognation-upload-foto.git
```

Masuk ke folder project:

```bash
cd face-recognation-upload-foto
```

Install semua library yang dibutuhkan  
*(pastikan Python sudah terinstall)*

```bash
python -m pip install streamlit opencv-python numpy pillow tensorflow scikit-learn
```

Menjalankan program:

Untuk **Tugas 1 (Image Processing)**

```bash
python -m streamlit run app_tugas1.py
```

Untuk **Tugas 2 (Deep Learning Face Recognition)**

```bash
python -m streamlit run app_tugas2.py
```

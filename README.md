# [Ongoing...]Optimasi Termogram Telapak Kaki Untuk Deteksi Dini Ulkus Kaki Diabetik: Pendekatan Peningkatan Citra

## Deskripsi Proyek

Proyek ini mengembangkan model Machine Learning untuk deteksi dini ulkus kaki diabetes (Diabetic Foot Ulcer/DFU) menggunakan citra termogram dan data suhu telapak kaki. Model ini menggabungkan Convolutional Neural Network (CNN) untuk memproses citra termogram dan Multi-Layer Perceptron (MLP) untuk data suhu, membentuk Multi-Classifier yang efektif dalam prediksi DFU.

Proyek ini juga menerapkan berbagai teknik image enhancement seperti Solarize, Posterize, CLAHE, dan Gamma Adjustment untuk meningkatkan kualitas citra dan performa model. Evaluasi model dilakukan menggunakan metrik seperti akurasi, presisi, recall, F1-score, dan ROC-AUC.

Pendekatan ini bertujuan memberikan kontribusi signifikan dalam pengembangan teknologi citra medis untuk diagnosis penyakit, khususnya dalam deteksi dini ulkus kaki pada penderita diabetes.

## Struktur Proyek

```
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── model_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   ├── mlp_model.py
│   │   └── multi_classifier.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_enhancement.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   ├── training.py
│   ├── evaluation.py
│   └── predict.py
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── predict.sh
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_utils.py
├── configs/
│   ├── config.yaml
│   └── logging.conf
├── logs/
│   ├── training.log
│   └── evaluation.log
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Fitur Utama

- **Penggunaan Citra Termogram**: Memanfaatkan citra termal telapak kaki untuk mendeteksi tanda-tanda awal ulkus kaki.
- **Data Suhu Telapak Kaki**: Menggunakan data suhu rinci sebagai input tambahan untuk meningkatkan akurasi prediksi.
- **Teknik Image Enhancement**: Menerapkan Solarize, Posterize, CLAHE, dan Gamma Adjustment untuk meningkatkan kualitas citra.
- **Model Multi-Classifier**: Kombinasi CNN dan MLP untuk memproses data citra dan tabular secara bersamaan.
- **Evaluasi Komprehensif**: Menggunakan metrik akurasi, presisi, recall, F1-score, dan ROC-AUC untuk menilai performa model.

## Persyaratan Sistem

- Python 3.9 (CUDA Available)
- Paket-paket yang tercantum dalam requirements.txt

## Instalasi

1. Clone repositori ini

```bash
git clone https://github.com/username/diabetic-foot-ulcer-detection.git
cd diabetic-foot-ulcer-detection
```

2. Buat virtual environment (opsional tetapi sangat direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
venv\Scripts\activate  # Untuk Windows
```

3. Install dependensi

```bash
pip install -r requirements.txt
```

4. Konfigurasi proyek

- Sesuaikan file configs/config.yaml sesuai dengan jalur data dan parameter yang diinginkan.
- Atur konfigurasi logging pada configs/logging.conf jika diperlukan.

## Struktur Direktori Data

```
data/
├── raw/               # Data mentah (citra termogram dan data suhu)
└── processed/         # Data setelah pra-pemrosesan
```

## Cara Penggunaan

### 1. Pra-pemrosesan Data

Lakukan pra-pemrosesan data mentah dan simpan hasilnya ke direktori data/processed/.

```bash
python src/data/data_preprocessing.py
```

### 2. Melatih Model

Untuk melatih model, jalankan skrip train.sh atau gunakan perintah berikut:

```bash
python src/training.py --config configs/config.yaml
```

Parameter seperti jalur data, hyperparameter model, dan pengaturan lainnya dapat disesuaikan melalui file config.yaml.

### 3. Evaluasi Model

Setelah model dilatih, evaluasi performanya dengan:

```bash
python src/evaluation.py --config configs/config.yaml
```

Hasil evaluasi akan disimpan dalam direktori logs/ dan ditampilkan di konsol.

### 4. Prediksi dengan Data Baru

Untuk melakukan prediksi pada data baru

```bash
python src/prediction.py --input_path path/to/new/data --output_path path/to/save/predictions
```

### 5. Menjalankan Unit Test

Untuk memastikan semua modul bekerja dengan baik, jalankan unit test:

```bash
python -m unittest discover -s tests
```

## Kontak

Untuk pertanyaan atau dukungan lebih lanjut, silakan hubungi:

Email: [muhammadnurilhuda@mail.ugm.ac.id](mailto:muhammadnurilhuda@mail.ugm.ac.id)

# Penjelasan Tambahan

Dalam proyek ini, kami menekankan pentingnya deteksi dini ulkus kaki pada penderita diabetes untuk mencegah komplikasi serius seperti infeksi dan amputasi. Dengan memanfaatkan citra termogram dan data suhu, kami dapat mendeteksi perubahan suhu yang mengindikasikan perkembangan ulkus.

Teknik Image Enhancement yang diterapkan membantu meningkatkan kualitas citra sehingga fitur penting dapat diekstraksi oleh model secara lebih efektif. Khususnya, metode Solarize dengan threshold 128 memberikan peningkatan akurasi yang signifikan.

Model Multi-Classifier kami dirancang untuk memproses dan mengintegrasikan informasi dari citra dan data tabular secara simultan, memberikan prediksi yang lebih akurat dan andal.

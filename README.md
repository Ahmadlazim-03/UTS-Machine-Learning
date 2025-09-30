# Cervical Cancer Risk Factors Data Preprocessing

Proyek ini adalah toolkit untuk preprocessing data faktor risiko kanker serviks yang mencakup 4 tahapan utama:

1. **Missing Value Analysis** - Deteksi dan penanganan nilai yang hilang
2. **Data Transformation** - Normalisasi data menggunakan MinMaxScaler
3. **Feature Extraction** - Ekstraksi fitur menggunakan PCA (Principal Component Analysis)
4. **Imbalanced Data Handling** - Penanganan data tidak seimbang menggunakan Random Over Sampling (ROS)

## 📁 Struktur Direktori

```
UTS-Machine-Learning/
├── risk_factors_cervical_cancer.csv    # Dataset utama
├── main_preprocessor.py                 # Script utama dengan menu interaktif
├── requirements.txt                     # Dependencies Python
├── README.md                           # Dokumentasi
├── MissingValue/
│   └── missing_value.py                # Modul penanganan missing value
├── Transformasi/
│   └── transformasi.py                 # Modul transformasi data (MinMaxScaler)
├── EkstraksiFitur/
│   └── ektraksi_fitur.py              # Modul ekstraksi fitur (PCA)
└── ImbalancedData/
    └── imbalanced_data.py             # Modul penanganan data tidak seimbang (ROS)
```

## 🚀 Cara Menggunakan

### 1. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### 2. Menjalankan Program

```bash
cd UTS-Machine-Learning
python main_preprocessor.py
```

### 3. Menu Interaktif

Program akan menampilkan menu interaktif dengan pilihan:

```
🏥 CERVICAL CANCER DATA PREPROCESSING TOOLKIT
============================================================
Choose an option:
1. Run Complete Pipeline (Terminal Output)
2. Run Complete Pipeline (Web/Visual Output)
3. Run Individual Steps
4. Custom Configuration
5. Exit
```

#### Pilihan 1: Complete Pipeline (Terminal)
- Menjalankan semua 4 tahap preprocessing secara berurutan
- Output ditampilkan di terminal/console
- Otomatis menyimpan hasil ke folder `output/`

#### Pilihan 2: Complete Pipeline (Web/Visual)
- Menjalankan semua 4 tahap preprocessing secara berurutan
- Menghasilkan visualisasi/plot untuk setiap tahap
- Menyimpan plot sebagai file PNG

#### Pilihan 3: Individual Steps
- Memungkinkan menjalankan tahap-tahap secara individual
- Pilihan output: terminal atau web
- Submen:
  1. Missing Value Analysis
  2. Data Transformation (MinMaxScaler)
  3. Feature Extraction (PCA)
  4. Imbalanced Data Handling (ROS)

#### Pilihan 4: Custom Configuration
- Konfigurasi custom untuk setiap tahap:
  - **Missing Value Strategy**: median, mean, mode, drop
  - **Scaler Type**: minmax, standard, robust
  - **Resampling Method**: ros, smote, adasyn

## 📊 Output Files

Setelah preprocessing, file-file berikut akan dibuat di folder `output/`:

- `final_processed_data.csv` - Data final setelah semua preprocessing
- `data_after_missing_value_handling.csv` - Data setelah menangani missing values
- `data_after_transformation.csv` - Data setelah transformasi MinMaxScaler
- `data_after_feature_extraction.csv` - Data setelah ekstraksi fitur PCA

## 🔧 Tahapan Preprocessing

### 1. Missing Value Analysis (`MissingValue/missing_value.py`)

**Fitur:**
- Deteksi missing values (termasuk nilai '?')
- Visualisasi pola missing values
- Strategi penanganan: median, mean, mode, atau drop
- Summary statistik sebelum dan sesudah penanganan

**Output Terminal:**
```
MISSING VALUES ANALYSIS
============================================================
Dataset Shape: (858, 36)
Total Missing Values: 3622

Missing Values by Column:
----------------------------------------------------------
STDs: Time since first diagnosis    |   765 |  89.21%
STDs: Time since last diagnosis     |   765 |  89.21%
...
```

### 2. Data Transformation (`Transformasi/transformasi.py`)

**Fitur:**
- Identifikasi otomatis fitur dan target
- Multiple scaler options: MinMaxScaler, StandardScaler, RobustScaler
- Analisis distribusi data sebelum dan sesudah transformasi
- Visualisasi perbandingan distribusi

**Output Terminal:**
```
DATA TRANSFORMATION SUMMARY
============================================================
Scaler used: MinMaxScaler
Features transformed: 30
Data shape: (858, 36)
Feature range after transformation: 0.0000 to 1.0000
```

### 3. Feature Extraction (`EkstraksiFitur/ektraksi_fitur.py`)

**Fitur:**
- Principal Component Analysis (PCA)
- Analisis feature importance
- Penentuan otomatis jumlah komponen berdasarkan variance threshold
- Visualisasi explained variance dan loadings
- Feature loadings untuk interpretasi

**Output Terminal:**
```
PCA ANALYSIS RESULTS
============================================================
Number of components: 15
Original features: 30
Reduced features: 15
Dimensionality reduction: 30 → 15

Explained Variance by Component:
--------------------------------------------------
PC1  |   0.2156 | Cumulative:   0.2156
PC2  |   0.1342 | Cumulative:   0.3498
...
```

### 4. Imbalanced Data Handling (`ImbalancedData/imbalanced_data.py`)

**Fitur:**
- Analisis distribusi kelas
- Multiple resampling methods: ROS, SMOTE, ADASYN
- Visualisasi perbandingan distribusi sebelum dan sesudah resampling
- Perhitungan imbalance ratio

**Output Terminal:**
```
CLASS DISTRIBUTION ANALYSIS - Biopsy
============================================================
Total samples: 858
Class 0:    804 samples ( 93.71%)
Class 1:     54 samples (  6.29%)

Imbalance Ratio: 14.89:1
⚠️  Dataset is imbalanced - resampling recommended
```

## 📈 Visualisasi (Web Output)

Saat memilih output "web", program akan menghasilkan plot-plot berikut:

1. **Missing Values**: Heatmap dan bar chart missing values
2. **Data Transformation**: Histogram distribusi sebelum/sesudah transformasi
3. **PCA**: Explained variance, cumulative variance, scatter plot PC1-PC2, loadings heatmap
4. **Imbalanced Data**: Bar chart dan pie chart distribusi kelas

## 🎯 Contoh Penggunaan

### Menjalankan Pipeline Lengkap dengan Output Terminal:

```python
from main_preprocessor import CervicalCancerPreprocessor

# Initialize
preprocessor = CervicalCancerPreprocessor("risk_factors_cervical_cancer.csv")

# Run complete pipeline
final_data = preprocessor.run_complete_pipeline(
    output_type='terminal',
    mv_strategy='median',
    scaler_type='minmax',
    variance_threshold=0.95,
    resampling_method='ros'
)
```

### Menjalankan Tahap Individual:

```python
# Hanya missing value analysis
preprocessor.run_missing_value_analysis(output_type='terminal', strategy='median')

# Hanya transformasi data
preprocessor.run_data_transformation(output_type='web', scaler_type='minmax')
```

## 📝 Catatan Penting

1. **Dataset**: Pastikan file `risk_factors_cervical_cancer.csv` ada di direktori yang sama
2. **Dependencies**: Install semua package yang diperlukan dari `requirements.txt`
3. **Memory**: Untuk dataset besar, pertimbangkan menggunakan output terminal untuk menghemat memory
4. **Visualisasi**: Plot akan disimpan sebagai file PNG di direktori kerja

## 🔍 Troubleshooting

**Error: "Data file not found"**
- Pastikan file CSV ada di direktori yang benar
- Periksa nama file (case sensitive)

**Error: Import module**
- Pastikan semua dependencies terinstall
- Jalankan `pip install -r requirements.txt`

**Memory Error pada visualisasi**
- Gunakan output terminal instead of web
- Kurangi jumlah fitur yang divisualisasikan

## 👥 Kontribusi

Proyek ini dibuat untuk keperluan pembelajaran preprocessing data pada dataset medical/healthcare dengan fokus pada:

- Penanganan missing values pada data medical
- Normalisasi fitur dengan range yang berbeda-beda  
- Dimensionality reduction untuk dataset dengan banyak fitur
- Handling imbalanced classes yang umum pada data medical

---

**Developed by: [Nama Anda]**  
**Course: Machine Learning - UTS Project**  
**Dataset: Cervical Cancer Risk Factors**
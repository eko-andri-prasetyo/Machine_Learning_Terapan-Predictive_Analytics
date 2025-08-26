# Submission 1: California Housing Price Prediction

Nama: Eko Andri Prasetyo

Username dicoding: ekoandriprasetyo

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-pricing) |
| Masalah | Prediksi harga rumah median di California berdasarkan atribut demografi dan geografis |
| Solusi machine learning | Membangun model regresi untuk memprediksi harga rumah dengan akurasi tinggi |
| Metode pengolahan | Feature engineering, handling missing values, encoding, scaling, dan data splitting |
| Arsitektur model | Linear Regression, Random Forest, XGBoost, Neural Network |
| Metrik evaluasi | MAE, MSE, RMSE, R² |
| Performa model | XGBoost tuned mencapai R² score 0.8419 |

## 1. Domain Proyek

Proyek ini berada dalam domain real estate dan perumahan, dengan fokus pada prediksi harga properti di California. Prediksi harga rumah yang akurat memiliki implikasi signifikan bagi berbagai pemangku kepentingan termasuk pembeli rumah, developer properti, dan pemerintah dalam perencanaan kebijakan perumahan.

## 2. Business Understanding

### 2.1. Latar Belakang Masalah
Pasar properti California merupakan salah yang paling dinamis dan kompleks di Amerika Serikat. Ketepatan dalam memprediksi harga rumah sangat penting untuk pengambilan keputusan investasi, penentuan harga jual yang kompetitif, dan perencanaan kebijakan perumahan yang efektif.

### 2.2. Pernyataan Masalah
Kesulitan dalam memprediksi harga rumah secara akurat berdasarkan karakteristik demografi dan geografis yang tersedia.

### 2.3. Tujuan Proyek
1. Membangun model prediktif dengan akurasi tinggi (R² > 0.8)
2. Mengidentifikasi faktor-faktor paling berpengaruh terhadap harga rumah
3. Membandingkan performa berbagai algoritma machine learning
4. Mengoptimalkan model terbaik melalui hyperparameter tuning

### 2.4. Manfaat Solusi
- **Bagi Pembeli Rumah**: Membantu dalam membuat keputusan investasi yang tepat
- **Bagi Developer Properti**: Menentukan harga jual yang kompetitif
- **Bagi Pemerintah**: Perencanaan kebijakan perumahan dan pengembangan wilayah

## 3. Data Understanding

### 3.1. Sumber Data
Dataset diperoleh dari [California Housing Prices di Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-pricing)

### 3.2. Karakteristik Dataset
- **Jumlah Data**: 20.640 sampel (baris) dan 10 fitur (kolom)
- **Kondisi Data Awal**:
  - **Missing Values**: 267 nilai yang hilang pada kolom `total_bedrooms`
  - **Duplikat**: Tidak terdapat data duplikat (0 duplicate)
  - **Outlier**: Terdeteksi outlier pada beberapa fitur numerik berdasarkan analisis boxplot

### 3.3. Uraian Fitur
1. `longitude`: Koordinat bujur rumah (numerik)
2. `latitude`: Koordinat lintang rumah (numerik)
3. `housing_median_age`: Usia median rumah dalam blok (numerik)
4. `total_rooms`: Total jumlah kamar dalam blok (numerik)
5. `total_bedrooms`: Total jumlah kamar tidur dalam blok (numerik)
6. `population`: Total populasi dalam blok (numerik)
7. `households`: Total jumlah rumah tangga dalam blok (numerik)
8. `median_income`: Pendapatan median rumah tangga (skala puluhan ribu dolar) (numerik)
9. `ocean_proximity`: Kedekatan dengan laut (kategorikal: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND)
10. `median_house_value`: Harga rumah median (target variable) (numerik)

### 3.4. Eksplorasi Data
Analisis korelasi menunjukkan bahwa `median_income` memiliki korelasi tertinggi dengan target variable (0.688), diikuti oleh `total_rooms` (0.134) dan `housing_median_age` (0.166). Variabel kategorikal `ocean_proximity` memiliki distribusi yang tidak merata dengan mayoritas data berada dalam kategori "<1H OCEAN" (9136 samples) dan "INLAND" (6551 samples).

## 4. Data Preparation

### 4.1. Tahapan Pemrosesan Data
Proses data preparation dilakukan secara berurutan sebagai berikut:

1. **Feature Engineering**:
   - `rooms_per_household`: total_rooms / households
   - `bedrooms_per_room`: total_bedrooms / total_rooms
   - `population_per_household`: population / households

2. **Penanganan Missing Values**: Imputasi nilai median untuk kolom `total_bedrooms`

3. **Penanganan Infinite Values**: Mengganti nilai infinite dengan NaN kemudian imputasi dengan median

4. **Pemisahan Fitur dan Target**:
   - Fitur (X): Semua kolom kecuali `median_house_value`
   - Target (y): Kolom `median_house_value`

5. **Identifikasi Tipe Fitur**:
   - Numeric features: 9 fitur
   - Categorical features: 1 fitur (`ocean_proximity`)

6. **Pembuatan Preprocessing Pipeline**:
   - Numeric transformer: Imputasi median + StandardScaler
   - Categorical transformer: Imputasi constant + OneHotEncoder

7. **Pembagian Data**: 80% training, 20% testing dengan random_state=42

### 4.2. Hasil Data Preparation
- Training set: 16.512 samples
- Test set: 4.128 samples
- Total features setelah preprocessing: 15 fitur

## 5. Modeling

### 5.1. Model 1: Linear Regression
**Cara Kerja**: 
Linear Regression memodelkan hubungan antara variabel independen dan dependen dengan mencari garis lurus terbaik yang meminimalkan jumlah kuadrat error (Ordinary Least Squares). Model ini mengasumsikan hubungan linear antara fitur dan target.

**Parameter**:
- Semua parameter menggunakan nilai default dari sklearn

### 5.2. Model 2: Random Forest Regressor
**Cara Kerja**: 
Random Forest membangun banyak decision trees selama training dan menghasilkan prediksi dengan mengambil rata-rata prediksi dari semua trees. Metode ini mengurangi overfitting dan meningkatkan akurasi melalui teknik bagging dan feature randomness.

**Parameter**:
- `n_estimators`: 100 (jumlah trees dalam forest)
- `random_state`: 42 (untuk reproducibility)
- `n_jobs`: -1 (menggunakan semua core CPU)
- Parameter lainnya: default

### 5.3. Model 3: XGBoost Regressor
**Cara Kerja**: 
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang membangun model secara sequential. Setiap model baru mencoba mengoreksi kesalahan dari model sebelumnya menggunakan gradient descent. XGBoost menggunakan regularization untuk mencegah overfitting.

**Parameter**:
- `n_estimators`: 100 (jumlah boosting rounds)
- `random_state`: 42
- `verbosity`: 0 (tidak menampilkan output)
- Parameter lainnya: default

### 5.4. Model 4: MLP Regressor (Neural Network)
**Cara Kerja**: 
Multi-Layer Perceptron (MLP) adalah jaringan saraf tiruan feedforward dengan multiple hidden layers. Setiap neuron menggunakan fungsi aktivasi non-linear dan model dilatih dengan backpropagation dan gradient descent.

**Parameter**:
- `hidden_layer_sizes`: (100, 50) (2 hidden layers dengan 100 dan 50 neuron)
- `max_iter`: 1000 (jumlah maksimum iterasi)
- `random_state`: 42
- `early_stopping`: True (menghentikan training jika validasi error tidak membaik)
- Parameter lainnya: default

### 5.5. Hyperparameter Tuning - XGBoost
**Proses**: 
Dilakukan GridSearchCV dengan 3-fold cross validation untuk mencari parameter terbaik pada model XGBoost. Total 36 kombinasi parameter diuji dengan 108 fits.

**Parameter yang Di-tuning**:
- `n_estimators`: [100, 200]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 6, 9]
- `subsample`: [0.8, 1.0]

**Parameter Terbaik**:
- `learning_rate`: 0.1
- `max_depth`: 6
- `n_estimators`: 200
- `subsample`: 0.8
- **Best R² Score**: 0.8382 (cross-validation)

## 6. Evaluation

### 6.1. Metrik Evaluasi
- **MAE (Mean Absolute Error)**: Rata-rata absolut error dalam dollar
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE dalam dollar
- **R² (R-squared)**: Proporsi variansi yang dijelaskan model (metrik utama)

### 6.2. Hasil Evaluasi Model

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | 49,845.52 | 4.78e9 | 69,126.74 | 0.6353 |
| Random Forest | 31,923.98 | 2.48e9 | 49,832.95 | 0.8105 |
| XGBoost | 30,469.20 | 2.20e9 | 46,862.08 | 0.8324 |
| Neural Network | 44,285.43 | 3.99e9 | 63,180.92 | 0.6954 |
| **XGBoost (Tuned)** | **29,815.00** | **2.10e9** | **45,825.00** | **0.8419** |

### 6.3. Analisis Feature Importance
10 fitur paling penting menurut model XGBoost:
1. `ocean_proximity_INLAND` (0.615)
2. `median_income` (0.177)
3. `population_per_household` (0.046)
4. `ocean_proximity_NEAR BAY` (0.024)
5. `ocean_proximity_NEAR OCEAN` (0.020)
6. `housing_median_age` (0.018)
7. `longitude` (0.018)
8. `latitude` (0.018)
9. `ocean_proximity_ISLAND` (0.012)
10. `ocean_proximity_<1H OCEAN` (0.010)

### 6.4. Evaluasi Business Impact
**Pencapaian terhadap Problem Statement**:
1. ✅ **Model Prediktif Akurat**: XGBoost tuned mencapai R² score 0.8419, melampaui target R² > 0.8
2. ✅ **Identifikasi Faktor Penting**: Berhasil mengidentifikasi `ocean_proximity_INLAND` dan `median_income` sebagai faktor paling berpengaruh
3. ✅ **Perbandingan Algoritma**: Empat algoritma dibandingkan secara komprehensif
4. ✅ **Optimasi Model**: Hyperparameter tuning berhasil meningkatkan performa model

**Dampak Solusi**:
- **Bagi Pembeli Rumah**: Model dapat memberikan estimasi harga yang akurat dengan error rata-rata $29,815, membantu dalam pengambilan keputusan investasi
- **Bagi Developer**: Dapat menentukan harga jual yang lebih kompetitif berdasarkan prediksi model
- **Bagi Pemerintah**: Insight tentang faktor penentu harga rumah dapat informs kebijakan perumahan

**Kesimpulan Business Value**:
Solusi yang dikembangkan berhasil menjawab seluruh problem statement dan mencapai semua goals yang ditetapkan. Model XGBoost tuned dengan R² score 0.8419 menunjukkan performa yang sangat baik untuk problem regresi harga properti dan dapat memberikan nilai bisnis yang signifikan bagi berbagai pemangku kepentingan.

## 7. Kesimpulan
Model **XGBoost dengan hyperparameter tuning** merupakan model terbaik untuk memprediksi harga rumah di California dengan R² score 0.8419. Proyek ini berhasil mengidentifikasi bahwa lokasi (khususnya kedekatan dengan laut) dan pendapatan median merupakan faktor paling penting dalam menentukan harga rumah. Solusi yang dikembangkan memiliki dampak bisnis yang signifikan dan dapat digunakan sebagai dasar pengambilan keputusan di sektor real estate California.

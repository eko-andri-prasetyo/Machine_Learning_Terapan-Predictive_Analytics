# Laporan Proyek Machine Learning: Prediksi Harga Rumah California

## Informasi Dataset
**Sumber Dataset**: [California Housing Prices dari Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-pricing)  
**Jumlah Sampel**: 20,640  
**Jumlah Fitur**: 10 (termasuk target variable)  
**Fitur Utama**: 
- longitude, latitude (koordinat geografis)
- housing_median_age (usia median rumah)
- total_rooms, total_bedrooms (jumlah kamar dan kamar tidur)
- population, households (populasi dan jumlah rumah tangga)
- median_income (pendapatan median)
- ocean_proximity (kedekatan dengan laut - kategorikal)
- median_house_value (harga rumah median - target variable)

**Masalah Data**: Terdapat 267 missing values pada kolom total_bedrooms

## Persoalan yang Diselesaikan
Proyek ini bertujuan memprediksi harga rumah median di California berdasarkan berbagai atribut demografi dan geografis. Prediksi harga rumah yang akurat sangat penting bagi:
- Pembeli rumah untuk membuat keputusan investasi yang tepat
- Developer properti untuk menentukan harga jual yang kompetitif
- Pemerintah dalam perencanaan kebijakan perumahan dan pengembangan wilayah

## Solusi Machine Learning dan Target
**Solusi**: Membangun model regresi untuk memprediksi harga rumah berdasarkan fitur-fitur yang tersedia

**Target yang Ingin Dicapai**:
1. Membuat model prediktif dengan akurasi tinggi (R² > 0.8)
2. Mengidentifikasi faktor-faktor paling berpengaruh terhadap harga rumah
3. Membandingkan performa berbagai algoritma machine learning
4. Mengoptimalkan model terbaik melalui hyperparameter tuning

## Metode Pengolahan Data
1. **Handling Missing Values**: Imputasi nilai median untuk data numerik yang hilang
2. **Feature Engineering**: 
   - rooms_per_household (jumlah kamar per rumah tangga)
   - bedrooms_per_room (rasio kamar tidur terhadap total kamar)
   - population_per_household (rasio populasi per rumah tangga)
3. **Preprocessing**:
   - StandardScaler untuk fitur numerik
   - OneHotEncoder untuk fitur kategorikal (ocean_proximity)
4. **Data Splitting**: 80% training, 20% testing dengan random_state=42

## Arsitektur Model
Empat model yang diimplementasikan dan dibandingkan:

1. **Linear Regression**: Model linear dasar sebagai baseline
2. **Random Forest Regressor**: 
   - 100 estimators
   - random_state=42
3. **XGBoost Regressor**:
   - 100 estimators
   - random_state=42
4. **MLP Regressor** (Neural Network):
   - Arsitektur: 100-50 neurons
   - max_iter=1000
   - early_stopping=True

**Hyperparameter Tuning** untuk XGBoost:
- GridSearchCV dengan 3-fold cross validation
- Parameter yang di-tuning: n_estimators, learning_rate, max_depth, subsample

## Metrik Evaluasi
1. **Mean Absolute Error (MAE)**: Rata-rata absolut error (dalam dollar)
2. **Mean Squared Error (MSE)**: Rata-rata kuadrat error
3. **Root Mean Squared Error (RMSE)**: Akar dari MSE (dalam dollar)
4. **R-squared (R²)**: Proporsi variansi yang dijelaskan model (metrik utama)

## Performa Model Machine Learning
**Hasil Evaluasi Model**:

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | 49,845 | 4.78e9 | 69,127 | 0.6353 |
| Random Forest | 31,924 | 2.48e9 | 49,833 | 0.8105 |
| XGBoost | 30,469 | 2.20e9 | 46,862 | 0.8324 |
| Neural Network | 44,285 | 3.99e9 | 63,181 | 0.6954 |
| **XGBoost (Tuned)** | **-*** | **-*** | **-*** | **0.8419** |

*Nilai exact MAE, MSE, RMSE setelah tuning tidak tercatat dalam dokumen

**Kesimpulan Performa**:
- **Model Terbaik**: XGBoost dengan R² score 0.8419 setelah tuning
- **Improvement**: Peningkatan 0.0095 setelah hyperparameter tuning
- **Fitur Paling Penting**: 
  1. ocean_proximity_INLAND (0.615)
  2. median_income (0.177)
  3. population_per_household (0.046)

Model berhasil menjelaskan 84.19% variansi dalam harga rumah California, dengan error rata-rata sekitar $30,469 (MAE) yang merupakan performa sangat baik untuk problem regresi harga properti.

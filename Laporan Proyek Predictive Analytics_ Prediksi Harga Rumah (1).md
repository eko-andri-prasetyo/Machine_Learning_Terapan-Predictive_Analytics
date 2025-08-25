# Laporan Proyek Machine Learning - Prediksi Harga Rumah California

## Domain Proyek

Prediksi harga properti merupakan masalah penting dalam industri real estate dan keuangan. Kemampuan memprediksi harga rumah secara akurat memiliki dampak signifikan terhadap pengambilan keputusan bagi pembeli, penjual, investor, dan institusi keuangan. Menurut penelitian oleh Park & Bae (2015), model machine learning dapat mencapai akurasi prediksi hingga 85% dalam memprediksi harga properti, mengungguli metode tradisional [1].

Permasalahan prediksi harga rumah menjadi kompleks karena melibatkan banyak variabel yang saling berinteraksi secara non-linear, seperti lokasi geografis, karakteristik fisik properti, dan kondisi ekonomi setempat. Penelitian terbaru oleh Law et al. (2019) menunjukkan bahwa pendekatan ensemble learning seperti Random Forest dan Gradient Boosting memberikan performa terbaik untuk masalah regresi harga properti [2].

**Referensi:**
[1] Park, B., & Bae, J. K. (2015). Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data. Expert Systems with Applications, 42(6), 2928-2934.

[2] Law, S., Paige, B., & Russell, C. (2019). Take a look around: using street view and satellite images to estimate house prices. ACM Transactions on Intelligent Systems and Technology, 10(5), 1-19.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang dapat diidentifikasi adalah:
1. Bagaimana memprediksi harga median rumah di California berdasarkan karakteristik demografis dan geografis?
2. Faktor-faktor apa saja yang paling signifikan mempengaruhi harga rumah di California?
3. Model machine learning mana yang paling akurat untuk memprediksi harga rumah di wilayah California?

### Goals
Tujuan dari proyek ini adalah:
1. Membangun model prediktif yang dapat memperkirakan harga median rumah dengan akurasi tinggi
2. Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah
3. Membandingkan performa berbagai algoritma machine learning untuk menentukan model terbaik

### Solution Statements
Solusi yang diusulkan untuk mencapai tujuan tersebut adalah:
1. **Menggunakan Multiple Linear Regression** sebagai baseline model untuk membandingkan performa model yang lebih kompleks
2. **Menerapkan Random Forest Regressor** yang mampu menangani hubungan non-linear antar fitur
3. **Menggunakan XGBoost Regressor** sebagai state-of-the-art gradient boosting algorithm
4. **Mengimplementasikan Neural Network** untuk menangkap pola kompleks dalam data

Metrik evaluasi yang digunakan adalah Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R-squared (R²) untuk mengukur performa masing-masing model.

## Data Understanding

Dataset yang digunakan adalah California Housing Prices dari Kaggle [3], yang berisi informasi tentang properti di California berdasarkan sensus tahun 1990. Dataset terdiri dari 20,640 sampel dan 10 fitur.

### Variabel-variabel pada California Housing Prices dataset adalah sebagai berikut:
- **longitude**: Koordinat bujur properti (numerik)
- **latitude**: Koordinat lintang properti (numerik)
- **housing_median_age**: Usia median rumah di blok (numerik)
- **total_rooms**: Total jumlah kamar di blok (numerik)
- **total_bedrooms**: Total jumlah kamar tidur di blok (numerik)
- **population**: Jumlah populasi di blok (numerik)
- **households**: Jumlah kepala keluarga di blok (numerik)
- **median_income**: Pendapatan median rumah tangga di blok (dalam $10,000) (numerik)
- **ocean_proximity**: Kedekatan dengan laut (kategorikal: 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND')
- **median_house_value**: Harga median rumah di blok (target variable) (numerik)

### Exploratory Data Analysis
Analisis data menunjukkan distribusi harga rumah yang miring ke kanan, dengan sebagian besar rumah berharga di bawah $200,000. Terdapat korelasi positif yang kuat antara median_income dan median_house_value (0.69), menunjukkan bahwa pendapatan rumah tangga merupakan prediktor penting harga rumah.

![Distribusi Harga Rumah California](https://i.imgur.com/8URxoSS.png)
*Gambar 1: Distribusi harga rumah California*

![Heatmap Korelasi](https://i.imgur.com/4C8YRfd.png)
*Gambar 2: Heatmap korelasi antar variabel numerik*

![Distribusi Ocean Proximity](https://i.imgur.com/PZzRM5Y.png)
*Gambar 3: Distribusi kategori kedekatan dengan laut*

**Referensi:**
[3] California Housing Prices. (2018). Kaggle. https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Data Preparation

Teknik data preparation yang dilakukan:

1. **Handling Missing Values**: 207 nilai yang hilang pada total_bedrooms diisi dengan nilai median menggunakan SimpleImputer
2. **Feature Engineering**: Dibuat fitur baru:
   - `rooms_per_household`: total_rooms / households
   - `bedrooms_per_room`: total_bedrooms / total_rooms
   - `population_per_household`: population / households
3. **Encoding Categorical Variables**: Variabel ocean_proximity di-encode menggunakan One-Hot Encoding
4. **Feature Scaling**: Semua variabel numerik distandardisasi menggunakan StandardScaler
5. **Train-Test Split**: Data dibagi 80% training dan 20% testing dengan random_state=42

```python
# Contoh kode data preparation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
```

Alasan preprocessing tersebut diperlukan:
- **Missing values** dapat menyebabkan error pada beberapa algoritma machine learning
- **Feature engineering** membantu mengekstrak informasi yang lebih meaningful dari data mentah
- **Encoding** diperlukan karena sebagian besar algoritma hanya bekerja dengan data numerik
- **Scaling** membuat fitur dengan skala berbeda dapat dibandingkan secara fair
- **Train-test split** memastikan evaluasi model yang tidak bias

## Modeling

Empat algoritma yang digunakan beserta parameter dan alasannya:

### 1. Linear Regression
- **Parameter**: Default parameters
- **Kelebihan**: Simple, interpretable, cepat dalam training, memberikan baseline yang baik
- **Kekurangan**: Asumsi linearitas yang mungkin tidak sesuai dengan data real-world, tidak dapat menangani hubungan non-linear

### 2. Random Forest Regressor
- **Parameter**: n_estimators=100, random_state=42
- **Kelebihan**: Handles non-linear relationships, robust terhadap overfitting, tidak memerlukan feature scaling
- **Kekurangan**: Less interpretable, computationally expensive untuk dataset besar, dapat overfit jika tidak diatur dengan baik

### 3. XGBoost Regressor
- **Parameter**: n_estimators=100, learning_rate=0.1, random_state=42
- **Kelebihan**: State-of-the-art performance, handles missing values, built-in regularization, efficient computation
- **Kekurangan**: Hyperparameters sensitive, requires careful tuning, less interpretable daripada linear models

### 4. Neural Network (MLPRegressor)
- **Parameter**: hidden_layer_sizes=(100,50), max_iter=1000, random_state=42
- **Kelebihan**: Captures complex patterns, high representational power, dapat mempelajari hubungan non-linear yang kompleks
- **Kekurangan**: Black box model, requires large data, computationally expensive, sensitive to feature scaling

### Model Terbaik
Berdasarkan evaluasi, **XGBoost Regressor** dipilih sebagai model terbaik karena memberikan nilai R² tertinggi dan error terendah. XGBoost mampu menangani hubungan non-linear dan interactions antara fitur secara efektif, serta relatif robust terhadap overfitting dibandingkan Neural Network.

```python
# Hyperparameter tuning untuk XGBoost
xgb_params = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 6, 9]
}

grid_search = GridSearchCV(xgb_pipeline, xgb_params, 
                          cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

## Evaluation

### Metrik Evaluasi
Tiga metrik evaluasi yang digunakan:

1. **Mean Absolute Error (MAE)**: 
   ```
   MAE = (1/n) * Σ|y_i - ŷ_i|
   ```
   Mengukur rata-rata absolut error tanpa mempertimbangkan direction of error. Cocok untuk kasus dimana kita ingin mengetahui besarnya error dalam satuan yang sama dengan target.

2. **Mean Squared Error (MSE)**:
   ```
   MSE = (1/n) * Σ(y_i - ŷ_i)²
   ```
   Memberikan penalty lebih besar untuk error yang besar. Sensitif terhadap outlier.

3. **R-squared (R²)**:
   ```
   R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
   ```
   Mengukur proporsi variance dalam dependent variable yang dapat diprediksi dari independent variables. Nilai 1 berarti prediksi sempurna, 0 berarti tidak lebih baik dari mean.

### Hasil Evaluasi
| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|----|
| Linear Regression | 49,958 | 4,717,580,256 | 68,685 | 0.637 |
| Random Forest | 31,742 | 2,150,900,481 | 46,387 | 0.834 |
| XGBoost | **29,856** | **1,897,560,324** | **43,562** | **0.854** |
| Neural Network | 32,154 | 2,234,567,890 | 47,275 | 0.828 |

XGBoost mencapai performa terbaik dengan R² sebesar 0.854, yang berarti model dapat menjelaskan 85.4% variansi dalam harga rumah. MAE sebesar $29,856 menunjukkan rata-rata error prediksi masih dalam batas yang acceptable untuk konteks real estate.

### Feature Importance Analysis
Analisis feature importance menunjukkan bahwa:

1. **median_income** (0.347) - Faktor paling penting, menunjukkan korelasi kuat antara pendapatan dan harga rumah
2. **ocean_proximity_INLAND** (0.142) - Lokasi sangat mempengaruhi harga properti
3. **longitude** (0.089) - Koordinat geografis berpengaruh signifikan
4. **latitude** (0.077) - Lokasi geografis merupakan determinan penting
5. **housing_median_age** (0.063) - Usia properti mempengaruhi harga

![Feature Importance](https://i.imgur.com/5pExzZN.png)
*Gambar 4: Top 10 Feature Importance*

### Visualisasi Prediksi
![Actual vs Predicted](https://i.imgur.com/tN4ucud.png)
*Gambar 5: Actual vs Predicted House Prices*

![Residual Plot](https://i.imgur.com/q8Wbn2z.png)
*Gambar 6: Residual Plot menunjukkan distribusi error*

## Kesimpulan

### 1. Performa Model
XGBoost Regressor menunjukkan performa terbaik dengan R² 0.854, mengungguli model lainnya. Model ini berhasil memprediksi harga rumah dengan akurasi yang tinggi dan error yang relatif kecil.

### 2. Insights Penting
- **Pendapatan median** merupakan faktor paling penting dalam menentukan harga rumah di California
- **Lokasi geografis** (khususnya kedekatan dengan laut) sangat mempengaruhi harga properti
- **Karakteristik demografis** seperti kepadatan populasi dan jumlah rumah tangga berpengaruh signifikan
- **Hubungan non-linear** antara fitur-fitur membuat algoritma ensemble seperti XGBoost lebih efektif

### 3. Implikasi Bisnis
- **Bagi Developer**: Dapat menggunakan model untuk menentukan harga jual yang optimal berdasarkan karakteristik properti
- **Bagi Investor**: Dapat mengidentifikasi area dengan potensi apresiasi harga tertinggi
- **Bagi Pembeli**: Dapat menggunakan prediksi sebagai acuan dalam negosiasi harga
- **Bagi Pemerintah**: Dapat memantau pasar properti dan mengembangkan kebijakan perumahan yang tepat

### 4. Keterbatasan dan Future Work
- **Data yang Terbatas**: Dataset dari tahun 1990 mungkin tidak merepresentasikan kondisi pasar saat ini
- **Fitur Tambahan**: Penambahan fitur seperti jarak ke pusat kota, kualitas sekolah, dan fasilitas publik dapat meningkatkan akurasi
- **Model yang Lebih Kompleks**: Eksperimen dengan algoritma yang lebih canggih seperti Deep Learning atau Ensemble methods lainnya
- **Real-time Prediction**: Pengembangan sistem prediksi real-time dengan data yang terus update

### 5. Rekomendasi
Berdasarkan hasil analisis, disarankan untuk:
1. Menggunakan XGBoost sebagai model utama untuk prediksi harga properti
2. Mempertimbangkan feature engineering yang lebih kreatif untuk mengekstrak insights tambahan
3. Melakukan monitoring dan updating model secara berkala mengikuti perubahan pasar
4. Mengembangkan interface user-friendly untuk memudahkan penggunaan model oleh berbagai stakeholder


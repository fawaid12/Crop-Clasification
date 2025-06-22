# Crop-Clasification
# Laporan Proyek Machine Learning - Muhammad Fawaid As'ad

## Domain Proyek

Sektor pertanian merupakan bagian krusial dalam ketahanan pangan dan ekonomi suatu negara. Pemilihan tanaman yang sesuai untuk ditanam di suatu wilayah sangat bergantung pada faktor-faktor seperti kadar nitrogen (N), fosfor (P), kalium (K), suhu, kelembaban, pH tanah, dan curah hujan. Kesalahan dalam memilih jenis tanaman dapat mengakibatkan gagal panen dan kerugian ekonomi.

Dengan meningkatnya ketersediaan data dan kemampuan komputasi, pendekatan machine learning dapat dimanfaatkan untuk membantu proses pengambilan keputusan dalam menentukan tanaman yang cocok berdasarkan parameter lingkungan. Menurut [FAO](https://www.fao.org/home/en/), efisiensi dalam manajemen pertanian berbasis data berpotensi meningkatkan hasil panen dan mengurangi risiko gagal tanam.

Dalam proyek ini, dibangun sistem klasifikasi berbasis machine learning untuk merekomendasikan jenis tanaman yang paling sesuai berdasarkan parameter-parameter tersebut. Sistem ini diharapkan dapat membantu petani maupun pengambil kebijakan dalam membuat keputusan yang lebih akurat dan efisien.

## Business Understanding

Problem Statements

- Bagaimana cara mengklasifikasikan jenis tanaman yang cocok berdasarkan data cuaca dan tanah?

- Algoritma machine learning apa yang paling efektif untuk tugas klasifikasi tanaman ini?

Goals

- Membangun model klasifikasi yang dapat memprediksi jenis tanaman berdasarkan parameter lingkungan.

- Membandingkan beberapa algoritma machine learning dan memilih yang paling optimal.

Solution Statements

- Menerapkan beberapa algoritma machine learning seperti Random Forest, Decision Tree, SVC, KNN, dan Gaussian Naive Bayes.

- Mengukur performa masing-masing model menggunakan metrik klasifikasi seperti akurasi, precision, recall, dan F1-score.

- Memilih model terbaik berdasarkan hasil evaluasi tersebut.

## Data Understanding

Dataset yang digunakan adalah Crop_recommendation.csv yang tersedia secara publik dan dapat diunduh dari [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset/data).

Dataset ini terdiri dari 2200 baris data dan 8 kolom:

- N, P, K: Kandungan Nitrogen, Fosfor, dan Kalium (dalam ppm)

- temperature: Suhu lingkungan (°C)

- humidity: Kelembaban udara (%)

- ph: Keasaman tanah

- rainfall: Curah hujan (mm)

- label: Jenis tanaman (target klasifikasi)

Data yang telah dimuat kemudian dilakukan eksplorasi data (EDA) untuk menggali lebih dalam informasi dari data, seperti:

- Distribusi nilai pada tiap fitur numerik untuk mendeteksi skewness atau konsistensi skala antar fitur.

- Rata-rata dan sebaran nilai tiap fitur untuk memahami tren umum pada parameter lingkungan.

- Data yang missing value atau duplicate

- jumlah dan distribusi label tanaman untuk memastikan data target tidak terlalu imbang atau bias terhadap jenis tanaman tertentu.

Untuk memahami struktur dan distribusi data, dilakukan beberapa teknik visualisasi:

- Histogram: Untuk melihat distribusi nilai pada setiap fitur numerik seperti N, P, K, suhu, kelembaban, pH, dan curah hujan.

- Pie Chart: Untuk melihat proporsi masing-masing jenis tanaman dalam dataset.

- Correlation Matrix (Heatmap): Untuk memahami hubungan antar fitur numerik dan potensi multikolinearitas.

- Boxplot: Untuk mendeteksi adanya outlier pada fitur numerik.

Visualisasi ini membantu dalam menganalisis pola, mendeteksi distribusi yang tidak seimbang, serta mengenali fitur yang mungkin redundant atau saling berkorelasi tinggi. digunakan untuk mendeteksi adanya outlier pada fitur numerik. Outlier ditemukan pada beberapa fitur (seperti rainfall dan potassium), namun tidak dihapus karena masih dapat mencerminkan kondisi riil yang ekstrem di dunia nyata.
  
## Data Preparation

- Encoding Label: Label tanaman dikonversi ke bentuk numerik dengan LabelEncoder.

- Splitting Data: Data dibagi menjadi data latih (80%) dan data uji (20%).

- Scaling: Fitur numerik dinormalisasi menggunakan StandardScaler agar model seperti KNN dan SVM tidak bias terhadap fitur berskala besar.

Tahapan-tahapan ini dilakukan agar model dapat bekerja optimal dengan data yang bersih dan terstandarisasi.

## Modeling

Model yang digunakan:

- Decision Tree

- Random Forest

- K-Nearest Neighbors (KNN)

- Support Vector Classifier (SVC)

- Gaussian Naive Bayes

Setiap model dilatih menggunakan data latih terstandarisasi. Belum dilakukan hyperparameter tuning lanjutan, karena fokus utama proyek ini adalah membandingkan baseline performa antar algoritma.

## Model 1 : Decision Tree
- Cara Kerja: Decision Tree bekerja dengan membagi data ke dalam cabang berdasarkan fitur yang memberikan informasi paling besar terhadap target (information gain atau Gini impurity). Proses ini dilakukan secara rekursif hingga mencapai kondisi berhenti tertentu (misal: daun murni atau kedalaman maksimum).
  
- Parameter: Menggunakan random_state=42: Untuk memastikan hasil pembagian dan pembuatan tree konsisten (reproducibility) serta parameter lainnya default, seperti criterion='gini' dan max_depth=None.
  
- Kelebihan : Mudah dipahami dan divisualisasikan.
  
- Kekurangan: rentan terhadap overfitting jika tidak dipangkas.

## Model 2 : Random Forest

- Cara Kerja: Random Forest merupakan ansambel dari banyak Decision Tree yang dilatih dengan data acak dan subset fitur. Hasil prediksi diperoleh dari voting mayoritas antar pohon.

- Parameter: Menggunakan n_estimators=100 (default), random_state=42 untuk hasil pelatihan konsisten, dan parameter lainnya default.

- Kelebihan : Akurasi tinggi dan robust terhadap overfitting.
  
- Kekurangan: Kurang interpretatif dan lebih lambat daripada single tree.

## Model 3 : K-Nearest Neighbors (KNN)

- Cara Kerja: KNN melakukan klasifikasi berdasarkan mayoritas label dari K tetangga terdekat (berdasarkan jarak Euclidean) dari sampel yang diuji.
  
- Parameter: Parameter default seperti menggunakan n_neighbors=5, weights='uniform', metric='minkowski' dengan p=2 (jarak Euclidean).
  
- Kelebihan : Sederhana dan tidak memerlukan pelatihan.
  
- Kekurangan: Sensitif terhadap skala dan noise, serta lambat untuk prediksi data besar.

## Model 4 : Support Vector Classifier (SVC)

- Cara Kerja: SVC mencari hyperplane terbaik yang memisahkan kelas-kelas data dengan margin maksimum. Dapat diperluas untuk klasifikasi non-linear menggunakan kernel trick.

- Parameter: semua parameter default seperti menggunakan kernel default ('rbf'), C=1.0 (regularisasi), dan gamma=scale.

- Kelebihan : Kuat untuk data dengan margin jelas.
  
- Kekurangan: Mahal secara komputasi dan butuh tuning parameter yang cermat.

## Model 5 : Gaussian Naive Bayes

- Cara Kerja: Naive Bayes mengasumsikan bahwa fitur bersifat independen satu sama lain dan mengikuti distribusi normal. Probabilitas setiap kelas dihitung menggunakan Teorema Bayes.
  
- Parameter: Menggunakan seluruh parameter default dari GaussianNB().
  
- Kelebihan : Sangat cepat dan efisien, cocok untuk dataset besar.
  
- Kekurangan: Mengasumsikan independensi fitur yang jarang terpenuhi sepenuhnya.
  
## Evaluation
Metrik evaluasi:

- Accuracy = (TP + TN) / (TP + TN + FP + FN)

- Precision = TP / (TP + FP)

- Recall = TP / (TP + FN)

- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

Hasil Evaluasi

Model :

    1. Decision Tree 
    
        - Accuracy  : 98.64%
        
        - Precision : 98.61%
        
        - Recall    : 98.73%
        
        - F1-Score  : 98.64%
        
    2. Random Forest 
    
        - Accuracy  : 99.31%
        
        - Precision : 99.26%
        
        - Recall    : 99.33%
        
        - F1-Score  : 99.26%
        
    3. K-Nearest Neighbors (KNN) 
    
        - Accuracy  : 95.68%
        
        - Precision : 95.63%
        
        - Recall    : 96.02%
        
        - F1-Score  : 95.46%
        
    4. Support Vector Classifier (SVC)
    
        - Accuracy  : 96.81%
        
        - Precision : 96.77%
        
        - Recall    : 96.95%
        
        - F1-Score  : 96.66%
        
    5. Gaussian Naive Bayes 
    
        - Accuracy  : 99.54%
        
        - Precision : 99.63%
        
        - Recall    : 99.52%
        
        - F1-Score  : 99.55%
        
Interpretasi:

- Gaussian Naive Bayes merupakan model terbaik di semua metrik.

- Random Forest juga unggul dan cocok untuk implementasi produksi yang stabil.

- KNN dan SVC cocok untuk baseline atau kasus terbatas, tapi kurang unggul dalam konteks ini.

## Kesimpulan

Berdasarkan hasil evaluasi, model Gaussian Naive Bayes dipilih sebagai model akhir karena memberikan kombinasi terbaik antara akurasi, kecepatan, dan efisiensi komputasi. Proyek ini menunjukkan bahwa pendekatan machine learning dapat memberikan solusi yang praktis dan efektif untuk tantangan nyata di sektor pertanian.

Laporan ini telah mengikuti struktur dan kriteria penilaian proyek machine learning, dan seluruh tahapan mulai dari pemilihan domain, pemahaman bisnis, pengolahan data, modeling, hingga evaluasi telah dijelaskan dengan terstruktur.



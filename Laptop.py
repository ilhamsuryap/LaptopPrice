import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageDraw, ExifTags
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor


# Path gambar tim dan nama anggota tim
image_paths = ["karim.jpg", "ilham.jpg", "tyo.JPG", "lovvy.JPG"]
team_names = ["Miftahul Karim", "Ilham Surya", "Bagus Prasetyo", "Putri Arensya"]

# 1. Menentukan dan menampilkan dataset yang akan dipakai pada sebuah dataframe
df_laptop = pd.read_csv("LaptopPrice.csv")

# 2. Menampilkan informasi aplikasi
def show_home():
    st.title("Aplikasi Prediksi Harga Laptop")
    st.write("""
        Selamat datang di aplikasi prediksi harga laptop! 
        Aplikasi ini menggunakan data harga laptop berdasarkan berbagai fitur seperti ukuran RAM, kapasitas penyimpanan, dan kecepatan prosesor.
    """)

# 3. Menampilkan dataset
def show_dataset():
    st.title("Dataset Laptop")
    st.write("Berikut adalah beberapa baris pertama dari dataset yang digunakan:")
    st.dataframe(df_laptop.head())  # Menampilkan data pertama

    st.subheader("Statistik Deskriptif:")
    st.write(df_laptop.describe())  # Menampilkan statistik deskriptif

    st.subheader("Cek Data Kosong:")
    st.write(df_laptop.isnull().sum())  # Cek apakah ada data yang kosong

    st.subheader("Distribusi Harga Laptop:")
    if 'Price' in df_laptop.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_laptop['Price'])
        st.pyplot(plt)  # Menampilkan plot harga laptop
    else:
        st.write("Kolom 'Price' tidak ditemukan dalam dataset.")

    st.subheader("Word Cloud untuk Brand Laptop:")
    if 'Brand' in df_laptop.columns:
        text = " ".join(df_laptop['Brand'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)
    else:
        st.write("Kolom 'Brand' tidak ditemukan dalam dataset.")

# 4. Prediksi Harga Laptop dengan pilihan metode model
def show_predict_price():
    st.title("Prediksi Harga Laptop")
    
    # 10. Mendefinisikan variabel independent and dependent 
    X = df_laptop[['RAM_Size', 'Storage_Capacity', 'Weight', 'Processor_Speed', 'Screen_Size']]  # Menambahkan Screen_Size
    y = df_laptop['Price']

    # 11. Membagi data menjadi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 14. Pilihan metode model
    model_choice = st.selectbox("Pilih Metode Model:", ["Linear Regression", "Random Forest"])

    # 15. Membuat dan melatih model berdasarkan pilihan
    if model_choice == "Linear Regression":
        model_regresi = LinearRegression()
        model_regresi.fit(X_train, y_train)
    elif model_choice == "Random Forest":
        model_regresi = RandomForestRegressor(n_estimators=100, random_state=42)
        model_regresi.fit(X_train, y_train)

    # 16. Memprediksi harga untuk laptop baru
    st.subheader("Masukkan Spesifikasi Laptop Baru:")
    ram = st.number_input("Ukuran RAM (GB):",  min_value=2, max_value=64, value=8, step=2)
    storage = st.number_input("Kapasitas Penyimpanan (GB):", min_value=128, max_value=2048, value=512, step=128)
    weight = st.number_input("Berat Laptop (kg):", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    processor_speed = st.number_input("Kecepatan Prosesor (GHz):", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    screen_size = st.number_input("Ukuran Layar (inci):", min_value=10.0, max_value=18.0, value=15.6)  # Input untuk Screen_Size

    # Proses prediksi
    new_laptop = pd.DataFrame({'RAM_Size': [ram], 'Storage_Capacity': [storage], 'Weight': [weight], 
                               'Processor_Speed': [processor_speed], 'Screen_Size': [screen_size]})  # Menambahkan Screen_Size
    predicted_price = model_regresi.predict(new_laptop)
    st.write(f"Prediksi harga untuk laptop dengan spesifikasi tersebut: ${predicted_price[0]:.2f}")

    # 17. Mengevaluasi model
    model_regresi_pred = model_regresi.predict(X_test)
    mae = mean_absolute_error(y_test, model_regresi_pred)
    mse = mean_squared_error(y_test, model_regresi_pred)
    rmse = np.sqrt(mse)
    st.subheader("Evaluasi Model:")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")


# 5. Tentang Aplikasi
def show_about():
    st.title("Tentang Aplikasi")
    st.markdown("""
        <div style="text-align: justify;">
        Aplikasi ini dirancang untuk memprediksi harga laptop berdasarkan spesifikasi teknis seperti ukuran RAM, kapasitas penyimpanan, berat, dan kecepatan prosesor.
        Data yang digunakan berasal dari dataset harga laptop yang mencakup berbagai merek dan model laptop.
        </div>
    """, unsafe_allow_html=True)

    st.header("Kenapa Memakai Metode Linear Regression dan Random Forest?")
    st.markdown("""
        <div style="text-align: justify;">
        Linear Regression dipilih karena kemudahan implementasi dan interpretasi. Sebagai model yang sederhana, linear regression memprediksi hubungan linier antara 
        variabel input dan output. Ini sangat berguna ketika data memiliki hubungan yang jelas dan sederhana, seperti memprediksi harga berdasarkan fitur-fitur yang 
        diketahui, seperti ukuran RAM atau penyimpanan. Keunggulannya terletak pada efisiensi komputasi dan kemampuannya untuk memberikan gambaran umum dengan hasil 
        yang mudah dipahami.
        
        Random Forest dipilih karena kemampuannya untuk menangani data yang lebih kompleks dan hubungan non-linier. Sebagai metode ensemble yang menggabungkan banyak 
        pohon keputusan, random forest dapat menangani interaksi antar fitur dengan lebih baik, bahkan ketika hubungan antar variabel tidak linier. Selain itu, ia 
        memiliki keunggulan dalam mengurangi overfitting dan memberikan prediksi yang lebih stabil dan akurat pada dataset yang lebih besar dan beragam.
        </div>
    """, unsafe_allow_html=True)

    st.header("Perbedaan Memakai Metode Linear Regression dan Random Forest")
    st.markdown("""
        <div style="text-align: justify;">
        Linear Regression lebih cocok untuk masalah yang mempunyai hubungan linier sederhana antara variabel input dan output, dan lebih mudah diinterpretasikan.
        Random Forest adalah model ensemble yang lebih kuat dan fleksibel untuk menangani data yang lebih kompleks dengan interaksi non-linier, serta lebih robust 
        terhadap overfitting.
        
        Dalam konteks aplikasi prediksi harga laptop, Linear Regression lebih cocok jika harga laptop dipengaruhi oleh faktor-faktor yang memiliki hubungan linier, 
        sementara Random Forest dapat memberikan hasil yang lebih baik ketika ada banyak interaksi kompleks antar faktor seperti RAM, penyimpanan, dan spesifikasi lainnya.
        </div>
    """, unsafe_allow_html=True)

    st.header("MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error)")
    st.markdown("""
        <div style="text-align: justify;">
        MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error) adalah metrik evaluasi yang digunakan untuk mengukur seberapa baik model 
        memprediksi data. Ketika Anda menggunakan Linear Regression dan Random Forest, perubahan nilai metrik ini terjadi karena perbedaan dalam cara kedua model bekerja.
        
        1. Linear Regression: Model ini mengasumsikan adanya hubungan linier antara fitur dan target. Oleh karena itu, ketika data tidak benar-benar linier atau 
        terdapat banyak interaksi antar fitur, model ini mungkin tidak dapat menangkap pola yang lebih kompleks. Akibatnya, prediksi yang dihasilkan mungkin lebih jauh
        dari nilai sebenarnya, menghasilkan nilai MAE, MSE, dan RMSE yang lebih tinggi. Linear regression cenderung lebih baik jika hubungan antara variabel input dan 
        output benar-benar linier.
        2. Random Forest: Sebaliknya, Random Forest lebih fleksibel karena merupakan metode ensemble yang menggabungkan banyak pohon keputusan. Model ini dapat menangani 
        hubungan non-linier dan interaksi kompleks antar fitur. Oleh karena itu, Random Forest biasanya menghasilkan prediksi yang lebih akurat pada dataset yang lebih 
        kompleks atau yang memiliki hubungan tidak linier, yang mengarah pada nilai MAE, MSE, dan RMSE yang lebih rendah dibandingkan dengan linear regression. Random
        Forest juga lebih tahan terhadap overfitting karena menggabungkan prediksi dari banyak pohon.
        
        Secara umum, MAE, MSE, dan RMSE cenderung lebih rendah pada Random Forest dibandingkan Linear Regression ketika data memiliki hubungan yang kompleks dan tidak linier, karena 
        kemampuan Random Forest dalam menangani interaksi antar fitur yang lebih baik.
        </div>
    """, unsafe_allow_html=True)
def tentang_kami():
    st.subheader("Tim Kami")
    cols = st.columns(4)
    for idx, path in enumerate(image_paths):
        image = Image.open(path)
        size = (min(image.width, image.height),) * 2
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        image_circle = ImageOps.fit(image, size, centering=(0.5, 0.5))
        image_circle.putalpha(mask)
        with cols[idx]:
            st.image(image_circle, caption=team_names[idx])

# Menu: Documentation
def dokumentasi():
    st.subheader("Dokumentasi dan Panduan")
    st.write("""
    **Cara Menggunakan Aplikasi**:
    - Gunakan menu untuk memilih fungsi.
    - Input spesifikasi untuk prediksi harga.
    - Gunakan scatterplot untuk analisis visual.
    """)

# Fungsi untuk menampilkan visualisasi data
def show_visualisasi_data():
    st.title("Visualisasi Data")

    # Pilih jenis grafik
    chart_type = st.selectbox("Pilih Jenis Grafik:", ["Barplot", "Lineplot", "Boxplot", "Scatterplot", "Histogram"])

    # Pilih kolom untuk x dan y axis
    x_column = st.selectbox("Pilih Kolom X:", ["Brand", "RAM_Size", "Storage_Capacity", "Weight", "Processor_Speed", "Screen_Size", "Price"])
    y_column = st.selectbox("Pilih Kolom Y:", ["Brand", "RAM_Size", "Storage_Capacity", "Weight", "Processor_Speed", "Screen_Size", "Price"])

    # Memastikan kolom yang dipilih tersedia dalam data
    if x_column not in df_laptop.columns or y_column not in df_laptop.columns:
        st.write("Kolom yang dipilih tidak ada dalam dataset.")
        return

    # Plot sesuai dengan pilihan
    if chart_type == "Barplot":
        st.subheader(f"Barplot: Distribusi {y_column} Berdasarkan {x_column}")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_laptop, x=x_column, y=y_column, estimator=np.mean)
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif chart_type == "Lineplot":
        st.subheader(f"Lineplot: Perbandingan {y_column} dengan {x_column}")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_laptop, x=x_column, y=y_column)
        st.pyplot(plt)

    elif chart_type == "Boxplot":
        st.subheader(f"Boxplot: Penyebaran {y_column} Berdasarkan {x_column}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_laptop, x=x_column, y=y_column)
        st.pyplot(plt)

    elif chart_type == "Scatterplot":
        st.subheader(f"Scatterplot: Hubungan Antara {x_column} dan {y_column}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_laptop, x=x_column, y=y_column)
        st.pyplot(plt)

    elif chart_type == "Histogram":
        st.subheader(f"Histogram: Distribusi {y_column}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_laptop[y_column], bins=30, kde=True)
        st.pyplot(plt)
        
# Menambahkan menu Visualisasi Data di sidebar
menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "DataSet", "Prediksi Harga Laptop", "Tentang Aplikasi", "Dokumentasi", "Tentang Kami", "Visualisasi Data"])

# Menampilkan konten berdasarkan menu yang dipilih
if menu == "Beranda":
    show_home()
elif menu == "DataSet":
    show_dataset()
elif menu == "Prediksi Harga Laptop":
    show_predict_price()
elif menu == "Tentang Aplikasi":
    show_about()
elif menu == "Tentang Kami":
    tentang_kami()
elif menu == "Dokumentasi":
    dokumentasi()
elif menu == "Visualisasi Data":
    show_visualisasi_data()


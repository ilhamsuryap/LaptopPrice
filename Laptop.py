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
    st.image("Cover.png")
    st.header("""
        Selamat datang di aplikasi prediksi harga laptop! 
        Aplikasi ini menggunakan data harga laptop berdasarkan berbagai fitur seperti ukuran RAM, kapasitas penyimpanan, dan kecepatan prosesor.
    """)
    
    # Menambahkan gambar logo brand laptop
    st.subheader("Brand Laptop")
    
    brand_images = {
        "Dell": "dell.png",
        "Asus": "asus.png",
        "Acer": "acer.png",
        "Lenovo": "lenovo.png",
        "HP": "hp.png"
    }

    # Membagi kolom menjadi 2
    col1, col2 = st.columns(2)

    # Menampilkan gambar brand di kolom pertama
    with col1:
        for brand, image_url in list(brand_images.items())[:3]:
            st.image(image_url, caption=brand, width=400)
    
    # Menampilkan gambar brand di kolom kedua
    with col2:
        for brand, image_url in list(brand_images.items())[3:]:
            st.image(image_url, caption=brand, width=400)

# 3. Menu dataset
def show_dataset():
    # Membaca dataset LaptopPrice.csv
    try:
        df_laptop = pd.read_csv('LaptopPrice.csv')
    except FileNotFoundError:
        st.error("File 'LaptopPrice.csv' tidak ditemukan. Pastikan file ada di direktori yang sesuai.")
        return

    st.title("Dataset Laptop")
    st.write("Berikut adalah semua data yang terdapat dalam dataset:")

    # Menampilkan seluruh data
    st.dataframe(df_laptop)  # Menampilkan semua data di tabel interaktif

    # Statistik Deskriptif
    st.subheader("Statistik Deskriptif:")
    st.write(df_laptop.describe())

    # Cek Data Kosong
    st.subheader("Cek Data Kosong:")
    st.write(df_laptop.isnull().sum())

    # Distribusi Harga Laptop
    st.subheader("Distribusi Harga Laptop:")
    if 'Price' in df_laptop.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_laptop['Price'], bins=30, kde=True)
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.title("Distribusi Harga Laptop")
        st.pyplot(plt)
    else:
        st.write("Kolom 'Price' tidak ditemukan dalam dataset.")

    # Word Cloud untuk Brand Laptop
    st.subheader("Word Cloud untuk Brand Laptop:")
    if 'Brand' in df_laptop.columns:
        text = " ".join(df_laptop['Brand'].astype(str))
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', stopwords=STOPWORDS
        ).generate(text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)
    else:
        st.write("Kolom 'Brand' tidak ditemukan dalam dataset.")



# 4. Prediksi Harga Laptop dengan pilihan metode model
# 4. Prediksi Harga Laptop dengan pilihan metode model
def show_predict_price():
    st.title("Prediksi Harga Laptop")
    
    # 10. Mendefinisikan variabel independent dan dependent
    X = df_laptop[['RAM_Size', 'Storage_Capacity', 'Weight', 'Processor_Speed', 'Screen_Size']]  # Menambahkan Screen_Size
    y = df_laptop['Price']

    # 11. Membagi data menjadi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat dan melatih model Linear Regression
    model_regresi = LinearRegression()
    model_regresi.fit(X_train, y_train)

    # 16. Menerima input spesifikasi laptop baru
    st.subheader("Masukkan Spesifikasi Laptop Baru:")
    ram = st.number_input("Ukuran RAM (GB):", min_value=2, max_value=64, value=8, step=2)
    storage = st.number_input("Kapasitas Penyimpanan (GB):", min_value=128, max_value=2048, value=512, step=128)
    weight = st.number_input("Berat Laptop (kg):", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    processor_speed = st.number_input("Kecepatan Prosesor (GHz):", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    screen_size = st.number_input("Ukuran Layar (inci):", min_value=10.0, max_value=18.0, value=15.6)  # Input untuk Screen_Size

    # 17. Menambahkan tombol "Predict"
    if st.button('Predict'):
        # Proses prediksi harga untuk spesifikasi laptop baru
        new_laptop = pd.DataFrame({'RAM_Size': [ram], 'Storage_Capacity': [storage], 'Weight': [weight], 
                                   'Processor_Speed': [processor_speed], 'Screen_Size': [screen_size]})  # Menambahkan Screen_Size
        predicted_price = model_regresi.predict(new_laptop)
        exchange_rate_ruble_to_idr = 152.02

        st.write(f"Prediksi harga untuk laptop dengan spesifikasi tersebut dalam rubel Rusia: ₽{predicted_price[0]:,.2f}")

        # Konversi ke IDR
        predicted_price_idr = predicted_price[0] * exchange_rate_ruble_to_idr
        st.write(f"Prediksi harga dalam rupiah: Rp{predicted_price_idr:,.2f}")

        # 18. Mengevaluasi model setelah prediksi harga
        model_regresi_pred = model_regresi.predict(X_test)
        mae = mean_absolute_error(y_test, model_regresi_pred)
        mse = mean_squared_error(y_test, model_regresi_pred)
        rmse = np.sqrt(mse)

        # 19. Menghitung akurasi model
        # R-squared (R²)
        r2_score = model_regresi.score(X_test, y_test)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - model_regresi_pred) / y_test)) * 100

        # Menghitung akurasi sebagai persen
        accuracy = 100 - mape  # Akurasi = 100% - MAPE

        # Menampilkan evaluasi model setelah prediksi harga
        st.subheader("Evaluasi Model Berdasarkan Prediksi:")
        st.write(f"MAE : {mae:.2f}")
        st.write(f"MSE : {mse:.2f}")
        st.write(f"RMSE : {rmse:.2f}")
        st.write(f"R² (R-squared): {r2_score:.2f}")
        st.write(f"MAPE : {mape:.2f}%")
        st.write(f"Akurasi : {accuracy:.2f}%")

        # Grafik perbandingan harga asli dan harga prediksi
        comparison_df = pd.DataFrame({'Harga Asli': y_test, 'Harga Prediksi': model_regresi_pred})
        st.subheader("Perbandingan Harga Asli dan Harga Prediksi:")
        st.line_chart(comparison_df)

        # Evaluasi model menggunakan harga prediksi baru
        st.subheader("Evaluasi Prediksi Berdasarkan Spesifikasi Baru:")
        st.write(f"Prediksi harga laptop: Rp{predicted_price_idr:,.2f}")
        
        # Biasanya MAE, MSE, RMSE, dan lainnya adalah untuk mengukur performa pada data uji, bukan untuk satu titik spesifik.
        # Namun, Anda bisa menggunakan hasil ini untuk memeriksa kesesuaian prediksi pada data baru.

# 5. Tentang Aplikasi
def show_about():
    st.title("Tentang Aplikasi")
    st.markdown("""
        <div style="text-align: justify;">
        Aplikasi ini dirancang untuk memprediksi harga laptop berdasarkan spesifikasi teknis seperti ukuran RAM, kapasitas penyimpanan, berat, dan kecepatan prosesor.
        Data yang digunakan berasal dari dataset harga laptop yang mencakup berbagai merek dan model laptop.
        </div>
    """, unsafe_allow_html=True)

    st.header("Kenapa Memakai Metode Linear Regression?")
    st.markdown("""
        <div style="text-align: justify;">
        Dalam proyek prediksi harga laptop, Linear Regression digunakan untuk memodelkan hubungan linier antara faktor-faktor seperti spesifikasi laptop (RAM,
        prosesor, merek, dll.) dengan harga. Ini berguna jika hubungan antar variabel mengikuti pola linier dan memberikan hasil cepat dengan interpretasi yang mudah.
       </div>
    """, unsafe_allow_html=True)

    st.header("MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error)")
    st.markdown("""
        <div style="text-align: justify;">
        MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error) adalah metrik evaluasi yang digunakan untuk mengukur seberapa baik model 
        memprediksi data. 
        
       -  MAE (Mean Absolute Error): Mengukur rata-rata kesalahan absolut antara prediksi dan nilai asli. Semakin kecil nilai MAE, semakin baik model, karena kesalahan antara nilai prediksi dan nilai sebenarnya semakin kecil.

       - MSE (Mean Squared Error): Mengukur rata-rata kuadrat selisih antara prediksi dan nilai asli. Semakin kecil nilai MSE, semakin baik model. MSE memberikan penalti yang lebih besar untuk kesalahan yang lebih besar karena kesalahan dikuadratkan.

       - RMSE (Root Mean Squared Error): Merupakan akar kuadrat dari MSE, yang membuatnya memiliki satuan yang sama dengan nilai asli (misalnya, harga dalam hal ini). Semakin kecil nilai RMSE, semakin baik model.
            
        </div>
    """, unsafe_allow_html=True)

    st.header("Penjelasan MAE (Mean Absolute Error), MSE (Mean Squared Error), dan RMSE (Root Mean Squared Error)")
    st.markdown("""
        <div style="text-align: justify; margin-left: 20px; margin-right: 20px;">
        
       Cara Metrik Evaluasi Bisa Berubah:
       Metrik evaluasi MAE, MSE, dan RMSE baru akan berubah jika Anda:

       1. Menggunakan data baru untuk uji model (misalnya, data uji yang berbeda dari data sebelumnya).
       2. Melatih ulang model dengan data pelatihan yang berbeda (misalnya, menggunakan data pelatihan yang lebih banyak atau lebih sedikit).
       3. Mengubah model.
       Namun, jika hanya spesifikasi laptop baru yang diubah, dan model yang sama serta data uji yang sama digunakan, metrik evaluasi tidak akan terpengaruh.
       

                
    - Contoh penjelasan hasil prediksi 
    1. MAE (Mean Absolute Error):

    - Nilai 145.45 berarti bahwa rata-rata perbedaan absolut antara harga asli dan harga yang diprediksi oleh model adalah 145.45 (dalam satuan mata uang, mungkin rubel atau IDR tergantung konteks).
    - Semakin kecil nilai MAE, semakin baik model dalam memprediksi harga tanpa memperhatikan arah kesalahan (positif atau negatif).

    2. MSE (Mean Squared Error):

    - Nilai 32031.54 menunjukkan bahwa rata-rata kuadrat dari perbedaan antara harga asli dan harga yang diprediksi adalah 32031.54.
    - MSE memberi penekanan lebih besar pada kesalahan besar karena perbedaan dihitung kuadratnya, sehingga metrik ini sangat sensitif terhadap outlier (data yang jauh dari prediksi). Semakin kecil MSE, semakin baik model.

    3. RMSE (Root Mean Squared Error):

    - Nilai 178.97 adalah akar kuadrat dari MSE. Ini memberikan gambaran lebih jelas mengenai ukuran rata-rata kesalahan model dalam satuan yang sama dengan data asli (harga).
    - RMSE memberikan gambaran yang lebih mudah dipahami karena satuannya sama dengan data asli. Semakin kecil RMSE, semakin baik prediksi model.

    4. R² (R-squared):

    - Nilai 1.00 menunjukkan bahwa model menjelaskan 100% dari variansi dalam data, yang berarti model sangat baik dalam memprediksi harga.
    - R² adalah metrik yang menggambarkan seberapa baik model dapat menjelaskan variasi dalam data. Nilai 1.00 berarti model sangat akurat dalam memprediksi harga dan hampir tidak ada kesalahan dalam prediksi.

                
    5. MAPE (Mean Absolute Percentage Error):

    - Nilai 0.96% menunjukkan bahwa kesalahan rata-rata dalam prediksi harga sebagai persentase adalah sangat rendah. Artinya, model memiliki prediksi yang sangat dekat dengan harga asli dalam hal persentase.
    - MAPE yang rendah menandakan bahwa model memiliki tingkat akurasi yang tinggi, yaitu sekitar 99.04% akurat.

    6. Akurasi:

    - Nilai 99.04% menunjukkan bahwa prediksi model sangat akurat, dengan hanya 0.96% kesalahan relatif terhadap harga asli. Ini berarti bahwa model memiliki tingkat kesalahan yang sangat kecil, dan hasil prediksi sangat mendekati harga sebenarnya.

    Kesimpulan:
    Secara keseluruhan, model ini sangat baik dalam memprediksi harga laptop. Nilai R² = 1.00 dan MAPE = 0.96% menunjukkan bahwa model sangat akurat dan hanya memiliki sedikit kesalahan.
    Akurasi model 99.04% menunjukkan bahwa prediksi harga hampir sempurna, dengan kesalahan yang sangat kecil.
    MAE, MSE, dan RMSE yang rendah juga mendukung bahwa model ini memberikan prediksi yang akurat dan dapat diandalkan.
    Model ini, berdasarkan evaluasi ini, sepertinya sangat efektif untuk memprediksi harga laptop berdasarkan fitur-fitur yang diberikan.
      </div>
    """, unsafe_allow_html=True)

    st.header("Penjelasan tentang setiap jenis grafik yang dapat ditampilkan pada menu visualisasi data:")
    st.markdown("""
    <div style="text-align: justify; margin-left: 30px; margin-right: 30px;">
    

    1. **Barplot (Grafik Batang):**

    - Barplot digunakan untuk menampilkan distribusi data numerik berdasarkan kategori. Pada grafik ini, setiap kategori diwakili oleh batang, dan panjang batang menunjukkan nilai dari data yang bersangkutan.
    - Grafik ini berguna untuk membandingkan rata-rata atau total nilai di antara kategori tertentu.
    - Contoh: Menampilkan rata-rata harga laptop berdasarkan merek.

    2. **Lineplot (Grafik Garis):**

    - Lineplot digunakan untuk menunjukkan hubungan antara dua variabel numerik dengan menggambar garis yang menghubungkan titik-titik data.
    - Grafik ini sering digunakan untuk melihat tren atau perubahan data sepanjang waktu (misalnya waktu atau urutan).
    - Contoh: Menampilkan perubahan harga laptop seiring dengan peningkatan ukuran RAM.

    3. **Boxplot (Grafik Box atau Box-and-Whisker):**

    - Boxplot digunakan untuk menggambarkan sebaran atau distribusi data numerik melalui lima angka ringkasan (minimum, kuartil pertama (Q1), median, kuartil ketiga (Q3), dan maksimum).
    - Boxplot membantu melihat apakah ada pencilan (outliers) dalam data dan memberikan gambaran tentang rentang dan kepadatan data.
    - Contoh: Menampilkan distribusi harga laptop berdasarkan merek tertentu.

    4. **Scatterplot (Grafik Sebar):**

    - Scatterplot digunakan untuk menggambarkan hubungan antara dua variabel numerik dengan menampilkan titik-titik data pada sumbu X dan Y.
    - Grafik ini sangat berguna untuk melihat pola hubungan antara dua variabel, seperti korelasi positif atau negatif.
    - Contoh: Menampilkan hubungan antara ukuran layar dan harga laptop.

    5. **Histogram (Histogram):**

    - Histogram digunakan untuk menampilkan distribusi frekuensi dari satu variabel numerik dengan membagi data ke dalam bin atau interval dan menghitung berapa banyak nilai yang jatuh ke dalam setiap bin.
    - Histogram berguna untuk memahami distribusi data, apakah data terdistribusi normal, miring, atau terpusat pada nilai tertentu.
    - Contoh: Menampilkan distribusi harga laptop di seluruh dataset.

    </div>
""", unsafe_allow_html=True)


    
def tentang_kami():
    st.title("About Us")
    st.write("Aplikasi web ini dibuat oleh kelompok 2. Aplikasi ini bertujuan untuk menvisualisasikan dan menganalisis data Harga Laptop berdasarkan beberapa spesifikasi yang diberikan.")

    st.subheader("Tim Kami")

    cols = st.columns(4)

    # Menampilkan gambar tim dalam bentuk lingkaran dengan orientasi yang benar
    for idx, path in enumerate(image_paths):
        image = Image.open(path)
        
        # Memperbaiki orientasi gambar
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # Jika gambar tidak memiliki data EXIF, lewati perbaikan orientasi
            pass
        
        size = (min(image.width, image.height),) * 2
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        image_circle = ImageOps.fit(image, size, centering=(0.5, 0.5))
        image_circle.putalpha(mask)
        with cols[idx]:
            st.image(image_circle, caption=team_names[idx])

    st.subheader("Anggota Kelompok 2")
    st.write("""
    - Bagus Prasetyo (06)
    - Ilham Suryaputra (13)
    - Muhammad Miftahul Karim (19)
    - Putri Arensya Ingke Dinar Lovyta (22)
    """)

    st.subheader("Kontak")
    st.write("""
    - Email : -
    - GitHub : https://github.com/ilhamsuryap/LaptopPrice/blob/main/Laptop.py
    - Web : https://kelompok2-predictionlaptopprice.streamlit.app/
    """)

    st.subheader("Feedback")
    st.write("Seberapa suka anda dengan Aplikasi Web ini ?")

    satisfaction = st.slider("Geser untuk memberi rating pada Aplikasi Web ini.", 0, 100, 50)
    st.write(f"Presentase Kepuasan: {satisfaction}%")

    feedback = st.text_area("Tulis komentar Anda di sini:")
    if st.button("Kirim Feedback"):
        st.write("Terima kasih atas feedback Anda!")

# Menu: Documentation
def dokumentasi():
    st.header("Panduan Penggunaan Aplikasi Prediksi Harga Laptop")
    st.markdown("""
        <div style='text-align: justify;'>
        Selamat datang di Aplikasi Prediksi Harga Laptop! Aplikasi ini dirancang untuk membantu Anda memprediksi harga laptop berdasarkan spesifikasi teknis yang Anda 
        masukkan. Berikut adalah panduan lengkap mengenai cara menggunakan aplikasi ini:
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Navigasi Menu Utama")
    st.markdown("""
        <div style='text-align: justify;'>
        Aplikasi ini memiliki beberapa menu yang dapat diakses melalui sidebar di sebelah kiri. Berikut adalah daftar menu beserta fungsinya:
        <ul>
            <li>Beranda</li>
            <li>Dataset</li>
            <li>Prediksi Harga Laptop</li>
            <li>Visualisasi Data</li>
            <li>Tentang Aplikasi</li>
            <li>Tentang Kami</li>
            <li>Dokumentasi</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Beranda")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Menampilkan informasi umum tentang aplikasi.</li>
            <li>Menampilkan logo dan gambar merek laptop terkenal.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("2. Dataset")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Menampilkan seluruh dataset yang digunakan dalam aplikasi.</li>
            <li>Menyediakan analisis dasar seperti statistik deskriptif, cek data kosong, distribusi harga, dan word cloud untuk merek laptop.</li>
            <li>Default price pada dataset menggunakan mata uang rubel rusia.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("3. Prediksi Harga Laptop")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Memungkinkan pengguna untuk memprediksi harga laptop berdasarkan spesifikasi yang dimasukkan.</li>
            <li>Mendukung dua metode model prediksi: Linear Regression dan Random Forest.</li>
            <li>Menampilkan evaluasi model menggunakan metrik MAE, MSE, dan RMSE.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("4. Visualisasi Data")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Menyediakan berbagai jenis grafik untuk menganalisis dan memahami data lebih mendalam.</li>
            <li>Memungkinkan pengguna untuk memilih jenis grafik dan kolom yang ingin divisualisasikan.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("5. Tentang Aplikasi")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Menjelaskan tujuan dan fitur utama dari aplikasi ini.</li>
            <li>Memberikan pemahaman mengenai metode yang digunakan untuk prediksi harga.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("6. Tentang Kami")
    st.markdown("""
        <div style='text-align: justify;'>
        Fungsi:
        <ul>
            <li>Memperkenalkan tim pengembang aplikasi.</li>
            <li>Menampilkan gambar anggota tim.</li>
            <li>Menyediakan informasi kontak dan feedback.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)



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
menu = st.sidebar.selectbox("Pilih Menu", ["Beranda", "Dataset","Visualisasi Data", "Prediksi Harga Laptop" , "Tentang Aplikasi", "Dokumentasi", "Tentang Kami", ])

# Menampilkan konten berdasarkan menu yang dipilih
if menu == "Beranda":
    show_home()
elif menu == "Dataset":
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


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ExifTags
import streamlit as st

# Path gambar tim dan nama anggota tim
image_paths = ["karim.jpg", "ilham.jpg", "tyo.JPG", "lovvy.JPG"]
team_names = ["Miftahul Karim", "Ilham Surya", "Bagus Prasetyo", "Putri Arensya"]


image = Image.open('laptop.png')
st.set_page_config(page_title="LaptopPrice", page_icon=image)# Judul aplikasi web

st.markdown(
    """
    <style>
        /* Mengubah background pada kontainer utama Streamlit */
        .stApp {
            background: linear-gradient(to bottom,  #ADD8E6, #FFFFFF);
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 4])
with col1:
    image = Image.open('blogger.png')
    st.image(image, use_container_width=True)
with col2:
    st.title("Aplikasi Prediksi Harga Laptop")

# Menu sidebar untuk memilih antara Overview, Data Visualization, atau Prediction
image = Image.open('laptop.png')
st.sidebar.image(image)
menu = st.sidebar.selectbox("Menu", ["Overview", "Data Visualization", "Prediction", "About Us"])

st.markdown(
    """
    <style>
        /* Mengubah background pada kontainer utama Streamlit */
        .stSidebar {
            background: linear-gradient(to bottom,  #ADD8E6, #FFFFFF);
            border: 2px solid silver;   /* Border putih */
            padding: 10px;            /* Spasi di dalam sidebar */
            border-radius: 10px;      /* Membuat sudut border melengkung */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fungsi untuk memuat dataset dan menyimpannya di cache
@st.cache_data
def load_data():
    # Membaca dataset dari file CSV
    df = pd.read_csv("LaptopPrice.csv")
    return df

# Memuat data ke dalam dataframe
df = load_data()

# Menu: Overview - Menampilkan gambaran umum dataset
if menu == "Overview":
    st.subheader("Gambaran Dataset")
    st.write(f"Total baris dalam dataset: {df.shape[0]}")  # Menampilkan total baris dalam dataset
    st.dataframe(df)  # Menampilkan seluruh dataset dalam format interaktif
    st.write("Shape dari dataset:", df.shape)  # Menampilkan dimensi dataset (baris dan kolom)


    import streamlit as st

    st.title("Overview Aplikasi Prediksi Harga Laptop")

    st.subheader("1. Tentang Aplikasi")
    st.write("""
    Aplikasi ini dirancang untuk membantu pengguna memprediksi harga laptop berdasarkan spesifikasi teknisnya menggunakan algoritma Machine Learning. 
    Selain itu, aplikasi menyediakan visualisasi data interaktif untuk memahami pola harga yang ada dalam dataset.
    """)

    st.subheader("2. Teknologi Machine Learning yang Digunakan")
    st.write("""
    Aplikasi ini menggunakan algoritma:
    - **Linear Regression**: Untuk memprediksi harga laptop berdasarkan data yang dimasukkan oleh pengguna.
    - Model dilatih menggunakan dataset laptop untuk memahami hubungan antara spesifikasi teknis dan harga.
    """)

    st.subheader("3. Jenis Grafik yang Tersedia")
    st.write("""
    Aplikasi ini menyediakan berbagai jenis visualisasi untuk membantu analisis data:
    - **Barplot**: Untuk membandingkan nilai antar fitur.
    - **Lineplot**: Untuk melihat tren data secara kronologis.
    - **Scatterplot**: Untuk menganalisis hubungan antar variabel.
    - **Histogram**: Untuk melihat distribusi data.
    - **Boxplot**: Untuk menganalisis persebaran data serta mengidentifikasi outlier.
    """)

    st.subheader("4. Cara Mengoperasikan Aplikasi")
    st.write("""
    Langkah-langkah menggunakan aplikasi ini:
    1. Masukkan spesifikasi laptop (misalnya: kecepatan prosesor, RAM, penyimpanan).
    2. Klik tombol prediksi untuk mendapatkan estimasi harga laptop.
    3. Gunakan fitur visualisasi untuk memahami pola data secara interaktif.
    """)

    st.subheader("5. Fitur Aplikasi")
    st.write("""
    Aplikasi ini memiliki beberapa fitur utama:
    - **Prediksi Harga Laptop**: Memberikan estimasi harga berdasarkan input spesifikasi.
    - **Visualisasi Data**: Menyediakan grafik interaktif untuk memahami pola dan hubungan antar variabel.
    - **Deskripsi Dataset**: Menampilkan informasi tentang dataset yang digunakan.
    - **Antarmuka yang Mudah Digunakan**: Dibangun menggunakan Streamlit untuk pengalaman pengguna yang intuitif.
    """)

    st.subheader("6. Dataset yang digunakan")
    st.write("""
    Dataset ini menggunakan dataset dari [Kaggle](https://www.kaggle.com/datasets/mrsimple07/laptoppriceprediction).  
    Dataset tersebut dipilih karena:
    - Memuat informasi yang relevan, seperti prosesor, RAM, penyimpanan, dan harga laptop.
    - Cocok untuk analisis pola harga dan membangun model Machine Learning.
    - Memiliki cakupan data sampai dengan 1000 baris data.
    """)

# Menu: Data Visualization - Menampilkan visualisasi data
elif menu == "Data Visualization":
    st.subheader("Visualisasi Data")

    # Pilih kolom untuk sumbu X dan Y untuk visualisasi
    x_col = st.selectbox("Pilih kolom X-axis", df.columns)
    y_col = st.selectbox("Pilih kolom Y-axis", df.columns)

    # Pilih jenis plot yang akan ditampilkan
    plot_type = st.selectbox("Pilih jenis plot", ["Barplot", "Lineplot", "Boxplot", "Scatterplot", "Histogram"])

    # Membuat plot berdasarkan pilihan pengguna
    fig, ax = plt.subplots()
    if plot_type == "Barplot":
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat barplot
        explanation = (f"Barplot ini menunjukkan hubungan antara {x_col} dan {y_col}. "
                       f"Setiap bar mewakili nilai {x_col}, dan tinggi bar menggambarkan nilai rata-rata {y_col}.")
        
    elif plot_type == "Lineplot":
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat lineplot
        explanation = (f"Lineplot ini menggambarkan tren {y_col} terhadap {x_col}. "
                       f"Garis yang terbentuk menunjukkan perubahan nilai {y_col} seiring dengan perubahan {x_col}.")
        
    elif plot_type == "Boxplot":
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat boxplot
        explanation = (f"Boxplot ini digunakan untuk menunjukkan distribusi {y_col} untuk setiap kategori {x_col}. "
                       f"Boxplot membantu untuk mengidentifikasi outlier dan persebaran data.")
        
    elif plot_type == "Scatterplot":
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat scatterplot
        explanation = (f"Scatterplot ini menunjukkan hubungan antara {x_col} dan {y_col}. "
                       f"Setiap titik mewakili pasangan nilai dari kedua variabel tersebut.")
        
    elif plot_type == "Histogram":
        sns.histplot(data=df, x=x_col, ax=ax)  # Membuat histogram
        explanation = (f"Histogram ini menggambarkan distribusi frekuensi nilai {x_col}. S"
                       f"umbu X menunjukkan rentang nilai, dan sumbu Y menunjukkan jumlah data yang jatuh dalam rentang tersebut.")
        

    # Menampilkan plot yang telah dibuat
    st.pyplot(fig)

    # Menampilkan penjelasan untuk grafik yang dipilih
    st.write(explanation)
    


# Menu: Prediction - Menampilkan fitur untuk prediksi harga laptop
elif menu == "Prediction":
    st.subheader("Prediksi Harga Laptop")

    # Mendefinisikan kolom numerik yang akan digunakan sebagai fitur
    numeric_cols = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']
    target_col = 'Price'  # Kolom target untuk prediksi harga laptop

    # Mengkodekan kolom kategorikal (jika ada)
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()  # Membuat objek LabelEncoder
            df[column] = le.fit_transform(df[column])  # Mengkodekan data kategorikal menjadi angka

    # Memisahkan fitur dan target
    X = df[numeric_cols]  # Fitur (kolom numerik)
    y = df[target_col]  # Target (harga laptop)

    # Membagi data menjadi data latih (train) dan data uji (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)  # Melatih model dengan data latih (tanpa scaling)

    # Simpan model hanya sekali (jika belum ada)
    if 'model_prediksi_harga_laptop.sav' not in st.session_state:
        joblib.dump(model, "model_prediksi_harga_laptop.sav")  # Menyimpan model ke file
        st.session_state['model_prediksi_harga_laptop.sav'] = True  # Tandai bahwa model sudah disimpan
        st.write("Model disimpan!")  # Memberikan notifikasi bahwa model telah disimpan

    # Input pengguna untuk spesifikasi laptop yang akan diprediksi
    st.write("Masukkan spesifikasi laptop yang anda inginkan :")
    processor_speed = st.number_input("Kecepatan Prosesor (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    ram_size = st.number_input("Ukuran RAM (GB)", min_value=2, max_value=64, value=8, step=2)
    storage_capacity = st.number_input("Kapasitas Penyimpanan (GB)", min_value=128, max_value=2048, value=512, step=128)
    screen_size = st.number_input("Ukuran Layar (inci)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    # Memuat model yang sudah disimpan
    model = joblib.load("model_prediksi_harga_laptop.sav")

    # Prediksi harga ketika tombol ditekan
    if st.button("Prediksi Harga"):
        # Input pengguna langsung digunakan tanpa skalasi
        user_input = np.array([[processor_speed, ram_size, storage_capacity, screen_size, weight]])  # Menggunakan input pengguna langsung
        predicted_price = model.predict(user_input)  # Melakukan prediksi harga menggunakan model

        st.write(f"Prediksi Harga Laptop: ${predicted_price[0]:,.2f}")  # Menampilkan hasil prediksi harga

        col1, col2 = st.columns([2, 1])
        with col2:
            image = Image.open('data.png')
            st.image(image, use_container_width=True)
            st.write(f"Tabel di samping adalah Tabel prediksi yang digunakan untuk prediksi harga Laptop berdasarkan spesifikasi yang dipilih: kecepatan prosesor {processor_speed} GHz, RAM {ram_size} GB, kapasitas penyimpanan {storage_capacity} GB, ukuran layar {screen_size} inci, dan berat {weight} kg.")
        
        with col1:
            fig, ax = plt.subplots()
            ax.plot(["Prediksi Harga"], predicted_price, marker="o", color="b", label="Harga Prediksi")
            ax.set_ylabel("Harga Laptop ($)")
            ax.set_title("Prediksi Harga Laptop")
            ax.legend()
            st.pyplot(fig)

# Menu: About Us - Menampilkan informasi pengembang
elif menu == "About Us":
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
    - Web : https://kelompok2-laptopprice.streamlit.app/
    """)

    st.subheader("Feedback")
    st.write("Seberapa suka anda dengan Aplikasi Web ini ?")

    satisfaction = st.slider("Geser untuk memberi rating pada Aplikasi Web ini.", 0, 100, 50)
    st.write(f"Presentase Kepuasan: {satisfaction}%")

    feedback = st.text_area("Tulis komentar Anda di sini:")
    if st.button("Kirim Feedback"):
        st.write("Terima kasih atas feedback Anda!")

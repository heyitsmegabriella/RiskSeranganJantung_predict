import streamlit as st
import pandas as pd
import re
import plotly.express as px
from datetime import datetime
import joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================
# LOAD KAMUS & NLP SETUP
# ==========================
@st.cache_resource
def load_nlp_resources():
    kamus = pd.read_csv(
        "kbba.txt",
        sep="\t",
        header=None,
        names=["slang", "formal"]
    )
    kamus_dict = dict(zip(kamus["slang"], kamus["formal"]))
    
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()
    
    return kamus_dict, stopwords, stemmer

kamus_dict, stopwords, stemmer = load_nlp_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(r"#[\w]*", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    tokens = text.split()

    # Normalisasi Slang
    tokens = [kamus_dict.get(kata, kata) for kata in tokens]

    # Hapus Stopwords
    tokens = [kata for kata in tokens if kata not in stopwords]

    # Gabung & Stemming
    text = " ".join(tokens)
    text = stemmer.stem(text)

    return text

# ===========================
# Konfigurasi Halaman
# ===========================
st.set_page_config(page_title="Prediksi Risiko Serangan Jantung", layout="wide")

# ===========================
# Load Dataset & Model
# ===========================
@st.cache_data
def load_dataset():
    return pd.read_csv("data/data_bersih.csv")

@st.cache_resource
def load_models():
    catboost = joblib.load("model/catboost.pkl")
    tfidf = joblib.load("model/tfidf.pkl")
    return catboost, tfidf

df = load_dataset()
catboost, tfidf = load_models()

# ===========================
# Statistik Dataset
# ===========================
total = len(df)
berisiko = len(df[df["label"] == "berisiko"])
tidak = len(df[df["label"] == "tidak berisiko"])

persen_berisiko = round((berisiko / total) * 100, 1) if total > 0 else 0
persen_tidak = round((tidak / total) * 100, 1) if total > 0 else 0

# ===========================
# Header
# ===========================
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("### 👋 Selamat Datang!")
    st.title("Prediksi Risiko Serangan Jantung")

with col2:
    st.info(f"""
📅 {datetime.now().strftime('%d-%m-%Y')}
""")


# ===========================
# Card Statistik
# ===========================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("📂 Total Data", total)
with c2:
    st.metric("❤️ Berisiko", berisiko, f"{persen_berisiko}%")
with c3:
    st.metric("💚 Tidak Berisiko", tidak, f"{persen_tidak}%")

st.divider()

# ===========================
# Kolom Kiri dan Kanan
# ===========================
left, right = st.columns([1.2, 1])

# ==================================
# KIRI: Visualisasi & Evaluasi
# ==================================
with left:
    st.subheader("Distribusi Label Risiko")

    pie_df = pd.DataFrame({
        "Label": ["Berisiko", "Tidak Berisiko"],
        "Jumlah": [berisiko, tidak]
    })

    pie = px.pie(
        pie_df,
        names="Label", 
        values="Jumlah",
        hole=0.55,
        color="Label",
        color_discrete_map={
            "Berisiko": "#ff4b4b",
            "Tidak Berisiko": "#3CCF4E"
        }
    )

    st.plotly_chart(pie, use_container_width=True)
    st.info("Dataset terdiri dari text yang telah melalui proses preprocessing dan pelabelan.")

    st.subheader("Perbandingan Model")
    hasil = pd.DataFrame({
        "Model": ["Naive Bayes", "CatBoost"],
        "Accuracy": [87, 93],
        "Precision": [88, 94],
        "Recall": [97, 97],
        "F1-Score": [92, 96]
    })
    st.dataframe(hasil, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Confusion Matrix")
    st.markdown("#### Naive Bayes")
    cm_nb = pd.DataFrame(
    [[282, 49],
     [26, 311]],
    index=["Aktual Berisiko", "Aktual Tidak Berisiko"],
    columns=["Prediksi Berisiko", "Prediksi Tidak Berisiko"])

    st.dataframe(cm_nb, use_container_width=True)

    st.markdown("#### CatBoost")

    cm_cb = pd.DataFrame(
        [[471, 9],
        [50, 138]],
        index=["Aktual Berisiko", "Aktual Tidak Berisiko"],
        columns=["Prediksi Berisiko", "Prediksi Tidak Berisiko"])

    st.dataframe(cm_cb, use_container_width=True)


# ==================================
# KANAN: Form Prediksi
# ==================================
with right:
    st.subheader("Prediksi Risiko")
    st.write("Masukkan teks untuk memprediksi risiko serangan jantung.")
    st.write("Contoh: dada saya terasa sakit sejak kemarin")

    text = st.text_area(
        "Masukkan Teks",
        height=180,
        placeholder="Ketikkan teks di sini..."
    )

    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        if not text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:

            text_preprocessed = preprocess(text)
            tokens = set(text_preprocessed.split())

            # 2. Cek Kata Kunci Terkait Kesehatan Jantung
            heart_keywords = {
                "jantung", "serang", "hipertensi", "kolesterol", "koroner", "aritmia",
                "angina", "stroke", "sesak", "napas", "nafas", "debar", "nadi", "iskemia", "infark"
            }
            symptom_keywords = {
                "dada", "nyeri", "lemas", "mual", "pusing", "keringat", "sakit"
            }

            medical_count = len(tokens & heart_keywords)
            symptom_count = len(tokens & symptom_keywords)

            if medical_count == 0 and symptom_count < 2:
                st.warning("⚠️ Tweet tidak berkaitan dengan serangan jantung sehingga tidak dapat diprediksi.")
                st.stop()

            education_words = {
                "cara", "cegah", "pencegahan", "tips", "artikel", "studi",
                "penelitian", "faktor", "risiko", "penyebab", "gejala"}

            other_people = {
                "ayah", "ibu", "papah", "mamah", "adik", "kakak", "saudara", "teman", 
                "pacar", "suami", "istri", "pasien", "orang"}

            first_person = {"aku", "saya", "gue", "gw", "ku"}

            override = (
                ("serang" in tokens and "jantung" in tokens)
                and not bool(tokens & education_words) 
                and not bool(tokens & other_people)
                and bool(tokens & first_person) 
            )

            if override:
                    st. subheader("Hasil Prediksi")
                    st.error("❤️ Berisiko Serangan Jantung")

                    st.write("### Probabilitas")
                    st.write(f"Berisiko : **100.00%**")
                    st.progress(1.0)
                    st.write(f"Tidak Berisiko : **0.00%**")
                    st.progress(0.0) 
            else :
                vector = tfidf.transform([text_preprocessed]).toarray()
                prediksi = catboost.predict(vector)[0].lower()

                prob = catboost.predict_proba(vector)[0]
                classes = list(catboost.classes_)
                
                idx_berisiko = classes.index("berisiko")
                idx_tidak = classes.index("tidak berisiko")

                prob_berisiko = prob[idx_berisiko]
                prob_tidak = prob[idx_tidak]

                # Tampilan Hasil Prediksi
                st.subheader("Hasil Prediksi")
                if prediksi == "berisiko":
                    st.error("❤️ Berisiko Serangan Jantung")
                else:
                    st.success("💚 Tidak Berisiko Serangan Jantung")

                st.write("### Probabilitas")
                st.write(f"Berisiko : **{prob_berisiko * 100:.2f}%**")
                st.progress(float(prob_berisiko))

                st.write(f"Tidak Berisiko : **{prob_tidak * 100:.2f}%**")
                st.progress(float(prob_tidak))

                st.subheader("📋 Rekomendasi")

                if prediksi == "berisiko":
                    st.warning("""
                    - Segera konsultasikan keluhan kepada tenaga medis atau dokter.
                    - Jika mengalami nyeri dada hebat, sesak napas, atau keringat dingin secara tiba-tiba, segera menuju fasilitas kesehatan terdekat.
                    - Jangan melakukan diagnosis mandiri hanya berdasarkan hasil sistem.
                    - Hasil ini merupakan prediksi menggunakan model machine learning, bukan diagnosis medis.
                    """)
                else:
                    st.success("""
                    - Saat ini teks tidak terindikasi menunjukkan risiko serangan jantung.
                    - Tetap terapkan pola hidup sehat seperti olahraga teratur, makan bergizi, dan menghindari rokok.
                    - Jika muncul keluhan yang mengarah pada gejala serangan jantung, segera periksakan diri ke tenaga medis.
                    - Hasil ini merupakan prediksi menggunakan model machine learning, bukan diagnosis medis.
                    """)
                
    else:
        st.info("Prediksi ini menggunakan Algoritma CatBoost!")
        st.info("Sistem ini dirancang untuk menganalisis teks yang berkaitan dengan faktor risiko serangan jantung. Hasil prediksi pada teks di luar konteks kesehatan, seperti lirik lagu, puisi, atau ungkapan kiasan, mungkin tidak akurat.")
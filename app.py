
import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier


# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Prediksi Stres Akademik Mahasiswa",
    page_icon="🌿",
    layout="centered"
)


# ==================================================
# 2. DIREKTORI DAN LOAD MODEL
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ASSET_DIR = os.path.join(BASE_DIR, "assets")


@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(
        os.path.join(MODEL_DIR, "model_xgboost_aco.json")
    )

    ordinal_pre = joblib.load(
        os.path.join(MODEL_DIR, "ordinal_pre.pkl")
    )

    onehot_post = joblib.load(
        os.path.join(MODEL_DIR, "onehot_post.pkl")
    )

    return model, ordinal_pre, onehot_post


model, ordinal_pre, onehot_post = load_model()


# ==================================================
# 3. URUTAN FITUR
# ==================================================

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Study_Hours",
    "Class_Attendance",
    "Tuition",
    "Exam_Frequency",
    "Assignment_Load",
    "Sleep_Hours",
    "Physical_Exercise",
    "Social_Media_Use",
    "Screen_Time",
    "Family_Income_Level",
    "Peer_Pressure",
    "Family_Support",
    "Anxiety_Level",
    "University_Type"
]


NUMERIC_FEATURES = [
    "Age",
    "Study_Hours",
    "Class_Attendance",
    "Exam_Frequency",
    "Assignment_Load",
    "Sleep_Hours",
    "Social_Media_Use",
    "Screen_Time",
    "Peer_Pressure",
    "Family_Support",
    "Anxiety_Level"
]



# ==================================================
# 4. DAFTAR PERTANYAAN
# ==================================================

questions = [

    {
        "feature": "Age",
        "question": "Berapa usia kamu saat ini?",
        "type": "select",
        "options": ["Pilih usia"] + [
            str(i) for i in range(19, 25)
        ],
        "help": "Pilih usia kamu saat ini dalam tahun."
    },

    {
        "feature": "Gender",
        "question": "Apa jenis kelamin kamu?",
        "type": "select",
        "options": [
            "Pilih jenis kelamin",
            "Male",
            "Female"
        ],
        "labels": {
            "Male": "Laki-laki",
            "Female": "Perempuan"
        }
    },

    {
        "feature": "Study_Hours",
        "question": "Berapa lama rata-rata kamu belajar dalam sehari?",
        "type": "select",
        "options": ["Pilih durasi belajar"] + [
            str(i) for i in range(0, 10)
        ],
        "help": "Pilih durasi belajar dalam jam per hari."
    },

    {
        "feature": "Class_Attendance",
        "question": "Berapa persentase kehadiran kamu di kelas?",
        "type": "select",
        "options": ["Pilih persentase kehadiran"] + [
            str(i) for i in range(40, 100)
        ],
        "help": "Masukkan persentase kehadiran dalam satuan persen."
    },

    {
        "feature": "Tuition",
        "question": "Apakah kamu mengikuti les atau bimbingan belajar tambahan?",
        "type": "select",
        "options": [
            "Pilih jawaban",
            "Yes",
            "No"
        ],
        "labels": {
            "Yes": "Ya",
            "No": "Tidak"
        }
    },

    {
        "feature": "Exam_Frequency",
        "question": "Bagaimana frekuensi ujian atau kuis yang kamu hadapi?",
        "type": "scale",
        "help": "Pilih nilai sesuai kondisi kamu pada skala 1–9."
    },

    {
        "feature": "Assignment_Load",
        "question": "Seberapa besar beban tugas kuliah yang kamu miliki?",
        "type": "scale",
        "help": "Pilih nilai sesuai kondisi kamu pada skala 1–9."
    },

    {
        "feature": "Sleep_Hours",
        "question": "Berapa lama rata-rata kamu tidur dalam sehari?",
        "type": "select",
        "options": ["Pilih durasi tidur"] + [
            str(i) for i in range(4, 10)
        ],
        "help": "Pilih durasi tidur dalam jam per hari."
    },

    {
        "feature": "Physical_Exercise",
        "question": "Apakah kamu melakukan aktivitas fisik atau olahraga?",
        "type": "select",
        "options": [
            "Pilih jawaban",
            "Yes",
            "No"
        ],
        "labels": {
            "Yes": "Ya",
            "No": "Tidak"
        }
    },

    {
        "feature": "Social_Media_Use",
        "question": "Berapa lama rata-rata kamu menggunakan media sosial dalam sehari?",
        "type": "select",
        "options": ["Pilih durasi penggunaan media sosial"] + [
            str(i) for i in range(0, 8)
        ],
        "help": "Pilih durasi penggunaan media sosial dalam jam per hari."
    },

    {
        "feature": "Screen_Time",
        "question": "Berapa lama total waktu penggunaan perangkat layar kamu dalam sehari?",
        "type": "select",
        "options": ["Pilih durasi penggunaan perangkat"] + [
            str(i) for i in range(1, 12)
        ],
        "help": "Hitung penggunaan HP, laptop, dan perangkat layar lainnya dalam jam per hari."
    },

    {
        "feature": "Family_Income_Level",
        "question": "Bagaimana tingkat pendapatan keluarga kamu?",
        "type": "select",
        "options": [
            "Pilih tingkat pendapatan",
            "Low",
            "Medium",
            "High"
        ],
        "labels": {
            "Low": "Rendah",
            "Medium": "Menengah",
            "High": "Tinggi"
        }
    },

    {
        "feature": "Peer_Pressure",
        "question": "Seberapa besar tekanan dari teman sebaya yang kamu rasakan?",
        "type": "scale",
        "help": "Pilih nilai sesuai kondisi kamu pada skala 1–9."
    },

    {
        "feature": "Family_Support",
        "question": "Seberapa besar dukungan yang kamu terima dari keluarga?",
        "type": "scale",
        "help": "Pilih nilai sesuai kondisi kamu pada skala 1–9."
    },

    {
        "feature": "Anxiety_Level",
        "question": "Seberapa tinggi tingkat kecemasan yang kamu rasakan?",
        "type": "scale",
        "help": "Pilih nilai sesuai kondisi kamu pada skala 1–9."
    },

    {
        "feature": "University_Type",
        "question": "Apa jenis perguruan tinggi tempat kamu kuliah?",
        "type": "select",
        "options": [
            "Pilih jenis perguruan tinggi",
            "National University",
            "Private University",
            "Public University"
        ],
        "labels": {
            "National University": "Universitas Nasional",
            "Private University": "Universitas Swasta",
            "Public University": "Universitas Negeri"
        }
    }

]


# ==================================================
# 5. SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# ==================================================
# 6. CSS / TAMPILAN
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F8F6;
    color: #263B35;
}

.block-container {
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #263B35;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #247A68;
    text-align: center;
}

.description {
    font-size: 16px;
    color: #52665E;
    text-align: center;
    margin-top: 12px;
}

.question-card {
    background-color: #FFFFFF;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #DCE8E2;
    margin-bottom: 20px;
}

.question-card h3 {
    color: #263B35;
    font-size: 22px;
    margin: 0;
}

div.stButton > button {
    background-color: #247A68;
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    min-height: 45px;
}

div.stButton > button:hover {
    background-color: #195E50;
    color: #FFFFFF;
    border: none;
}

div.stButton > button p {
    color: #FFFFFF;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #247A68;
}

div[data-testid="stSelectbox"] label {
    color: #263B35;
}

.result-card {
    background-color: #FFFFFF;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #DCE8E2;
    text-align: center;
}

.result-title {
    color: #247A68;
    font-size: 28px;
    font-weight: 700;
}

div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 7. FUNGSI PREDIKSI
# ==================================================

def predict_stress(answers):

    input_answers = answers.copy()

    # Konversi fitur numerik dari string ke integer
    for feature in NUMERIC_FEATURES:
        input_answers[feature] = int(input_answers[feature])

    # Menyusun input sesuai urutan fitur model
    input_data = pd.DataFrame(
        [input_answers],
        columns=FEATURE_ORDER
    )

    # Preprocessing
    data_ordinal = ordinal_pre.transform(input_data)

    data_final = onehot_post.transform(data_ordinal)

    # Prediksi
    prediction = int(model.predict(data_final)[0])

    probabilities = model.predict_proba(data_final)[0]

    label_mapping = {
        0: "Rendah",
        1: "Sedang",
        2: "Tinggi"
    }

    return label_mapping[prediction], probabilities


# ==================================================
# 8. FUNGSI VALIDASI JAWABAN
# ==================================================

def is_answer_valid(answer):

    if not isinstance(answer, str):
        return False

    if answer.startswith("Pilih"):
        return False

    return True


# ==================================================
# 9. HALAMAN BERANDA
# ==================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">Prediksi Stres Akademik Mahasiswa</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
            Kenali tingkat stres akademik berdasarkan kondisi
            dan pengalaman perkuliahan kamu.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    foto_path = os.path.join(
        ASSET_DIR,
        "foto_beranda.png"
    )

    if os.path.exists(foto_path):
        st.image(
            foto_path,
            use_container_width=True
        )

    st.write("")

    st.markdown("### Tentang Tes")

    st.write(
        """
        Tes ini digunakan untuk memprediksi tingkat stres akademik
        mahasiswa berdasarkan beberapa faktor yang berkaitan
        dengan aktivitas perkuliahan dan kondisi pribadi.
        """
    )

    st.markdown("""
    <div style="
        background-color:  #808080;
        color: #FFFFFF;
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 600;
        text-align: center;
    ">
        📝 Tes ini terdiri dari 16 pertanyaan.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button(
        "Mulai Tes",
        use_container_width=True
    ):
        st.markdown("""
        <style>
            div[data-testid="stAlert"][kind="success"] {
            background-color: #247A68 !important;
            border: none !important;
            }

        div[data-testid="stAlert"][kind="success"] p,
        div[data-testid="stAlert"][kind="success"] span,
        div[data-testid="stAlert"][kind="success"] div {
            color: #FFFFFF !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.session_state.answers = {}
        st.session_state.current_question = 0
        st.session_state.prediction = None
        st.session_state.probabilities = None
        st.session_state.page = "quiz"

        st.rerun()


# ==================================================
# 10. HALAMAN KUESIONER
# ==================================================

elif st.session_state.page == "quiz":

    index = st.session_state.current_question

    question = questions[index]

    total = len(questions)

    feature = question["feature"]

    st.markdown(
        '<div class="main-title">Kuesioner Stres Akademik</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.caption(
        f"Pertanyaan {index + 1} dari {total}"
    )

    st.progress(
        (index + 1) / total
    )

    st.write("")

    st.markdown(
        f"""
        <div class="question-card">
            <h3>{question["question"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    widget_key = f"question_v3_{index}"

    # Mengambil jawaban sebelumnya
    previous_answer = st.session_state.answers.get(
        feature,
        question["options"][0]
        if question["type"] == "select"
        else "Pilih skala"
    )

    # ----------------------------------------------
    # DROPDOWN KATEGORI DAN NUMERIK
    # ----------------------------------------------

    if question["type"] == "select":

        options = question["options"]

        selected_index = (
            options.index(previous_answer)
            if previous_answer in options
            else 0
        )

        answer = st.selectbox(
            "Pilih jawaban",
            options=options,
            index=selected_index,
            format_func=lambda x: question.get(
                "labels", {}
            ).get(x, x),
            key=widget_key
        )

    # ----------------------------------------------
    # DROPDOWN SKALA 1-9
    # ----------------------------------------------

    elif question["type"] == "scale":

        scale_options = ["Pilih skala"] + [
            str(i) for i in range(1, 10)
        ]

        selected_index = (
            scale_options.index(previous_answer)
            if previous_answer in scale_options
            else 0
        )

        answer = st.selectbox(
            "Pilih skala",
            options=scale_options,
            index=selected_index,
            key=widget_key
        )

    if question.get("help"):
        st.caption(question["help"])

    # Simpan jawaban
    st.session_state.answers[feature] = answer

    st.write("")

    # ----------------------------------------------
    # TOMBOL NAVIGASI
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if index > 0:

            if st.button(
                "Sebelumnya",
                use_container_width=True
            ):

                st.session_state.current_question -= 1
                st.rerun()

        else:

            if st.button(
                "Beranda",
                use_container_width=True
            ):

                st.session_state.page = "home"
                st.rerun()

    with col2:

        if index < total - 1:

            if st.button(
                "Selanjutnya",
                use_container_width=True
            ):

                if not is_answer_valid(answer):

                    st.warning(
                        "Silakan pilih jawaban terlebih dahulu."
                    )

                else:

                    st.session_state.current_question += 1
                    st.rerun()

        else:

            if st.button(
                "Lihat Hasil Prediksi",
                use_container_width=True
            ):

                all_answers_valid = all(
                    is_answer_valid(
                        st.session_state.answers.get(
                            item["feature"], ""
                        )
                    )
                    for item in questions
                )

                if not all_answers_valid:

                    st.warning(
                        "Silakan lengkapi semua pertanyaan."
                    )

                else:

                    try:

                        label, probabilities = predict_stress(
                            st.session_state.answers
                        )

                        st.session_state.prediction = label
                        st.session_state.probabilities = probabilities
                        st.session_state.page = "result"

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Terjadi kesalahan saat melakukan prediksi: {e}"
                        )


# ==================================================
# 11. HALAMAN HASIL PREDIKSI
# ==================================================

elif st.session_state.page == "result":

    st.markdown(
        '<div class="main-title">Hasil Prediksi</div>',
        unsafe_allow_html=True
    )

    st.write("")

    prediction = st.session_state.prediction

    probabilities = st.session_state.probabilities

    st.success(
        "Prediksi berhasil dilakukan."
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">
                Tingkat Stres Akademik
            </div>
            <h1>{prediction}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("### Probabilitas Prediksi")

    class_labels = [
        "Rendah",
        "Sedang",
        "Tinggi"
    ]

    for label, probability in zip(
        class_labels,
        probabilities
    ):

        percentage = float(probability) * 100

        st.write(
            f"**{label}**: {percentage:.2f}%"
        )

        st.progress(
            float(probability)
        )

    st.write("")

    st.caption(
        """
        Hasil ini merupakan prediksi berdasarkan pola data
        yang dipelajari oleh model machine learning.
        """
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Kembali ke Beranda",
            use_container_width=True
        ):

            st.session_state.page = "home"
            st.rerun()

    with col2:

        if st.button(
            "Ulangi Tes",
            use_container_width=True
        ):

            st.session_state.answers = {}
            st.session_state.current_question = 0
            st.session_state.prediction = None
            st.session_state.probabilities = None
            st.session_state.page = "quiz"

            st.rerun()
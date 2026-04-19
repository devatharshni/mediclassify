# ✅ FIXED: Direct imports (NO subprocess / NO pip install)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import streamlit as st
import re
import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MediClassify", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

# ---------------- SESSION STATE ----------------
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'page' not in st.session_state: st.session_state.page = "login"
if 'history' not in st.session_state: st.session_state.history = []
if 'counts' not in st.session_state: st.session_state.counts = {"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}

USERS = {"admin":"admin123","doctor":"medi2024","student":"project123"}

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    data = [
        ("X-ray shows fracture in the left femur bone","Radiology"),
        ("MRI scan reveals herniated disc in lumbar region","Radiology"),
        ("CT scan of chest shows no signs of pneumonia","Radiology"),
        ("Ultrasound of abdomen shows gallstones present","Radiology"),
        ("MRI of brain shows tumor in temporal lobe","Radiology"),
        ("PET scan reveals metastatic activity in lymph nodes","Radiology"),

        ("Blood test shows high glucose level indicating diabetes","Lab Report"),
        ("Hemoglobin level is low patient has anemia","Lab Report"),
        ("White blood cell count is elevated possible infection","Lab Report"),
        ("Urine test shows presence of protein and bacteria","Lab Report"),

        ("ECG shows irregular heartbeat and atrial fibrillation","Cardiology"),
        ("Patient has high blood pressure and chest pain","Cardiology"),
        ("Coronary angiogram shows 70 percent blockage","Cardiology"),

        ("Patient complains of fever cough and body pain","Clinical Notes"),
        ("Patient has headache and vomiting since morning","Clinical Notes"),
        ("Patient prescribed antibiotics for throat infection","Clinical Notes"),
    ]

    texts, labels = zip(*data)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', MultinomialNB())
    ])

    model.fit(texts, labels)
    return model

model = train_model()

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def classify_report(text):
    cleaned = clean_text(text)
    category = model.predict([cleaned])[0]
    confidence = max(model.predict_proba([cleaned])[0]) * 100
    return category, round(confidence, 2)

# ---------------- UI STYLES ----------------
st.markdown("""
<style>
body {background-color:#16144a;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def show_login():
    st.title("🏥 MediClassify Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "dashboard"
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
def show_dashboard():
    st.title("🏥 MediClassify Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.text_input("Age")
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

    with col2:
        doctor = st.text_input("Doctor Name")
        date = st.date_input("Visit Date")

    report = st.text_area("Enter Medical Report")

    if st.button("Classify Report"):
        if report.strip():
            category, confidence = classify_report(report)

            st.success(f"Category: {category}")
            st.info(f"Confidence: {confidence}%")

            st.session_state.counts[category] += 1

            st.session_state.history.insert(0, {
                "name": name or "Unknown",
                "report": report[:60],
                "category": category,
                "confidence": confidence,
                "time": datetime.datetime.now().strftime("%H:%M")
            })
        else:
            st.error("Enter report first")

    # ---------------- STATS ----------------
    st.subheader("📊 Statistics")
    st.write(st.session_state.counts)

    # ---------------- HISTORY ----------------
    st.subheader("📜 History")
    for h in st.session_state.history:
        st.write(f"{h['name']} → {h['category']} ({h['confidence']}%) at {h['time']}")

# ---------------- ROUTER ----------------
if not st.session_state.logged_in:
    show_login()
else:
    show_dashboard()

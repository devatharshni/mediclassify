import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Requirements for OCR and ML
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import easyocr
    from PIL import Image
    import numpy as np
except ImportError:
    install("scikit-learn")
    install("easyocr")
    install("opencv-python-headless") 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import easyocr
    from PIL import Image
    import numpy as np

import streamlit as st
import re
import datetime

st.set_page_config(page_title="MediClassify", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

# Initialize OCR Reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'page' not in st.session_state: st.session_state.page = "login"
if 'history' not in st.session_state: st.session_state.history = []
if 'counts' not in st.session_state: st.session_state.counts = {"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}

USERS = {"admin":"admin123","doctor":"medi2024","student":"project123"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background-color:#16144a;}
.glass-card{background:#252360;border:1px solid #3d3a8a;border-radius:22px;padding:28px 32px;margin-bottom:20px;}
.sec-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#f5a623;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #3d3a8a;display:flex;align-items:center;gap:8px;}
.sec-title::before{content:'';width:4px;height:15px;background:linear-gradient(180deg,#f5a623,#d4881a);border-radius:2px;}
.page-title{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f0f0ff;margin-bottom:6px;}
.page-sub{font-size:14px;color:#9b98cc;margin-bottom:28px;}
.stat-box{background:#252360;border:1px solid #3d3a8a;border-radius:16px;padding:18px 20px;text-align:center;margin-bottom:16px;}
.stat-num{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#f5a623;}
.stat-lbl{color:#9b98cc;font-size:12px;margin-top:4px;}
.result-rad{background:rgba(77,159,255,0.12);border:1px solid rgba(77,159,255,0.3);border-radius:15px;padding:18px 22px;margin-top:14px;}
.result-lab{background:rgba(0,212,180,0.12);border:1px solid rgba(0,212,180,0.3);border-radius:15px;padding:18px 22px;margin-top:14px;}
.result-card{background:rgba(255,107,157,0.12);border:1px solid rgba(255,107,157,0.3);border-radius:15px;padding:18px 22px;margin-top:14px;}
.result-clin{background:rgba(245,166,35,0.12);border:1px solid rgba(245,166,35,0.3);border-radius:15px;padding:18px 22px;margin-top:14px;}
.result-unk{background:rgba(155,152,204,0.12);border:1px solid rgba(155,152,204,0.3);border-radius:15px;padding:18px 22px;margin-top:14px;}
.result-cat{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;margin:6px 0 4px;}
.result-info{font-size:12px;opacity:0.75;}
.hist-card{background:rgba(255,255,255,0.03);border:1px solid #3d3a8a;border-radius:13px;padding:13px 16px;margin-bottom:10px;}
.hist-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#f0f0ff;}
.hist-meta{font-size:11px;color:#9b98cc;margin:3px 0;}
.hist-rep{font-size:12px;color:rgba(240,240,255,0.7);}
.hist-time{font-size:10px;color:#9b98cc;margin-top:4px;}
.badge{font-size:10px;font-weight:700;padding:4px 13px;border-radius:20px;font-family:'Syne',sans-serif;letter-spacing:0.04em;text-transform:uppercase;}
.b-rad{background:rgba(77,159,255,0.15);color:#4d9fff;border:1px solid rgba(77,159,255,0.3);}
.b-lab{background:rgba(0,212,180,0.15);color:#00d4b4;border:1px solid rgba(0,212,180,0.3);}
.b-card{background:rgba(255,107,157,0.15);color:#ff6b9d;border:1px solid rgba(255,107,157,0.3);}
.b-clin{background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3);}
.feature-card{background:#252360;border:1px solid #3d3a8a;border-radius:18px;padding:24px;text-align:center;}
.feature-icon{font-size:36px;margin-bottom:12px;}
.feature-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:#f5a623;margin-bottom:8px;}
.feature-desc{font-size:13px;color:#9b98cc;line-height:1.6;}
.contact-info-card{background:#252360;border:1px solid #3d3a8a;border-radius:18px;padding:22px;margin-bottom:14px;display:flex;align-items:center;gap:16px;}
.contact-icon{font-size:28px;}
.contact-label{font-size:12px;color:#9b98cc;}
.contact-value{font-size:14px;color:#f0f0ff;font-weight:500;margin-top:2px;}
.flow-step{background:#252360;border:1px solid #3d3a8a;border-radius:11px;padding:11px 15px;margin:5px 0;font-size:13px;color:#f0f0ff;display:flex;align-items:center;gap:10px;}
.footer-bar{text-align:center;margin-top:30px;padding:16px;border-top:1px solid #3d3a8a;font-size:12px;color:#9b98cc;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background-color:rgba(255,255,255,0.05)!important;border:1px solid #3d3a8a!important;border-radius:11px!important;color:#f0f0ff!important;}
.stButton>button{background:linear-gradient(135deg,#f5a623,#d4881a)!important;color:#1e1c4a!important;border:none!important;border-radius:12px!important;font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:14px!important;padding:11px 24px!important;width:100%!important;box-shadow:0 4px 22px rgba(245,166,35,0.35)!important;}
label{color:#9b98cc!important;font-size:12px!important;}
</style>
""", unsafe_allow_html=True)

SHIELD = """<svg width="48" height="54" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

SHIELD_BIG = """<svg width="90" height="100" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg2" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg2)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

@st.cache_resource
def train_model():
    data = [
        ("X-ray shows fracture in the left femur bone","Radiology"),
        ("MRI scan reveals herniated disc in lumbar region","Radiology"),
        ("CT scan of chest shows no signs of pneumonia","Radiology"),
        ("Ultrasound of abdomen shows gallstones present","Radiology"),
        ("X-ray of spine shows mild scoliosis","Radiology"),
        ("MRI of brain shows tumor in temporal lobe","Radiology"),
        ("Mammogram shows dense tissue with calcification","Radiology"),
        ("CT scan abdomen reveals liver lesion","Radiology"),
        ("Ultrasound shows enlarged spleen and kidney stones","Radiology"),
        ("Pet scan reveals metastatic activity in lymph nodes","Radiology"),
        ("Blood test shows high glucose level indicating diabetes","Lab Report"),
        ("Hemoglobin level is low patient has anemia","Lab Report"),
        ("White blood cell count is elevated possible infection","Lab Report"),
        ("Urine test shows presence of protein and bacteria","Lab Report"),
        ("Thyroid function test shows hypothyroidism","Lab Report"),
        ("Liver enzymes are elevated possible hepatitis","Lab Report"),
        ("Platelet count is dangerously low","Lab Report"),
        ("Creatinine levels indicate chronic kidney disease","Lab Report"),
        ("Blood culture shows bacterial infection present","Lab Report"),
        ("Cholesterol panel shows LDL at critical level","Lab Report"),
        ("ECG shows irregular heartbeat and atrial fibrillation","Cardiology"),
        ("Patient has high blood pressure and chest pain","Cardiology"),
        ("Echocardiogram shows reduced ejection fraction","Cardiology"),
        ("Patient reports shortness of breath and palpitations","Cardiology"),
        ("Coronary angiogram shows 70 percent blockage","Cardiology"),
        ("Patient diagnosed with congestive heart failure","Cardiology"),
        ("Hypertension not controlled despite medication","Cardiology"),
        ("Troponin levels elevated after cardiac episode","Cardiology"),
        ("Patient has history of myocardial infarction","Cardiology"),
        ("Heart rate irregular and blood pressure high","Cardiology"),
        ("Patient complains of fever cough and body pain","Clinical Notes"),
        ("Patient has headache and vomiting since morning","Clinical Notes"),
        ("Follow up visit for diabetes management","Clinical Notes"),
        ("Patient prescribed antibiotics for throat infection","Clinical Notes"),
        ("Patient recovering well after surgery","Clinical Notes"),
        ("Child brought in with high fever and skin rash","Clinical Notes"),
        ("Patient has seasonal allergies and runny nose","Clinical Notes"),
        ("Post operative care note for appendix removal","Clinical Notes"),
        ("Patient reports fatigue and loss of appetite","Clinical Notes"),
        ("Routine checkup all vitals normal","Clinical Notes"),
    ]
    texts, labels = zip(*data)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', MultinomialNB())
    ])
    model.fit(texts, labels)
    return model

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def classify_report(model, text):
    cleaned = clean_text(text)
    category = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    return category, round(max(proba) * 100)

cat_css   = {"Radiology":"result-rad","Lab Report":"result-lab","Cardiology":"result-card","Clinical Notes":"result-clin","Unknown":"result-unk"}
badge_css = {"Radiology":"b-rad","Lab Report":"b-lab","Cardiology":"b-card","Clinical Notes":"b-clin","Unknown":"b-unk"}
model = train_model()

def show_navbar():
    col_logo, col_links = st.columns([1, 2])
    with col_logo:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:12px 0;">'
            + SHIELD +
            '<div><div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div>'
            '<div style="font-size:10px;color:#9b98cc;">Diagnose Faster. Treat Better.</div></div></div>',
            unsafe_allow_html=True
        )
    with col_links:
        n1, n2, n3, n4, n5 = st.columns(5)
        with n1:
            if st.button("🏠 Home"): st.session_state.page = "home"; st.rerun()
        with n2:
            if st.button("ℹ️ About"): st.session_state.page = "about"; st.rerun()
        with n3:
            if st.button("📬 Contact"): st.session_state.page = "contact"; st.rerun()
        with n4:
            if st.button("🏥 Dashboard"): st.session_state.page = "dashboard"; st.rerun()
        with n5:
            if st.button("🚪 Logout"): st.session_state.logged_in = False; st.session_state.page = "login"; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #3d3a8a;margin:0 0 24px 0'>", unsafe_allow_html=True)

def show_login():
    st.markdown(
        '<div style="text-align:center;padding:40px 0 10px;">'
        '<div style="display:inline-block;filter:drop-shadow(0 0 30px rgba(245,166,35,0.5))">' + SHIELD_BIG + '</div>'
        '<div style="font-family:Syne,sans-serif;font-size:40px;font-weight:800;color:#f5a623;margin-top:10px;">MEDICLASSIFY</div>'
        '<div style="font-size:15px;color:#9b98cc;font-style:italic;margin-top:6px;">Diagnose Faster. Treat Better.</div></div>',
        unsafe_allow_html=True
    )
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Login to MediClassify</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        login_btn = st.button("🔐  Login")
        st.markdown('</div>', unsafe_allow_html=True)
        if login_btn:
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.page      = "home"
                st.rerun()
            else:
                st.error("❌ Wrong username or password!")

def show_home():
    show_navbar()
    st.markdown('<div style="text-align:center;padding:30px 0 20px;"><div style="display:inline-block;">' + SHIELD_BIG + '</div><div class="page-title">MEDICLASSIFY</div></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    feats = [("🤖","AI Engine"),("⚡","Instant Scan"),("📊","Smart Dashboard")]
    for i, (icon, title) in enumerate(feats):
        with cols[i]:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div></div>', unsafe_allow_html=True)

def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About Project</div><div class="glass-card">MediClassify is an AI-powered medical report classifier built for speed and accuracy.</div>', unsafe_allow_html=True)

def show_contact():
    show_navbar()
    st.markdown('<div class="page-title">Contact</div><div class="glass-card">Email: support@mediclassify.com</div>', unsafe_allow_html=True)

def show_dashboard():
    show_navbar()
    st.markdown('<div class="page-title">Medical Dashboard</div>', unsafe_allow_html=True)
    
    # Stats row
    cols = st.columns(4)
    for i, (cat, count) in enumerate(st.session_state.counts.items()):
        with cols[i]:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{count}</div><div class="stat-lbl">{cat}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")
    
    with left:
        st.markdown('<div class="glass-card"><div class="sec-title">Classify Report</div>', unsafe_allow_html=True)
        p_name = st.text_input("Patient Name")
        
        # --- IMAGE SCANNER ADDITION ---
        uploaded_img = st.file_uploader("📷 Scan Prescription Image", type=['jpg','jpeg','png'])
        ocr_text = ""
        if uploaded_img:
            with st.spinner("Extracting text from image..."):
                img = Image.open(uploaded_img)
                results = reader.readtext(np.array(img), detail=0)
                ocr_text = " ".join(results)
        
        # Pre-fill text area if OCR has results, otherwise normal input
        final_text = st.text_area("Report Content", value=ocr_text, height=150)
        
        if st.button("🚀 Analyze Report"):
            if final_text:
                cat, conf = classify_report(model, final_text)
                st.session_state.counts[cat] += 1
                st.session_state.history.insert(0, {"name": p_name, "cat": cat, "conf": conf, "time": datetime.datetime.now().strftime("%H:%M")})
                st.markdown(f'<div class="{cat_css[cat]}"><div class="result-info">CONFIDENCE: {conf}%</div><div class="result-cat">{cat}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card"><div class="sec-title">Recent History</div>', unsafe_allow_html=True)
        for h in st.session_state.history[:5]:
            st.markdown(f'<div class="hist-card"><div class="hist-name">{h["name"]}</div><div class="badge {badge_css[h["cat"]]}">{h["cat"]}</div><div class="hist-time">{h["time"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Navigation
if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page == "home": show_home()
    elif st.session_state.page == "about": show_about()
    elif st.session_state.page == "contact": show_contact()
    elif st.session_state.page == "dashboard": show_dashboard()

st.markdown('<div class="footer-bar">© 2026 MediClassify AI System</div>', unsafe_allow_html=True)

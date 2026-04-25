import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"])

for pkg in ["scikit-learn","Pillow","pytesseract","plotly","numpy","pandas"]:
    try: __import__(pkg.replace("-","_").split("==")[0])
    except ImportError: install(pkg)

import streamlit as st
import re, datetime, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import pytesseract
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="MediClassify",page_icon="🏥",layout="wide",initial_sidebar_state="collapsed")

if 'logged_in' not in st.session_state: st.session_state.logged_in=False
if 'username' not in st.session_state: st.session_state.username=""
if 'page' not in st.session_state: st.session_state.page="login"
if 'history' not in st.session_state: st.session_state.history=[]
if 'counts' not in st.session_state: st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}

USERS={"admin":"admin123","doctor":"medi2024","student":"project123"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background-color:#16144a !important;}

/* ── LOGIN PAGE ── */
.login-page{background:linear-gradient(135deg,#16144a 0%,#1e1c5e 100%);min-height:100vh;}
.login-card{background:#1e1c5e;border:1.5px solid #4a47a3;border-radius:24px;padding:40px;max-width:460px;margin:0 auto;}
.login-title{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#f5a623;text-align:center;margin-bottom:6px;}
.login-sub{font-size:13px;color:#b0aee8;text-align:center;margin-bottom:24px;}

/* Force dark inputs everywhere */
.stTextInput input{
    background:#1a184e !important;
    color:#ffffff !important;
    border:1.5px solid #4a47a3 !important;
    border-radius:12px !important;
    padding:12px 16px !important;
    font-size:14px !important;
}
.stTextInput input::placeholder{color:#7875b5 !important;}
.stTextInput input:focus{border-color:#f5a623 !important;box-shadow:0 0 0 2px rgba(245,166,35,0.2) !important;}
.stTextInput label{color:#b0aee8 !important;font-size:13px !important;font-weight:500 !important;}
.stTextArea textarea{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stTextArea textarea::placeholder{color:#7875b5 !important;}
.stTextArea label{color:#b0aee8 !important;font-size:13px !important;}
.stSelectbox label{color:#b0aee8 !important;font-size:13px !important;}
.stSelectbox>div>div{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stDateInput label{color:#b0aee8 !important;font-size:13px !important;}
.stDateInput input{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stRadio label{color:#b0aee8 !important;font-size:13px !important;}
.stRadio>div{background:transparent !important;}

/* BUTTON */
.stButton>button{
    background:linear-gradient(135deg,#f5a623,#d4881a) !important;
    color:#1a0f00 !important;border:none !important;
    border-radius:14px !important;
    font-family:'Syne',sans-serif !important;
    font-weight:800 !important;font-size:14px !important;
    padding:12px 24px !important;width:100% !important;
    letter-spacing:0.04em !important;
    box-shadow:0 4px 20px rgba(245,166,35,0.4) !important;
    transition:all 0.2s !important;
}
.stButton>button:hover{box-shadow:0 8px 30px rgba(245,166,35,0.6) !important;transform:translateY(-1px) !important;}

/* CARDS */
.glass-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:22px;padding:26px 30px;margin-bottom:18px;}
.sec-title{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;color:#f5a623;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #3d3a8a;display:flex;align-items:center;gap:8px;}
.sec-title::before{content:'';width:4px;height:14px;background:linear-gradient(180deg,#f5a623,#d4881a);border-radius:2px;flex-shrink:0;}

/* STAT BOX */
.stat-box{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;padding:16px 18px;text-align:center;margin-bottom:14px;transition:border-color 0.2s;}
.stat-box:hover{border-color:#f5a623;}
.stat-num{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f5a623;}
.stat-lbl{color:#8886c8;font-size:12px;margin-top:4px;}

/* RESULT CARD */
.result-wrapper{border-radius:20px;padding:0;margin-top:18px;overflow:hidden;}
.result-header{padding:22px 26px;display:flex;align-items:center;justify-content:space-between;}
.result-category{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;}
.result-icon{font-size:44px;}
.result-body{padding:0 26px 22px;}
.info-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:14px 0;}
.info-box{background:rgba(0,0,0,0.15);border-radius:12px;padding:12px 14px;text-align:center;}
.info-box-label{font-size:10px;opacity:0.7;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;}
.info-box-value{font-size:16px;font-weight:700;font-family:'Syne',sans-serif;}
.sev-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;font-family:'Syne',sans-serif;}
.sev-critical{background:rgba(255,70,70,0.2);color:#ff6b6b;border:1px solid rgba(255,70,70,0.4);}
.sev-moderate{background:rgba(255,179,71,0.2);color:#ffb347;border:1px solid rgba(255,179,71,0.4);}
.sev-normal{background:rgba(0,212,100,0.2);color:#00d464;border:1px solid rgba(0,212,100,0.4);}
.ai-box{background:rgba(0,0,0,0.18);border-radius:14px;padding:14px 16px;margin-top:14px;}
.ai-box-title{font-size:11px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;display:flex;align-items:center;gap:6px;}
.ai-box-text{font-size:13px;color:rgba(255,255,255,0.85);line-height:1.75;}
.dept-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;margin-top:10px;}
.rx-box{background:rgba(0,0,0,0.15);border-radius:12px;padding:10px 14px;margin-top:12px;font-size:13px;border-left:3px solid #f5a623;}

/* HISTORY */
.hist-card{background:rgba(255,255,255,0.03);border:1px solid #3d3a8a;border-radius:14px;padding:14px 16px;margin-bottom:10px;transition:border-color 0.2s;}
.hist-card:hover{border-color:rgba(245,166,35,0.5);}
.hist-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#f0f0ff;}
.hist-meta{font-size:11px;color:#8886c8;margin:3px 0;}
.hist-rep{font-size:12px;color:rgba(240,240,255,0.7);margin:4px 0;}
.hist-rx{font-size:11px;color:#8886c8;font-style:italic;}
.hist-time{font-size:10px;color:#8886c8;margin-top:4px;}
.badge{font-size:10px;font-weight:700;padding:4px 12px;border-radius:20px;font-family:'Syne',sans-serif;text-transform:uppercase;letter-spacing:0.04em;}
.b-rad{background:rgba(77,159,255,0.15);color:#4d9fff;border:1px solid rgba(77,159,255,0.3);}
.b-lab{background:rgba(0,212,180,0.15);color:#00d4b4;border:1px solid rgba(0,212,180,0.3);}
.b-card{background:rgba(255,107,157,0.15);color:#ff6b9d;border:1px solid rgba(255,107,157,0.3);}
.b-clin{background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3);}

/* FEATURE CARD */
.feature-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:18px;padding:22px;text-align:center;transition:border-color 0.2s,transform 0.2s;}
.feature-card:hover{border-color:#f5a623;transform:translateY(-3px);}
.feature-icon{font-size:34px;margin-bottom:10px;}
.feature-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#f5a623;margin-bottom:6px;}
.feature-desc{font-size:12px;color:#8886c8;line-height:1.6;}

.contact-info-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;padding:18px;margin-bottom:12px;display:flex;align-items:center;gap:14px;}
.flow-step{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:11px;padding:10px 14px;margin:4px 0;font-size:13px;color:#e0e0ff;display:flex;align-items:center;gap:10px;}
.footer-bar{text-align:center;margin-top:28px;padding:14px;border-top:1px solid #3d3a8a;font-size:12px;color:#8886c8;}
.divider{height:1px;background:#3d3a8a;margin:14px 0;}
.ocr-box{background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.25);border-radius:14px;padding:14px 16px;margin-top:10px;}
.ocr-label{font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;}
.ocr-text{font-size:13px;color:#e0e0ff;line-height:1.7;}
.pulse{display:inline-block;width:7px;height:7px;background:#f5a623;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.7);}}
.page-title{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#f0f0ff;margin-bottom:6px;}
.page-sub{font-size:14px;color:#8886c8;margin-bottom:24px;}
.accuracy-bar{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:14px;padding:14px 18px;margin-top:14px;}
</style>
""", unsafe_allow_html=True)

SHIELD="""<svg width="46" height="52" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

SHIELD_BIG="""<svg width="88" height="98" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg2" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg2)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

# ── LOAD CSV & TRAIN ─────────────────────────────────────────
@st.cache_resource
def train_model():
    csv_path = "medical_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna()
        texts  = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        texts = [
            "X-ray shows fracture in the left femur bone","MRI scan reveals herniated disc","CT scan of chest shows pneumonia",
            "Ultrasound shows gallstones","X-ray spine shows scoliosis","MRI brain shows tumor","Mammogram shows calcification",
            "Blood test shows high glucose diabetes","Hemoglobin low patient has anemia","White blood cell count elevated infection",
            "Urine test shows protein and bacteria","Thyroid test shows hypothyroidism","Liver enzymes elevated hepatitis","Platelet count dangerously low",
            "ECG shows irregular heartbeat atrial fibrillation","Patient has high blood pressure chest pain","Echocardiogram shows reduced ejection fraction",
            "Shortness of breath palpitations","Coronary angiogram shows blockage","Heart failure diagnosed","Hypertension uncontrolled",
            "Patient has fever cough body pain","Headache and vomiting since morning","Follow up diabetes management",
            "Prescribed antibiotics throat infection","Patient recovering after surgery","Child high fever skin rash","Fatigue loss of appetite",
        ]
        labels = [
            "Radiology","Radiology","Radiology","Radiology","Radiology","Radiology","Radiology",
            "Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report",
            "Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology",
            "Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes",
        ]

    X_train,X_test,y_train,y_test = train_test_split(texts,labels,test_size=0.2,random_state=42)
    model = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=8000)),('clf',MultinomialNB())])
    model.fit(X_train,y_train)
    acc = round(accuracy_score(y_test, model.predict(X_test))*100,1)
    model.fit(texts,labels)
    return model, acc, len(texts)

def clean_text(t):
    t=t.lower()
    t=re.sub(r'[^a-z0-9\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

def classify_report(model,text):
    cleaned=clean_text(text)
    cat=model.predict([cleaned])[0]
    proba=model.predict_proba([cleaned])[0]
    conf=round(max(proba)*100)
    all_proba={c:round(p*100,1) for c,p in zip(model.classes_,proba)}
    return cat,conf,all_proba

def get_severity(conf):
    if conf>=80: return "Critical","sev-critical","🔴","Immediate attention required"
    elif conf>=55: return "Moderate","sev-moderate","🟡","Monitor closely"
    else: return "Normal","sev-normal","🟢","Routine follow-up"

def get_dept(cat):
    return {"Radiology":"🏥 Radiology Department","Lab Report":"🔬 Pathology Lab","Cardiology":"❤️ Cardiology Department","Clinical Notes":"🩺 General Medicine"}. get(cat,"🏥 General")

def get_icon(cat):
    return {"Radiology":"🩻","Lab Report":"🔬","Cardiology":"❤️","Clinical Notes":"🩺"}.get(cat,"🏥")

def get_color(cat):
    return {"Radiology":"#4d9fff","Lab Report":"#00d4b4","Cardiology":"#ff6b9d","Clinical Notes":"#f5a623"}.get(cat,"#9b98cc")

def get_bg(cat):
    return {"Radiology":"linear-gradient(135deg,rgba(77,159,255,0.18),rgba(77,159,255,0.06))","Lab Report":"linear-gradient(135deg,rgba(0,212,180,0.18),rgba(0,212,180,0.06))","Cardiology":"linear-gradient(135deg,rgba(255,107,157,0.18),rgba(255,107,157,0.06))","Clinical Notes":"linear-gradient(135deg,rgba(245,166,35,0.18),rgba(245,166,35,0.06))"}.get(cat,"rgba(155,152,204,0.12)")

def get_explanation(cat,conf,severity):
    tips={"Radiology":["Consult a radiologist for detailed scan analysis","Imaging studies may need follow-up within 48 hours","Compare with previous scans if available"],"Lab Report":["Review abnormal values with a pathologist","Repeat test after 7 days to confirm result","Dietary changes may be needed based on results"],"Cardiology":["Cardiology consultation strongly recommended","Monitor BP and pulse every 4 hours","Avoid strenuous activity until reviewed"],"Clinical Notes":["Follow prescribed medication schedule","Rest and hydration are essential","Return if symptoms worsen within 24 hours"]}
    desc={"Radiology":f"AI detected imaging-related keywords with {conf}% confidence. The report contains scan, bone, or imaging terminology suggesting a radiological case.","Lab Report":f"AI detected laboratory test keywords with {conf}% confidence. Abnormal blood/urine values were identified requiring pathology review.","Cardiology":f"AI detected cardiac keywords with {conf}% confidence. Heart-related symptoms or ECG findings suggest cardiology involvement.","Clinical Notes":f"AI detected general clinical keywords with {conf}% confidence. Symptoms suggest a general medicine or outpatient case."}
    return desc.get(cat,"AI analyzed the report and found matching medical keywords."), tips.get(cat,[])

def extract_text_from_image(img):
    try: return pytesseract.image_to_string(img,config='--psm 6').strip()
    except: return ""

badge_css={"Radiology":"b-rad","Lab Report":"b-lab","Cardiology":"b-card","Clinical Notes":"b-clin"}
model,model_acc,total_samples=train_model()

# ── NAVBAR ───────────────────────────────────────────────────
def show_navbar():
    c1,c2=st.columns([1,2])
    with c1:
        st.markdown('<div style="display:flex;align-items:center;gap:12px;padding:10px 0;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:19px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:10px;color:#8886c8;">Diagnose Faster. Treat Better.</div></div></div>',unsafe_allow_html=True)
    with c2:
        n1,n2,n3,n4,n5=st.columns(5)
        with n1:
            if st.button("🏠 Home"): st.session_state.page="home"; st.rerun()
        with n2:
            if st.button("ℹ️ About"): st.session_state.page="about"; st.rerun()
        with n3:
            if st.button("📬 Contact"): st.session_state.page="contact"; st.rerun()
        with n4:
            if st.button("🏥 Dashboard"): st.session_state.page="dashboard"; st.rerun()
        with n5:
            if st.button("🚪 Logout"): st.session_state.logged_in=False; st.session_state.page="login"; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #3d3a8a;margin:0 0 20px 0'>",unsafe_allow_html=True)

# ── LOGIN ────────────────────────────────────────────────────
def show_login():
    st.markdown('<div style="text-align:center;padding:36px 0 8px;"><div style="display:inline-block;filter:drop-shadow(0 0 28px rgba(245,166,35,0.55))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:38px;font-weight:800;color:#f5a623;letter-spacing:0.05em;margin-top:8px;">MEDICLASSIFY</div><div style="font-size:14px;color:#b0aee8;font-style:italic;margin-top:5px;">Diagnose Faster. Treat Better.</div></div>',unsafe_allow_html=True)
    _,mid,_=st.columns([1,1.1,1])
    with mid:
        st.markdown("""
        <div style="background:#1e1c5e;border:1.5px solid #4a47a3;border-radius:24px;padding:32px 36px;margin-top:16px;">
        <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:#f5a623;margin-bottom:4px;text-align:center;">🔐 Welcome Back</div>
        <div style="font-size:13px;color:#b0aee8;text-align:center;margin-bottom:20px;">Login to access MediClassify</div>
        </div>""",unsafe_allow_html=True)
        with st.container():
            username=st.text_input("👤  Username",placeholder="Enter your username",key="login_user")
            password=st.text_input("🔑  Password",placeholder="Enter your password",type="password",key="login_pass")
            st.markdown("<br>",unsafe_allow_html=True)
            login_btn=st.button("Login  →")
            st.markdown("""
            <div style="margin-top:16px;padding:14px 16px;background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.25);border-radius:12px;">
            <div style="font-size:11px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Demo Accounts</div>
            <div style="font-size:12px;color:#b0aee8;line-height:2;">
            👤 <strong style="color:#f0f0ff;">admin</strong> / admin123<br>
            👤 <strong style="color:#f0f0ff;">doctor</strong> / medi2024<br>
            👤 <strong style="color:#f0f0ff;">student</strong> / project123
            </div></div>""",unsafe_allow_html=True)
        if login_btn:
            if username in USERS and USERS[username]==password:
                st.session_state.logged_in=True; st.session_state.username=username; st.session_state.page="home"; st.rerun()
            else: st.error("❌ Wrong username or password! Try demo accounts above.")
    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; Placement Mini Project &nbsp;|&nbsp; Python · ML · Streamlit</div>',unsafe_allow_html=True)

# ── HOME ─────────────────────────────────────────────────────
def show_home():
    show_navbar()
    st.markdown('<div style="text-align:center;padding:24px 0 16px;"><div style="display:inline-block;filter:drop-shadow(0 0 22px rgba(245,166,35,0.4))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:42px;font-weight:800;color:#f5a623;margin-top:8px;letter-spacing:0.05em;">MEDICLASSIFY</div><div style="font-size:15px;color:#8886c8;font-style:italic;margin:6px 0 12px;">Diagnose Faster. Treat Better.</div><div style="font-size:14px;color:rgba(240,240,255,0.75);max-width:540px;margin:0 auto;line-height:1.7;">Welcome back, <strong style="color:#f5a623">'+st.session_state.username+'</strong>! 👋<br>AI-powered medical report classification with severity detection.</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # Model stats
    m1,m2,m3=st.columns(3)
    with m1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total_samples)+'</div><div class="stat-lbl">Training samples</div></div>',unsafe_allow_html=True)
    with m2: st.markdown('<div class="stat-box"><div class="stat-num">'+str(model_acc)+'%</div><div class="stat-lbl">Model accuracy</div></div>',unsafe_allow_html=True)
    with m3: st.markdown('<div class="stat-box"><div class="stat-num">4</div><div class="stat-lbl">Categories</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    features=[("🤖","AI Classification","Naive Bayes + TF-IDF ML model"),("🖼️","OCR Support","Reads text from prescription images"),("📊","Visual Output","Charts, severity, AI explanation"),("🔴🟡🟢","Severity Levels","Critical, Moderate, Normal"),("📈","Confidence Chart","All category probabilities"),("📋","History Log","Complete patient classification history")]
    cols=st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:14px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+title+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    _,mid,_=st.columns([1,1,1])
    with mid:
        if st.button("🏥  Open Dashboard →"): st.session_state.page="dashboard"; st.rerun()

# ── ABOUT ────────────────────────────────────────────────────
def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About MediClassify</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Project overview, technologies used, and how it works</div>',unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div class="sec-title">Project overview</div><p style="font-size:14px;color:rgba(220,220,255,0.85);line-height:1.85;"><strong style="color:#f5a623">MediClassify v3.0</strong> is a Placement Mini Project — an AI-powered Medical Report Classification System. It reads medical reports (text or image) and automatically classifies them into the correct department using NLP and Machine Learning. Features include OCR image reading, severity detection, confidence charts, and AI explanations.</p></div>',unsafe_allow_html=True)
    techs=[("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes + TF-IDF"),("📊","Streamlit","Web UI framework"),("🖼️","Tesseract OCR","Image to text"),("📈","Plotly","Interactive charts"),("📁","CSV Dataset","Training data source")]
    cols=st.columns(3)
    for i,(icon,name,desc) in enumerate(techs):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:14px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+name+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="glass-card" style="margin-top:14px;"><div class="sec-title">How it works</div>',unsafe_allow_html=True)
    for i,(icon,label) in enumerate([("📄","Input: Medical report text or prescription image"),("🖼️","OCR: Extract text from uploaded image automatically"),("🔤","NLP: Clean and preprocess text"),("📐","TF-IDF: Convert text to numerical features"),("🤖","ML Model: Naive Bayes predicts category"),("🎯","Output: Category + Severity + Confidence + AI Explanation")]):
        st.markdown('<div class="flow-step"><span style="font-size:18px">'+icon+'</span> <strong style="color:#f5a623">Step '+str(i+1)+':</strong> &nbsp;'+label+'</div>',unsafe_allow_html=True)
        if i<5: st.markdown('<div style="text-align:center;color:#f5a623;font-size:20px;margin:-2px 0">↓</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ── CONTACT ──────────────────────────────────────────────────
def show_contact():
    show_navbar()
    st.markdown('<div class="page-title">Contact Us</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Have a question or feedback? Send us a message!</div>',unsafe_allow_html=True)
    left,right=st.columns([1.2,0.8],gap="large")
    with left:
        st.markdown('<div class="glass-card"><div class="sec-title">Send a message</div>',unsafe_allow_html=True)
        c_name=st.text_input("Your name",placeholder="e.g. Rahul Kumar")
        c_email=st.text_input("Email",placeholder="e.g. rahul@email.com")
        c_subject=st.selectbox("Subject",["","General Inquiry","Technical Support","Feedback","Bug Report"])
        c_msg=st.text_area("Message",placeholder="Write your message here...",height=120)
        if st.button("📨  Send Message"):
            if not c_name or not c_email or not c_msg or not c_subject: st.error("Please fill all fields!")
            elif "@" not in c_email: st.error("Enter a valid email!")
            else: st.success("✅ Thank you "+c_name+"! We will reply to "+c_email+" soon.")
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        for icon,label,value in [("📧","Email","mediclassify@gmail.com"),("📱","Phone","+91 98765 43210"),("📍","Location","Chennai, Tamil Nadu, India"),("🕐","Hours","Mon–Fri, 9 AM – 6 PM")]:
            st.markdown('<div class="contact-info-card"><div style="font-size:26px;">'+icon+'</div><div><div style="font-size:11px;color:#8886c8;">'+label+'</div><div style="font-size:13px;color:#f0f0ff;font-weight:500;margin-top:2px;">'+value+'</div></div></div>',unsafe_allow_html=True)

# ── DASHBOARD ────────────────────────────────────────────────
def show_dashboard():
    show_navbar()
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:20px;padding:20px 28px;margin-bottom:20px;"><div style="display:flex;align-items:center;gap:14px;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:11px;color:#8886c8;">Diagnose Faster. Treat Better. &nbsp;|&nbsp; Model Accuracy: <strong style="color:#f5a623">'+str(model_acc)+'%</strong> &nbsp;|&nbsp; Dataset: <strong style="color:#f5a623">'+str(total_samples)+' samples</strong></div></div></div><div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);color:#f5a623;padding:7px 18px;border-radius:20px;font-size:12px;font-weight:600;"><span class="pulse"></span>AI Active</div></div>',unsafe_allow_html=True)

    total=sum(st.session_state.counts.values())
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total)+'</div><div class="stat-lbl">Total classified</div></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#4d9fff">'+str(st.session_state.counts["Radiology"])+'</div><div class="stat-lbl">Radiology</div></div>',unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#00d4b4">'+str(st.session_state.counts["Lab Report"])+'</div><div class="stat-lbl">Lab Report</div></div>',unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#ff6b9d">'+str(st.session_state.counts["Cardiology"])+'</div><div class="stat-lbl">Cardiology</div></div>',unsafe_allow_html=True)

    left,right=st.columns([1.1,0.9],gap="large")

    with left:
        st.markdown('<div class="glass-card"><div class="sec-title">Patient information</div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: p_name=st.text_input("Patient name",placeholder="e.g. Rahul Kumar")
        with col2: p_age=st.text_input("Age",placeholder="e.g. 35")
        col3,col4=st.columns(2)
        with col3: p_gender=st.selectbox("Gender",["","Male","Female","Other"])
        with col4: p_doc=st.text_input("Doctor name",placeholder="e.g. Dr. Priya")
        p_date=st.date_input("Date of visit")
        st.markdown('<div class="divider"></div><div class="sec-title">Medical report input</div>',unsafe_allow_html=True)
        input_type=st.radio("Choose input",["📝 Type Report","🖼️ Upload Image","📝 + 🖼️ Both"],horizontal=True,label_visibility="collapsed")
        p_report=""; uploaded_img=None; ocr_text=""

        if input_type in ["📝 Type Report","📝 + 🖼️ Both"]:
            p_report=st.text_area("Report / symptoms",placeholder="e.g. MRI scan shows fracture in left leg\nBlood test shows high glucose level\nPatient has chest pain and shortness of breath",height=110)

        if input_type in ["🖼️ Upload Image","📝 + 🖼️ Both"]:
            uploaded_img=st.file_uploader("Upload prescription / report image",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uploaded_img is not None:
                img=Image.open(uploaded_img)
                st.image(img,caption="Uploaded image",use_container_width=True)
                with st.spinner("🔍 Extracting text from image using OCR..."):
                    ocr_text=extract_text_from_image(img)
                if ocr_text.strip():
                    st.markdown('<div class="ocr-box"><div class="ocr-label">📄 OCR extracted text</div><div class="ocr-text">'+ocr_text+'</div></div>',unsafe_allow_html=True)
                    p_report=(p_report+" "+ocr_text).strip() if input_type=="📝 + 🖼️ Both" else ocr_text
                else:
                    st.warning("⚠️ Could not extract text from image. Please type the report manually.")

        st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
        p_rx=st.text_area("Prescription / treatment notes",placeholder="e.g. Paracetamol 500mg twice daily, rest for 5 days",height=72)
        classify_btn=st.button("▶  Classify Report")
        st.markdown('</div>',unsafe_allow_html=True)

        if classify_btn:
            if not p_report.strip():
                st.error("⚠️ Please type a report or upload a readable image!")
            else:
                category,confidence,all_proba=classify_report(model,p_report)
                severity,sev_class,sev_icon,sev_msg=get_severity(confidence)
                explanation,tips=get_explanation(category,confidence,severity)
                dept=get_dept(category)
                icon=get_icon(category)
                color=get_color(category)
                bg=get_bg(category)
                name=p_name.strip() or "Unknown patient"
                meta=" · ".join(filter(None,["Age "+p_age if p_age else "",p_gender,"Dr: "+p_doc if p_doc else ""]))

                # ── RESULT CARD ──────────────────────────────
                rx_section=('<div class="rx-box" style="color:'+color+';"><strong>💊 Prescription:</strong> '+p_rx+'</div>') if p_rx.strip() else ""
                tips_html="".join(['<div style="padding:5px 0;font-size:12px;color:rgba(255,255,255,0.8);">✦ '+t+'</div>' for t in tips])

                st.markdown(
                    '<div class="result-wrapper" style="background:'+bg+';border:1px solid '+color+'40;">'

                    # Header
                    '<div class="result-header" style="border-bottom:1px solid '+color+'30;">'
                    '<div>'
                    '<div style="font-size:11px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Classification result</div>'
                    '<div class="result-category" style="color:'+color+';">'+category+'</div>'
                    '<div style="font-size:13px;color:rgba(255,255,255,0.7);margin-top:4px;">'+name+((" &nbsp;·&nbsp; "+meta) if meta else "")+'</div>'
                    '</div>'
                    '<div style="font-size:52px;filter:drop-shadow(0 0 12px '+color+'60);">'+icon+'</div>'
                    '</div>'

                    # Info grid
                    '<div class="result-body">'
                    '<div class="info-grid">'
                    '<div class="info-box" style="border:1px solid '+color+'30;">'
                    '<div class="info-box-label" style="color:'+color+';">Confidence</div>'
                    '<div class="info-box-value" style="color:'+color+';">'+str(confidence)+'%</div>'
                    '</div>'
                    '<div class="info-box" style="border:1px solid '+color+'30;">'
                    '<div class="info-box-label" style="color:'+color+';">Severity</div>'
                    '<div class="info-box-value" style="color:'+color+';">'+sev_icon+' '+severity+'</div>'
                    '</div>'
                    '<div class="info-box" style="border:1px solid '+color+'30;">'
                    '<div class="info-box-label" style="color:'+color+';">Date</div>'
                    '<div class="info-box-value" style="color:'+color+';font-size:12px;">'+str(p_date)+'</div>'
                    '</div>'
                    '</div>'

                    # Severity + dept
                    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
                    '<span class="sev-pill '+sev_class+'">'+sev_icon+' '+severity+' — '+sev_msg+'</span>'
                    '<span style="background:rgba(255,255,255,0.08);padding:5px 14px;border-radius:20px;font-size:11px;color:rgba(255,255,255,0.8);">'+dept+'</span>'
                    '</div>'

                    # AI explanation
                    '<div class="ai-box">'
                    '<div class="ai-box-title">🤖 AI Explanation</div>'
                    '<div class="ai-box-text">'+explanation+'</div>'
                    '<div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.1);">'
                    '<div style="font-size:11px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">💡 Recommendations</div>'
                    +tips_html+
                    '</div></div>'
                    +rx_section+
                    '</div></div>',
                    unsafe_allow_html=True
                )

                # ── CONFIDENCE BAR CHART ──────────────────────
                st.markdown('<div style="margin-top:18px;">',unsafe_allow_html=True)
                cats=list(all_proba.keys())
                vals=list(all_proba.values())
                bar_colors=["#4d9fff","#00d4b4","#ff6b9d","#f5a623"]
                fig=go.Figure()
                fig.add_trace(go.Bar(x=cats,y=vals,marker_color=bar_colors,text=[str(v)+"%" for v in vals],textposition="outside",textfont=dict(color="white",size=12),marker_line_width=0))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(30,28,94,0.6)",font=dict(color="white",family="DM Sans"),title=dict(text="📊 Confidence distribution across all categories",font=dict(color="#f5a623",size=13)),yaxis=dict(range=[0,115],gridcolor="rgba(255,255,255,0.06)",ticksuffix="%",tickfont=dict(color="#8886c8")),xaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color="white",size=12)),showlegend=False,height=270,margin=dict(t=44,b=16,l=16,r=16))
                st.plotly_chart(fig,use_container_width=True)
                st.markdown('</div>',unsafe_allow_html=True)

                # Update counts + history
                st.session_state.counts[category]+=1
                now=datetime.datetime.now()
                st.session_state.history.insert(0,{"name":name,"age":p_age,"gender":p_gender,"doc":p_doc,"date":str(p_date),"time":now.strftime("%I:%M %p"),"report":p_report[:65]+("..." if len(p_report)>65 else ""),"rx":(p_rx[:50]+"...") if len(p_rx)>50 else p_rx,"cat":category,"conf":confidence,"severity":severity,"sev_icon":sev_icon,"color":color})
                st.rerun()

    with right:
        with st.expander("📊 Project flow",expanded=False):
            for i,(icon,label) in enumerate([("📄","Input report or image"),("🖼️","OCR reads image text"),("🔤","NLP text cleaning"),("📁","CSV dataset loaded"),("🤖","Naive Bayes predicts"),("🎯","Rich output + charts")]):
                st.markdown('<div class="flow-step"><span style="font-size:16px;">'+icon+'</span> '+label+'</div>',unsafe_allow_html=True)
                if i<5: st.markdown('<div style="text-align:center;color:#f5a623;font-size:18px;margin:-2px 0">↓</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:16px">Classification history</div>',unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:32px 16px;color:#8886c8;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:36px;margin-bottom:10px">📋</div><div>No reports classified yet.</div><div style="font-size:12px;margin-top:5px">Fill in details and click Classify.</div></div>',unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:10]:
                meta=" · ".join(filter(None,["Age "+h['age'] if h['age'] else "",h['gender'],"Dr: "+h['doc'] if h['doc'] else ""]))
                bc=badge_css.get(h['cat'],'b-clin')
                color=h.get('color','#f5a623')
                rx_line='<div class="hist-rx">💊 '+h['rx']+'</div>' if h['rx'] else ""
                meta_line='<div class="hist-meta">'+meta+'</div>' if meta else ""
                st.markdown(
                    '<div class="hist-card">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
                    '<span class="hist-name">'+h['name']+'</span>'
                    '<span class="badge '+bc+'">'+h['cat']+'</span></div>'
                    +meta_line+
                    '<div class="hist-rep">'+h['report']+'</div>'
                    +rx_line+
                    '<div class="hist-time">'+h['date']+' at '+h['time']+' &nbsp;|&nbsp; <span style="color:'+color+'">'+str(h['conf'])+'% &nbsp;'+h.get('sev_icon','')+'&nbsp;'+h.get('severity','')+'</span></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            if st.button("🗑️ Clear history"):
                st.session_state.history=[]
                st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}
                st.rerun()

        # PIE CHART
        if sum(st.session_state.counts.values())>0:
            st.markdown('<div class="sec-title" style="margin-top:14px">Distribution</div>',unsafe_allow_html=True)
            labels=[k for k,v in st.session_state.counts.items() if v>0]
            values=[v for v in st.session_state.counts.values() if v>0]
            fig2=go.Figure(go.Pie(labels=labels,values=values,marker=dict(colors=["#4d9fff","#00d4b4","#ff6b9d","#f5a623"]),textfont=dict(color="white",size=11),hole=0.45,hovertemplate="%{label}: %{value} reports<extra></extra>"))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="white"),legend=dict(font=dict(color="#b0aee8",size=11)),height=240,margin=dict(t=8,b=8,l=8,r=8))
            st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project &nbsp;|&nbsp; Python · Scikit-Learn · Streamlit · OCR</div>',unsafe_allow_html=True)

# ── ROUTER ───────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page=="home": show_home()
    elif st.session_state.page=="about": show_about()
    elif st.session_state.page=="contact": show_contact()
    elif st.session_state.page=="dashboard": show_dashboard()
    else: show_home()

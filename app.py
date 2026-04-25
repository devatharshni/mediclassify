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

st.set_page_config(page_title="MediClassify",page_icon="🏥",layout="wide",initial_sidebar_state="collapsed")

if 'logged_in' not in st.session_state: st.session_state.logged_in=False
if 'username' not in st.session_state: st.session_state.username=""
if 'page' not in st.session_state: st.session_state.page="login"
if 'history' not in st.session_state: st.session_state.history=[]
if 'counts' not in st.session_state: st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}

USERS={"admin":"admin123","doctor":"medi2024","student":"project123"}

# ── AUTO PRESCRIPTION DATABASE ───────────────────────────────
PRESCRIPTIONS = {
    "Radiology": {
        "fracture":     {"medicines":["Tab. Ibuprofen 400mg — Twice daily after food","Tab. Calcium + Vitamin D3 — Once daily","Oint. Diclofenac Gel — Apply on affected area 3x daily"],"advice":["Keep the injured area immobilized","Apply ice pack for 20 min every 4 hours","Avoid weight bearing on fracture site","Follow up X-ray after 4 weeks"],"diet":["High calcium foods: milk, yogurt, cheese","Vitamin D: eggs, fish, sunlight exposure","Avoid alcohol and smoking"],"followup":"4 weeks"},
        "tumor":        {"medicines":["Refer to Oncology immediately","Tab. Dexamethasone 4mg — As directed","Tab. Pantoprazole 40mg — Before breakfast"],"advice":["Urgent oncology consultation required","MRI with contrast recommended","Biopsy may be needed for confirmation","Avoid stress and maintain rest"],"diet":["High protein diet","Antioxidant rich foods","Avoid processed foods"],"followup":"1 week — URGENT"},
        "default":      {"medicines":["Tab. Analgesic as prescribed","Tab. Calcium Supplement — Once daily","Topical anti-inflammatory gel — Twice daily"],"advice":["Complete rest for affected area","Follow radiologist recommendations","Repeat imaging if symptoms worsen","Avoid strenuous activity"],"diet":["Balanced nutrition","Adequate hydration","Calcium and Vitamin D rich foods"],"followup":"2 weeks"},
    },
    "Lab Report": {
        "diabetes":     {"medicines":["Tab. Metformin 500mg — Twice daily after food","Tab. Glimepiride 1mg — Before breakfast","Tab. Vitamin B12 500mcg — Once daily"],"advice":["Monitor blood sugar daily (fasting & post-meal)","HbA1c test every 3 months","Regular foot and eye checkup","Exercise 30 min daily"],"diet":["Avoid sugar, white rice, maida","Eat whole grains, vegetables, salads","Small frequent meals (5-6/day)","Avoid fruit juices — eat whole fruits"],"followup":"1 month"},
        "anemia":       {"medicines":["Tab. Ferrous Sulfate 200mg — Twice daily","Tab. Folic Acid 5mg — Once daily","Syp. Iron + Vitamin C — Twice daily"],"advice":["Take iron tablets on empty stomach","Avoid tea/coffee with iron tablets","Check hemoglobin after 4 weeks","Blood transfusion if Hb < 7"],"diet":["Spinach, beetroot, pomegranate","Dates, raisins, jaggery","Vitamin C foods to enhance iron absorption","Avoid tea after meals"],"followup":"4 weeks"},
        "cholesterol":  {"medicines":["Tab. Atorvastatin 10mg — Once daily (night)","Tab. Omega-3 1000mg — Twice daily","Tab. Aspirin 75mg — Once daily after food"],"advice":["Lipid profile test every 3 months","Exercise 45 min daily","Quit smoking and alcohol","Weight management essential"],"diet":["Avoid fried and fatty foods","Eat oats, nuts, olive oil","Fruits: avocado, berries","Avoid egg yolk and red meat"],"followup":"3 months"},
        "default":      {"medicines":["Medicines based on specific lab values","Tab. Multivitamin — Once daily","Tab. Probiotic — Twice daily"],"advice":["Repeat lab test after treatment course","Maintain healthy lifestyle","Stay hydrated — 2-3 litres water/day","Regular health checkups"],"diet":["Balanced diet with all nutrients","Fresh fruits and vegetables","Avoid junk and processed foods","Adequate protein intake"],"followup":"2-4 weeks"},
    },
    "Cardiology": {
        "hypertension": {"medicines":["Tab. Amlodipine 5mg — Once daily (morning)","Tab. Telmisartan 40mg — Once daily","Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily (night)"],"advice":["Monitor BP twice daily — morning & evening","Target BP: below 130/80 mmHg","Avoid salt completely","Avoid stress — practice meditation","Emergency: BP > 180/120 — go to hospital"],"diet":["DASH diet — fruits, vegetables, whole grains","Avoid salt, pickles, papad","Avoid alcohol and smoking","Limit caffeine — max 1 cup/day"],"followup":"2 weeks"},
        "heart":        {"medicines":["Tab. Metoprolol 25mg — Twice daily","Tab. Ramipril 5mg — Once daily","Tab. Furosemide 40mg — Once daily (morning)","Tab. Spironolactone 25mg — Once daily"],"advice":["URGENT — Cardiology OPD visit required","Avoid all physical exertion","Sleep with head elevated 30 degrees","Report any chest pain immediately — EMERGENCY","Daily weight monitoring for fluid retention"],"diet":["Low sodium diet — less than 2g/day","Fluid restriction if advised","Heart healthy: fish, nuts, olive oil","Avoid caffeine and alcohol completely"],"followup":"1 week — URGENT"},
        "default":      {"medicines":["Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily","Tab. Metoprolol 25mg — Twice daily"],"advice":["Cardiology consultation required","ECG and Echo recommended","Avoid strenuous exercise","Report chest pain or breathlessness immediately"],"diet":["Heart healthy diet","Low salt and low fat","Avoid fried and oily foods","Plenty of fruits and vegetables"],"followup":"2 weeks"},
    },
    "Clinical Notes": {
        "fever":        {"medicines":["Tab. Paracetamol 500mg — Three times daily","Tab. Cetirizine 10mg — Once daily (night)","Syp. Benadryl 10ml — Three times daily","ORS Sachet — Dissolve in 1L water, drink throughout day"],"advice":["Complete bed rest for 3-5 days","Tepid sponging if fever > 102°F","Drink 3-4 litres of fluids daily","Return if fever persists > 3 days or worsens"],"diet":["Light diet: khichdi, soup, boiled vegetables","Plenty of fluids: coconut water, ORS, juices","Avoid spicy and oily food","Eat small frequent meals"],"followup":"3-5 days or earlier if worsening"},
        "infection":    {"medicines":["Tab. Amoxicillin 500mg — Three times daily x 5 days","Tab. Ibuprofen 400mg — Twice daily after food","Tab. Probiotic — Twice daily","Tab. Paracetamol 500mg — SOS (if pain/fever)"],"advice":["Complete the full antibiotic course","Do not stop antibiotics midway","Rest and avoid exertion","Maintain good hand hygiene"],"diet":["Immunity boosting foods: turmeric milk, ginger tea","Vitamin C: oranges, lemons, amla","Avoid cold drinks and ice cream","Plenty of warm fluids"],"followup":"5-7 days"},
        "default":      {"medicines":["Tab. Paracetamol 500mg — SOS (if fever/pain)","Tab. Vitamin C 500mg — Once daily","Tab. Zinc 20mg — Once daily","ORS — As needed for hydration"],"advice":["Rest at home for 2-3 days","Monitor symptoms closely","Drink plenty of warm fluids","Return if no improvement in 3 days"],"diet":["Light and easily digestible food","Warm soups and broths","Fresh fruits for immunity","Avoid junk food and cold drinks"],"followup":"3-5 days"},
    }
}

def get_auto_prescription(category, report_text):
    text = report_text.lower()
    db = PRESCRIPTIONS.get(category, PRESCRIPTIONS["Clinical Notes"])
    for key in db:
        if key != "default" and key in text:
            return db[key]
    return db["default"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background-color:#16144a !important;}
.stTextInput input{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;padding:12px 16px !important;font-size:14px !important;}
.stTextInput input::placeholder{color:#7875b5 !important;}
.stTextInput input:focus{border-color:#f5a623 !important;}
.stTextInput label,.stTextArea label,.stSelectbox label,.stDateInput label,.stRadio label{color:#b0aee8 !important;font-size:13px !important;font-weight:500 !important;}
.stTextArea textarea{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stTextArea textarea::placeholder{color:#7875b5 !important;}
.stSelectbox>div>div{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stDateInput input{background:#1a184e !important;color:#ffffff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stButton>button{background:linear-gradient(135deg,#f5a623,#d4881a) !important;color:#1a0f00 !important;border:none !important;border-radius:14px !important;font-family:'Syne',sans-serif !important;font-weight:800 !important;font-size:14px !important;padding:12px 24px !important;width:100% !important;letter-spacing:0.04em !important;box-shadow:0 4px 20px rgba(245,166,35,0.4) !important;}
.stButton>button:hover{box-shadow:0 8px 30px rgba(245,166,35,0.6) !important;transform:translateY(-1px) !important;}
.glass-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:22px;padding:26px 30px;margin-bottom:18px;}
.sec-title{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;color:#f5a623;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #3d3a8a;display:flex;align-items:center;gap:8px;}
.sec-title::before{content:'';width:4px;height:14px;background:linear-gradient(180deg,#f5a623,#d4881a);border-radius:2px;flex-shrink:0;}
.stat-box{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;padding:16px 18px;text-align:center;margin-bottom:14px;}
.stat-num{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f5a623;}
.stat-lbl{color:#8886c8;font-size:12px;margin-top:4px;}
.result-wrapper{border-radius:22px;overflow:hidden;margin-top:18px;}
.result-header{padding:24px 28px 20px;}
.result-body{padding:0 28px 24px;}
.info-trio{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:14px 0;}
.info-box{border-radius:14px;padding:13px 14px;text-align:center;}
.info-lbl{font-size:10px;opacity:0.75;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;}
.info-val{font-size:17px;font-weight:700;font-family:'Syne',sans-serif;}
.sev-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;}
.sev-critical{background:rgba(255,70,70,0.2);color:#ff6b6b;border:1px solid rgba(255,70,70,0.4);}
.sev-moderate{background:rgba(255,179,71,0.2);color:#ffb347;border:1px solid rgba(255,179,71,0.4);}
.sev-normal{background:rgba(0,212,100,0.2);color:#00d464;border:1px solid rgba(0,212,100,0.4);}
.ai-box{border-radius:14px;padding:16px 18px;margin:14px 0;}
.rx-section{border-radius:16px;padding:18px 20px;margin-top:14px;}
.rx-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.rx-item{padding:8px 12px;border-radius:10px;margin-bottom:7px;font-size:13px;display:flex;align-items:flex-start;gap:8px;}
.advice-item{padding:7px 12px;border-radius:10px;margin-bottom:6px;font-size:12px;display:flex;align-items:flex-start;gap:8px;}
.diet-item{padding:7px 12px;border-radius:10px;margin-bottom:6px;font-size:12px;}
.hist-card{background:rgba(255,255,255,0.03);border:1px solid #3d3a8a;border-radius:14px;padding:13px 16px;margin-bottom:10px;transition:border-color 0.2s;}
.hist-card:hover{border-color:rgba(245,166,35,0.4);}
.hist-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#f0f0ff;}
.hist-meta{font-size:11px;color:#8886c8;margin:3px 0;}
.hist-rep{font-size:12px;color:rgba(240,240,255,0.7);margin:4px 0;}
.hist-time{font-size:10px;color:#8886c8;margin-top:4px;}
.badge{font-size:10px;font-weight:700;padding:4px 12px;border-radius:20px;font-family:'Syne',sans-serif;text-transform:uppercase;letter-spacing:0.04em;}
.b-rad{background:rgba(77,159,255,0.15);color:#4d9fff;border:1px solid rgba(77,159,255,0.3);}
.b-lab{background:rgba(0,212,180,0.15);color:#00d4b4;border:1px solid rgba(0,212,180,0.3);}
.b-card{background:rgba(255,107,157,0.15);color:#ff6b9d;border:1px solid rgba(255,107,157,0.3);}
.b-clin{background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3);}
.feature-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:18px;padding:22px;text-align:center;transition:border-color 0.2s,transform 0.2s;}
.feature-card:hover{border-color:#f5a623;transform:translateY(-3px);}
.feature-icon{font-size:34px;margin-bottom:10px;}
.feature-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#f5a623;margin-bottom:6px;}
.feature-desc{font-size:12px;color:#8886c8;line-height:1.6;}
.contact-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;padding:18px;margin-bottom:12px;display:flex;align-items:center;gap:14px;}
.flow-step{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:11px;padding:10px 14px;margin:4px 0;font-size:13px;color:#e0e0ff;display:flex;align-items:center;gap:10px;}
.footer-bar{text-align:center;margin-top:28px;padding:14px;border-top:1px solid #3d3a8a;font-size:12px;color:#8886c8;}
.divider{height:1px;background:#3d3a8a;margin:14px 0;}
.ocr-box{background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.25);border-radius:14px;padding:14px 16px;margin-top:10px;}
.ocr-label{font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;}
.pulse{display:inline-block;width:7px;height:7px;background:#f5a623;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.7);}}
.page-title{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#f0f0ff;margin-bottom:6px;}
.page-sub{font-size:14px;color:#8886c8;margin-bottom:24px;}
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

@st.cache_resource
def train_model():
    csv_path="medical_data.csv"
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path)
        df.columns=[c.strip().lower() for c in df.columns]
        df=df.dropna()
        texts=df['text'].tolist(); labels=df['label'].tolist()
    else:
        texts=["X-ray shows fracture femur","MRI scan herniated disc","CT scan chest pneumonia","Ultrasound gallstones","MRI brain tumor","Blood test high glucose diabetes","Hemoglobin low anemia","White blood cell elevated infection","Thyroid hypothyroidism","Cholesterol LDL critical","ECG irregular heartbeat atrial fibrillation","High blood pressure chest pain","Echocardiogram reduced ejection fraction","Shortness breath palpitations","Heart failure diagnosed","Patient fever cough body pain","Headache vomiting morning","Follow up diabetes management","Antibiotics throat infection","Patient recovering surgery"]
        labels=["Radiology","Radiology","Radiology","Radiology","Radiology","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes"]
    X_train,X_test,y_train,y_test=train_test_split(texts,labels,test_size=0.2,random_state=42)
    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=8000)),('clf',MultinomialNB())])
    model.fit(X_train,y_train)
    acc=round(accuracy_score(y_test,model.predict(X_test))*100,1)
    model.fit(texts,labels)
    return model,acc,len(texts)

def clean_text(t):
    t=t.lower(); t=re.sub(r'[^a-z0-9\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

def classify_report(model,text):
    cleaned=clean_text(text)
    cat=model.predict([cleaned])[0]
    proba=model.predict_proba([cleaned])[0]
    conf=round(max(proba)*100)
    all_proba={c:round(p*100,1) for c,p in zip(model.classes_,proba)}
    return cat,conf,all_proba

def get_severity(conf):
    if conf>=80: return "Critical","sev-critical","🔴","Immediate medical attention required"
    elif conf>=55: return "Moderate","sev-moderate","🟡","Close monitoring recommended"
    else: return "Normal","sev-normal","🟢","Routine follow-up sufficient"

def get_cat_info(cat):
    return {
        "Radiology":     {"icon":"🩻","color":"#4d9fff","bg":"linear-gradient(135deg,rgba(77,159,255,0.2),rgba(77,159,255,0.05))","border":"rgba(77,159,255,0.4)","dept":"Radiology & Imaging Department","bc":"b-rad"},
        "Lab Report":    {"icon":"🔬","color":"#00d4b4","bg":"linear-gradient(135deg,rgba(0,212,180,0.2),rgba(0,212,180,0.05))","border":"rgba(0,212,180,0.4)","dept":"Pathology & Laboratory Department","bc":"b-lab"},
        "Cardiology":    {"icon":"❤️","color":"#ff6b9d","bg":"linear-gradient(135deg,rgba(255,107,157,0.2),rgba(255,107,157,0.05))","border":"rgba(255,107,157,0.4)","dept":"Cardiology Department","bc":"b-card"},
        "Clinical Notes":{"icon":"🩺","color":"#f5a623","bg":"linear-gradient(135deg,rgba(245,166,35,0.2),rgba(245,166,35,0.05))","border":"rgba(245,166,35,0.4)","dept":"General Medicine & OPD","bc":"b-clin"},
    }.get(cat,{"icon":"🏥","color":"#9b98cc","bg":"rgba(155,152,204,0.12)","border":"rgba(155,152,204,0.3)","dept":"General Department","bc":"b-clin"})

def get_explanation(cat,conf):
    return {
        "Radiology":     f"The report contains imaging-related terminology (scan, X-ray, MRI, fracture, bone). AI detected radiological keywords with {conf}% confidence. Immediate radiologist review is {'strongly recommended' if conf>75 else 'suggested'}.",
        "Lab Report":    f"The report contains laboratory test values (blood levels, glucose, cell counts, enzymes). AI detected pathology keywords with {conf}% confidence. {'Abnormal values require urgent attention.' if conf>75 else 'Values should be reviewed by a pathologist.'}",
        "Cardiology":    f"The report contains cardiac terminology (ECG, blood pressure, heart rate, chest pain). AI detected cardiology indicators with {conf}% confidence. {'Urgent cardiology consultation is required.' if conf>75 else 'Cardiologist evaluation recommended.'}",
        "Clinical Notes":f"The report contains general clinical symptoms (fever, pain, cough, infection). AI classified this as a general medical case with {conf}% confidence. {'Active treatment required.' if conf>75 else 'Routine medical care recommended.'}",
    }.get(cat,"AI analyzed the report and matched medical keyword patterns.")

def extract_text_from_image(img):
    try: return pytesseract.image_to_string(img,config='--psm 6').strip()
    except: return ""

model,model_acc,total_samples=train_model()

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

def show_login():
    st.markdown('<div style="text-align:center;padding:36px 0 8px;"><div style="display:inline-block;filter:drop-shadow(0 0 28px rgba(245,166,35,0.55))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:38px;font-weight:800;color:#f5a623;letter-spacing:0.05em;margin-top:8px;">MEDICLASSIFY</div><div style="font-size:14px;color:#b0aee8;font-style:italic;margin-top:5px;">Diagnose Faster. Treat Better.</div></div>',unsafe_allow_html=True)
    _,mid,_=st.columns([1,1.1,1])
    with mid:
        st.markdown('<div style="background:#1e1c5e;border:1.5px solid #4a47a3;border-radius:24px;padding:32px 36px;margin-top:16px;"><div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:#f5a623;margin-bottom:4px;text-align:center;">🔐 Welcome Back</div><div style="font-size:13px;color:#b0aee8;text-align:center;margin-bottom:20px;">Login to access MediClassify</div></div>',unsafe_allow_html=True)
        username=st.text_input("👤  Username",placeholder="Enter your username",key="lu")
        password=st.text_input("🔑  Password",placeholder="Enter your password",type="password",key="lp")
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Login  →"):
            if username in USERS and USERS[username]==password:
                st.session_state.logged_in=True; st.session_state.username=username; st.session_state.page="home"; st.rerun()
            else: st.error("❌ Wrong username or password!")
        st.markdown('<div style="margin-top:16px;padding:14px 16px;background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.25);border-radius:12px;"><div style="font-size:11px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Demo Accounts</div><div style="font-size:12px;color:#b0aee8;line-height:2;">👤 <strong style="color:#f0f0ff;">admin</strong> / admin123<br>👤 <strong style="color:#f0f0ff;">doctor</strong> / medi2024<br>👤 <strong style="color:#f0f0ff;">student</strong> / project123</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; Placement Mini Project</div>',unsafe_allow_html=True)

def show_home():
    show_navbar()
    st.markdown('<div style="text-align:center;padding:24px 0 16px;"><div style="display:inline-block;filter:drop-shadow(0 0 22px rgba(245,166,35,0.4))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:42px;font-weight:800;color:#f5a623;margin-top:8px;letter-spacing:0.05em;">MEDICLASSIFY</div><div style="font-size:15px;color:#8886c8;font-style:italic;margin:6px 0 12px;">Diagnose Faster. Treat Better.</div><div style="font-size:14px;color:rgba(240,240,255,0.75);max-width:540px;margin:0 auto;line-height:1.7;">Welcome back, <strong style="color:#f5a623">'+st.session_state.username+'</strong>! 👋<br>AI-powered medical report classification with auto-prescription.</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    with m1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total_samples)+'</div><div class="stat-lbl">Training samples (CSV)</div></div>',unsafe_allow_html=True)
    with m2: st.markdown('<div class="stat-box"><div class="stat-num">'+str(model_acc)+'%</div><div class="stat-lbl">Model accuracy</div></div>',unsafe_allow_html=True)
    with m3: st.markdown('<div class="stat-box"><div class="stat-num">4</div><div class="stat-lbl">Medical categories</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    features=[("🤖","AI Classification","Naive Bayes + TF-IDF on CSV dataset"),("🖼️","OCR Support","Auto reads prescription images"),("💊","Auto Prescription","Smart medicines based on report"),("🔴🟡🟢","Severity Detection","Critical, Moderate, Normal levels"),("📊","Advanced Charts","Radar, gauge, confidence charts"),("📋","History Log","Complete patient record history")]
    cols=st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:14px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+title+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    _,mid,_=st.columns([1,1,1])
    with mid:
        if st.button("🏥  Open Dashboard →"): st.session_state.page="dashboard"; st.rerun()

def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About MediClassify</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Project overview, technologies, and how it works</div>',unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div class="sec-title">Project overview</div><p style="font-size:14px;color:rgba(220,220,255,0.85);line-height:1.85;"><strong style="color:#f5a623">MediClassify v3.0</strong> is an AI-powered Medical Report Classification System. It reads medical reports (text or image), classifies into the correct department, generates auto prescription, detects severity, and shows advanced visualizations — all powered by NLP and Machine Learning trained on a CSV dataset.</p></div>',unsafe_allow_html=True)
    techs=[("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes + TF-IDF"),("📊","Streamlit","Web UI framework"),("🖼️","Tesseract OCR","Image to text"),("📈","Plotly","Advanced charts"),("📁","CSV Dataset","80+ training samples")]
    cols=st.columns(3)
    for i,(icon,name,desc) in enumerate(techs):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:14px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+name+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)

def show_contact():
    show_navbar()
    st.markdown('<div class="page-title">Contact Us</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Have a question? Send us a message!</div>',unsafe_allow_html=True)
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
            st.markdown('<div class="contact-card"><div style="font-size:26px;">'+icon+'</div><div><div style="font-size:11px;color:#8886c8;">'+label+'</div><div style="font-size:13px;color:#f0f0ff;font-weight:500;margin-top:2px;">'+value+'</div></div></div>',unsafe_allow_html=True)

def show_dashboard():
    show_navbar()
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:20px;padding:20px 28px;margin-bottom:20px;"><div style="display:flex;align-items:center;gap:14px;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:11px;color:#8886c8;">Model Accuracy: <strong style="color:#f5a623">'+str(model_acc)+'%</strong> &nbsp;|&nbsp; CSV Dataset: <strong style="color:#f5a623">'+str(total_samples)+' samples</strong></div></div></div><div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);color:#f5a623;padding:7px 18px;border-radius:20px;font-size:12px;font-weight:600;"><span class="pulse"></span>AI Active</div></div>',unsafe_allow_html=True)

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
        input_type=st.radio("Input",["📝 Type Report","🖼️ Upload Image","📝 + 🖼️ Both"],horizontal=True,label_visibility="collapsed")
        p_report=""; uploaded_img=None

        if input_type in ["📝 Type Report","📝 + 🖼️ Both"]:
            p_report=st.text_area("Report / symptoms",placeholder="e.g. MRI scan shows fracture in left leg\nBlood test shows high glucose level\nPatient has chest pain and shortness of breath",height=110)

        if input_type in ["🖼️ Upload Image","📝 + 🖼️ Both"]:
            uploaded_img=st.file_uploader("Upload prescription image",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uploaded_img is not None:
                img=Image.open(uploaded_img)
                st.image(img,caption="Uploaded image",use_container_width=True)
                with st.spinner("🔍 Reading text from image (OCR)..."):
                    ocr_text=extract_text_from_image(img)
                if ocr_text.strip():
                    st.markdown('<div class="ocr-box"><div class="ocr-label">📄 OCR extracted text</div><div style="font-size:13px;color:#e0e0ff;line-height:1.7;">'+ocr_text+'</div></div>',unsafe_allow_html=True)
                    p_report=(p_report+" "+ocr_text).strip() if input_type=="📝 + 🖼️ Both" else ocr_text
                else:
                    st.warning("⚠️ Could not extract text. Please type the report manually.")

        st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
        classify_btn=st.button("▶  Classify Report & Generate Prescription")
        st.markdown('</div>',unsafe_allow_html=True)

        if classify_btn:
            if not p_report.strip():
                st.error("⚠️ Please enter a report or upload a readable image!")
            else:
                category,confidence,all_proba=classify_report(model,p_report)
                severity,sev_class,sev_icon,sev_msg=get_severity(confidence)
                explanation=get_explanation(category,confidence)
                info=get_cat_info(category)
                rx_data=get_auto_prescription(category,p_report)
                name=p_name.strip() or "Unknown patient"
                meta=" · ".join(filter(None,["Age "+p_age if p_age else "",p_gender,"Dr: "+p_doc if p_doc else ""]))

                # ── RESULT HEADER CARD ────────────────────────
                st.markdown(
                    '<div class="result-wrapper" style="background:'+info["bg"]+';border:1.5px solid '+info["border"]+';margin-top:18px;">'
                    '<div class="result-header" style="border-bottom:1px solid '+info["border"]+';padding:24px 28px;">'
                    '<div style="display:flex;align-items:center;justify-content:space-between;">'
                    '<div>'
                    '<div style="font-size:10px;color:'+info["color"]+';opacity:0.8;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:5px;">✦ AI Classification Result</div>'
                    '<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:'+info["color"]+';">'+category+'</div>'
                    '<div style="font-size:13px;color:rgba(255,255,255,0.7);margin-top:5px;">'+name+((" &nbsp;·&nbsp; "+meta) if meta else "")+'</div>'
                    '<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">'
                    '<span class="sev-pill '+sev_class+'">'+sev_icon+' '+severity+' — '+sev_msg+'</span>'
                    '<span style="background:rgba(255,255,255,0.08);padding:5px 14px;border-radius:20px;font-size:11px;color:rgba(255,255,255,0.8);">🏥 '+info["dept"]+'</span>'
                    '</div></div>'
                    '<div style="font-size:58px;filter:drop-shadow(0 0 14px '+info["color"]+'70);">'+info["icon"]+'</div>'
                    '</div></div>'

                    # INFO TRIO
                    '<div class="result-body">'
                    '<div class="info-trio">'
                    '<div class="info-box" style="background:rgba(0,0,0,0.18);border:1px solid '+info["border"]+';"><div class="info-lbl" style="color:'+info["color"]+';">Confidence</div><div class="info-val" style="color:'+info["color"]+';">'+str(confidence)+'%</div></div>'
                    '<div class="info-box" style="background:rgba(0,0,0,0.18);border:1px solid '+info["border"]+';"><div class="info-lbl" style="color:'+info["color"]+';">Severity</div><div class="info-val" style="color:'+info["color"]+';">'+sev_icon+' '+severity+'</div></div>'
                    '<div class="info-box" style="background:rgba(0,0,0,0.18);border:1px solid '+info["border"]+';"><div class="info-lbl" style="color:'+info["color"]+';">Follow-up</div><div class="info-val" style="color:'+info["color"]+';font-size:13px;">'+rx_data["followup"]+'</div></div>'
                    '</div>'

                    # AI EXPLANATION
                    '<div class="ai-box" style="background:rgba(0,0,0,0.18);border:1px solid '+info["border"]+';"><div style="font-size:11px;color:'+info["color"]+';font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🤖 AI Explanation</div><div style="font-size:13px;color:rgba(255,255,255,0.85);line-height:1.75;">'+explanation+'</div></div>'
                    '</div></div>',
                    unsafe_allow_html=True
                )

                # ── CONFIDENCE GAUGE CHART ────────────────────
                fig_gauge=go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    title={"text":"AI Confidence Score","font":{"color":"#f5a623","size":14}},
                    number={"suffix":"%","font":{"color":info["color"],"size":36}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#8886c8","tickfont":{"color":"#8886c8"}},
                        "bar":{"color":info["color"],"thickness":0.25},
                        "bgcolor":"rgba(30,28,94,0.8)",
                        "bordercolor":info["border"],
                        "steps":[
                            {"range":[0,40],"color":"rgba(0,212,100,0.15)"},
                            {"range":[40,70],"color":"rgba(255,179,71,0.15)"},
                            {"range":[70,100],"color":"rgba(255,70,70,0.15)"},
                        ],
                        "threshold":{"line":{"color":info["color"],"width":3},"thickness":0.75,"value":confidence}
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"white"},height=240,margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig_gauge,use_container_width=True)

                # ── HORIZONTAL BAR CHART ─────────────────────
                cats=list(all_proba.keys())
                vals=list(all_proba.values())
                bar_colors=["#4d9fff","#00d4b4","#ff6b9d","#f5a623"]
                fig_bar=go.Figure(go.Bar(
                    y=cats,x=vals,orientation='h',
                    marker_color=bar_colors,
                    text=[str(v)+"%" for v in vals],
                    textposition="outside",
                    textfont=dict(color="white",size=12),
                    marker_line_width=0
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,28,94,0.5)",
                    font=dict(color="white",family="DM Sans"),
                    title=dict(text="📊 Category confidence breakdown",font=dict(color="#f5a623",size=13)),
                    xaxis=dict(range=[0,115],ticksuffix="%",gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="#8886c8")),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color="white",size=12)),
                    showlegend=False,height=230,
                    margin=dict(t=44,b=16,l=10,r=50)
                )
                st.plotly_chart(fig_bar,use_container_width=True)

                # ── AUTO PRESCRIPTION CARD ────────────────────
                rx_items="".join(['<div class="rx-item" style="background:rgba(0,0,0,0.15);border:1px solid '+info["border"]+';color:rgba(255,255,255,0.9);"><span style="color:'+info["color"]+';font-size:16px;">💊</span><span>'+m+'</span></div>' for m in rx_data["medicines"]])
                advice_items="".join(['<div class="advice-item" style="background:rgba(0,0,0,0.12);color:rgba(255,255,255,0.85);"><span style="color:#f5a623;">✦</span><span>'+a+'</span></div>' for a in rx_data["advice"]])
                diet_items="".join(['<div class="diet-item" style="background:rgba(0,0,0,0.12);color:rgba(255,255,255,0.8);padding:6px 12px;border-radius:8px;margin-bottom:5px;font-size:12px;">🥗 '+d+'</div>' for d in rx_data["diet"]])

                st.markdown(
                    '<div class="rx-section" style="background:'+info["bg"]+';border:1.5px solid '+info["border"]+';margin-top:6px;">'
                    '<div class="rx-title" style="color:'+info["color"]+';">💊 Auto-Generated Prescription<span style="font-size:10px;background:rgba(0,0,0,0.2);padding:3px 10px;border-radius:10px;color:rgba(255,255,255,0.6);font-weight:400;font-family:DM Sans;">Based on patient report</span></div>'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                    '<div>'
                    '<div style="font-size:11px;color:'+info["color"]+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Medicines</div>'
                    +rx_items+
                    '</div>'
                    '<div>'
                    '<div style="font-size:11px;color:'+info["color"]+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Medical Advice</div>'
                    +advice_items+
                    '<div style="font-size:11px;color:'+info["color"]+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 8px;">Diet Recommendations</div>'
                    +diet_items+
                    '</div>'
                    '</div>'
                    '<div style="margin-top:14px;padding:10px 14px;background:rgba(0,0,0,0.2);border-radius:10px;font-size:12px;color:rgba(255,255,255,0.7);">'
                    '⚠️ <strong style="color:'+info["color"]+';">Disclaimer:</strong> This is an AI-generated suggestion. Always consult a qualified doctor before taking any medication.'
                    '</div></div>',
                    unsafe_allow_html=True
                )

                st.session_state.counts[category]+=1
                now=datetime.datetime.now()
                st.session_state.history.insert(0,{"name":name,"age":p_age,"gender":p_gender,"doc":p_doc,"date":str(p_date),"time":now.strftime("%I:%M %p"),"report":p_report[:65]+("..." if len(p_report)>65 else ""),"cat":category,"conf":confidence,"severity":severity,"sev_icon":sev_icon,"color":info["color"],"bc":info["bc"],"followup":rx_data["followup"]})
                st.rerun()

    with right:
        with st.expander("📊 Project flow",expanded=False):
            for i,(icon,label) in enumerate([("📄","Input report text or image"),("🖼️","OCR reads image automatically"),("🔤","NLP text cleaning"),("📁","CSV dataset trained model"),("🤖","Naive Bayes classifies"),("💊","Auto prescription generated"),("📊","Gauge + bar charts shown")]):
                st.markdown('<div class="flow-step"><span style="font-size:16px;">'+icon+'</span> '+label+'</div>',unsafe_allow_html=True)
                if i<6: st.markdown('<div style="text-align:center;color:#f5a623;font-size:18px;margin:-2px 0">↓</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:16px">Classification history</div>',unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:32px 16px;color:#8886c8;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:36px;margin-bottom:10px">📋</div><div>No reports classified yet.</div></div>',unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:10]:
                meta=" · ".join(filter(None,["Age "+h['age'] if h['age'] else "",h['gender'],"Dr: "+h['doc'] if h['doc'] else ""]))
                color=h.get('color','#f5a623'); bc=h.get('bc','b-clin')
                meta_line='<div class="hist-meta">'+meta+'</div>' if meta else ""
                st.markdown(
                    '<div class="hist-card">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
                    '<span class="hist-name">'+h['name']+'</span>'
                    '<span class="badge '+bc+'">'+h['cat']+'</span></div>'
                    +meta_line+
                    '<div class="hist-rep">'+h['report']+'</div>'
                    '<div class="hist-time">'+h['date']+' at '+h['time']+' &nbsp;|&nbsp; <span style="color:'+color+'">'+str(h['conf'])+'% &nbsp;'+h.get('sev_icon','')+'&nbsp;'+h.get('severity','')+'</span> &nbsp;|&nbsp; Follow-up: '+h.get('followup','—')+'</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            if st.button("🗑️ Clear history"):
                st.session_state.history=[]
                st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}
                st.rerun()

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project &nbsp;|&nbsp; Python · Scikit-Learn · Streamlit · OCR · Plotly</div>',unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page=="home": show_home()
    elif st.session_state.page=="about": show_about()
    elif st.session_state.page=="contact": show_contact()
    elif st.session_state.page=="dashboard": show_dashboard()
    else: show_home()

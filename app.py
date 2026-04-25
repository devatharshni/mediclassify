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
if 'last_result' not in st.session_state: st.session_state.last_result=None

USERS={"admin":"admin123","doctor":"medi2024","student":"project123"}

PRESCRIPTIONS={
    "Radiology":{
        "fracture":{"medicines":["Tab. Ibuprofen 400mg — Twice daily after food","Tab. Calcium + Vitamin D3 — Once daily","Oint. Diclofenac Gel — Apply 3x daily on affected area","Tab. Pantoprazole 40mg — Before breakfast"],"advice":["Keep injured area immobilized with splint/cast","Apply ice pack for 20 min every 4 hours","Avoid weight bearing on fracture site","Physiotherapy after 4 weeks as advised","X-ray follow-up after 4 weeks"],"diet":["High calcium: milk, yogurt, cheese, broccoli","Vitamin D: eggs, fish, sunlight 15 min/day","Protein rich: chicken, dal, eggs for healing","Avoid alcohol and smoking — delays bone healing"],"followup":"4 weeks"},
        "tumor":{"medicines":["URGENT — Refer to Oncology immediately","Tab. Dexamethasone 4mg — As directed by specialist","Tab. Pantoprazole 40mg — Before breakfast","Pain management as prescribed by oncologist"],"advice":["Urgent oncology consultation required TODAY","MRI with contrast recommended","Biopsy needed for confirmation","Avoid stress — maintain complete rest","Do not delay — time sensitive condition"],"diet":["High protein diet for strength","Antioxidant rich: berries, turmeric, green tea","Avoid processed and junk foods","Stay hydrated — 3L water daily"],"followup":"URGENT — 1 week"},
        "default":{"medicines":["Tab. Analgesic 400mg — Twice daily after food","Tab. Calcium + D3 Supplement — Once daily","Topical anti-inflammatory gel — Twice daily","Tab. Pantoprazole 40mg — Before breakfast"],"advice":["Complete rest for affected area","Follow radiologist recommendations carefully","Repeat imaging if symptoms worsen","Avoid strenuous physical activity","Attend all scheduled follow-ups"],"diet":["Balanced nutrition with all food groups","Adequate hydration — 2-3 litres water/day","Calcium and Vitamin D rich foods","Avoid alcohol and smoking"],"followup":"2 weeks"},
    },
    "Lab Report":{
        "diabetes":{"medicines":["Tab. Metformin 500mg — Twice daily after food","Tab. Glimepiride 1mg — Before breakfast","Tab. Vitamin B12 500mcg — Once daily","Tab. Aspirin 75mg — Once daily (if BP high)"],"advice":["Monitor blood sugar daily — fasting & post-meal","HbA1c test every 3 months","Regular foot and eye checkup every 6 months","Exercise 30-45 min daily — walking/cycling","Emergency: blood sugar < 70 — eat sugar immediately"],"diet":["Avoid: sugar, white rice, maida, sweets, fruit juice","Eat: whole grains, vegetables, dal, salads","Small frequent meals — 5-6 times/day","Fruits: guava, apple, papaya (avoid mango/banana)","No alcohol — raises blood sugar dangerously"],"followup":"1 month"},
        "anemia":{"medicines":["Tab. Ferrous Sulfate 200mg — Twice daily empty stomach","Tab. Folic Acid 5mg — Once daily","Syp. Iron + Vitamin C — Twice daily","Tab. Vitamin B12 500mcg — Once daily"],"advice":["Take iron on empty stomach for best absorption","Avoid tea/coffee within 1 hour of iron tablets","Check hemoglobin after 4 weeks of treatment","Blood transfusion if Hb falls below 7 g/dL","Avoid NSAIDs — worsens stomach irritation"],"diet":["Spinach, beetroot, pomegranate, dates daily","Vitamin C foods to boost iron: lemon, orange, amla","Jaggery, raisins, dried figs are excellent","Avoid tea immediately after meals","Chicken liver and red meat for non-vegetarians"],"followup":"4 weeks"},
        "cholesterol":{"medicines":["Tab. Atorvastatin 10mg — Once daily at night","Tab. Omega-3 1000mg — Twice daily after food","Tab. Aspirin 75mg — Once daily after breakfast","Tab. Ezetimibe 10mg — Once daily (if LDL very high)"],"advice":["Lipid profile test every 3 months","Exercise 45 min daily — brisk walk minimum","Quit smoking and alcohol completely","Weight management: target BMI below 25","Avoid sitting for long — take breaks every hour"],"diet":["Avoid: fried food, butter, ghee, red meat","Eat: oats, nuts, olive oil, avocado","Omega-3 rich: fish, flaxseeds, walnuts","Fruits: berries, citrus, apples — daily","No egg yolk — egg white is allowed"],"followup":"3 months"},
        "default":{"medicines":["Medicines based on specific lab report values","Tab. Multivitamin — Once daily after food","Tab. Probiotic — Twice daily","Consult doctor for specific medication"],"advice":["Repeat lab test after completing treatment","Maintain healthy lifestyle consistently","Stay hydrated — 2-3 litres water/day","Regular health checkups every 6 months","Bring all previous reports to next visit"],"diet":["Balanced diet with all nutrients","Fresh fruits and vegetables daily","Avoid junk and processed foods completely","Adequate protein intake every meal","Reduce salt and sugar in diet"],"followup":"2-4 weeks"},
    },
    "Cardiology":{
        "hypertension":{"medicines":["Tab. Amlodipine 5mg — Once daily morning","Tab. Telmisartan 40mg — Once daily","Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily night"],"advice":["Monitor BP twice daily — morning and evening","Target BP: below 130/80 mmHg strictly","EMERGENCY: BP above 180/120 — go to hospital NOW","Avoid all forms of stress — practice meditation","Walk 30 min daily — no intense exercise"],"diet":["DASH diet — fruits vegetables whole grains strictly","Avoid salt completely — no pickles papad sauces","No alcohol and smoking — major BP triggers","Limit caffeine — maximum 1 cup/day","Banana, watermelon, beetroot help lower BP"],"followup":"2 weeks"},
        "heart":{"medicines":["Tab. Metoprolol 25mg — Twice daily","Tab. Ramipril 5mg — Once daily","Tab. Furosemide 40mg — Once daily morning","Tab. Spironolactone 25mg — Once daily","Tab. Aspirin 75mg — Once daily after food"],"advice":["URGENT — Cardiology OPD visit required immediately","Avoid ALL physical exertion until reviewed","Sleep with head elevated 30 degrees","Report chest pain or breathlessness — EMERGENCY","Daily weight monitoring — gain of 2kg means fluid retention"],"diet":["Low sodium — less than 2g salt/day strictly","Fluid restriction as advised by cardiologist","Heart healthy: fish, nuts, olive oil, oats","Avoid caffeine and alcohol completely","Small light meals — no heavy meals ever"],"followup":"1 week — URGENT"},
        "cholesterol":{"medicines":["Tab. Atorvastatin 20mg — Once daily night","Tab. Omega-3 1000mg — Twice daily","Tab. Aspirin 75mg — Once daily","Tab. Ezetimibe 10mg — Once daily"],"advice":["Lipid profile every 3 months","Exercise daily — 45 minutes brisk walk","Quit smoking — increases cardiac risk 3x","Stress management — yoga and meditation","Annual cardiac checkup essential"],"diet":["Avoid fried oily fatty foods completely","Oats walnuts flaxseeds olive oil daily","Plenty of vegetables and fresh fruits","No red meat — fish allowed 2x/week","No alcohol — damages heart directly"],"followup":"3 months"},
        "default":{"medicines":["Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily night","Tab. Metoprolol 25mg — Twice daily","Tab. Ramipril 2.5mg — Once daily"],"advice":["Cardiology consultation required urgently","ECG and Echocardiogram recommended","Avoid strenuous exercise until cleared","Report chest pain or breathlessness IMMEDIATELY","Carry emergency contact and medical history"],"diet":["Heart healthy balanced diet","Low salt low fat diet strictly","Avoid fried oily fatty foods","Plenty of fruits and vegetables","No alcohol no smoking"],"followup":"2 weeks"},
    },
    "Clinical Notes":{
        "fever":{"medicines":["Tab. Paracetamol 500mg — Three times daily","Tab. Cetirizine 10mg — Once daily night","Syp. Benadryl 10ml — Three times daily","ORS Sachet — Dissolve in 1L water drink throughout day"],"advice":["Complete bed rest for 3-5 days","Tepid sponging if fever exceeds 102 degree F","Drink 3-4 litres of fluids daily","Return immediately if fever above 104F or convulsions","Monitor temperature every 4 hours"],"diet":["Light food: khichdi soup boiled rice","Plenty of fluids: coconut water ORS lemon juice","Avoid spicy oily and heavy food","Small frequent meals every 3 hours","No cold drinks or ice cream"],"followup":"3-5 days or earlier if worsening"},
        "infection":{"medicines":["Tab. Amoxicillin 500mg — Three times daily x 5 days","Tab. Ibuprofen 400mg — Twice daily after food","Tab. Probiotic — Twice daily","Tab. Paracetamol 500mg — SOS for fever or pain"],"advice":["Complete the FULL antibiotic course — never stop early","Do not share antibiotics with others","Rest and avoid exertion completely","Maintain strict hand hygiene","Return if no improvement after 3 days"],"diet":["Immunity boosting: turmeric milk ginger tea daily","Vitamin C: oranges lemons amla guava","Avoid cold drinks ice cream cold water","Plenty of warm fluids and soups","Garlic and ginger in meals — natural antibiotics"],"followup":"5-7 days"},
        "diabetes":{"medicines":["Tab. Metformin 500mg — Twice daily after food","Tab. Glimepiride 1mg — Before breakfast","Tab. Vitamin B12 — Once daily","Blood sugar monitoring strip — check daily"],"advice":["Monitor blood sugar every morning fasting","HbA1c every 3 months","Foot care daily — check for wounds","Regular eye checkup every 6 months","Exercise 30 min daily minimum"],"diet":["Avoid sugar white rice maida completely","Whole grains vegetables fruits with low glycemic index","Small frequent meals 5-6 times per day","No alcohol no fruit juices","Drink 2-3 litres water daily"],"followup":"1 month"},
        "default":{"medicines":["Tab. Paracetamol 500mg — SOS for fever or pain","Tab. Vitamin C 500mg — Once daily","Tab. Zinc 20mg — Once daily","ORS sachet — As needed for hydration"],"advice":["Rest at home for 2-3 days minimum","Monitor symptoms closely every few hours","Drink plenty of warm fluids throughout day","Return immediately if no improvement in 3 days","Avoid self-medication beyond what prescribed"],"diet":["Light easily digestible food: khichdi dal rice","Warm soups and broths throughout day","Fresh fruits for immunity boost","Avoid junk food cold drinks outside food","Honey and tulsi tea for throat and immunity"],"followup":"3-5 days"},
    }
}

def get_auto_prescription(category,report_text):
    text=report_text.lower()
    db=PRESCRIPTIONS.get(category,PRESCRIPTIONS["Clinical Notes"])
    for key in db:
        if key!="default" and key in text:
            return db[key]
    # Smart keyword matching
    keyword_map={
        "Cardiology":{"hypertension":["high bp","blood pressure","hypertension","htn"],"heart":["heart failure","cardiac","ecg","palpitation","chest pain","myocardial","ejection"],"cholesterol":["cholesterol","ldl","hdl","lipid"]},
        "Lab Report":{"diabetes":["diabetes","glucose","sugar","hba1c","diabetic"],"anemia":["anemia","hemoglobin","iron","hb low"],"cholesterol":["cholesterol","ldl","lipid","triglyceride"]},
        "Radiology":{"fracture":["fracture","broken","crack","dislocation"],"tumor":["tumor","cancer","malignant","metastatic","mass"]},
        "Clinical Notes":{"fever":["fever","temperature","viral","flu","cold","cough"],"infection":["infection","bacteria","antibiotic","pus","wound"],"diabetes":["diabetes","sugar","glucose"]},
    }
    cat_keywords=keyword_map.get(category,{})
    for key,words in cat_keywords.items():
        if any(w in text for w in words):
            return db.get(key,db["default"])
    return db["default"]

def get_cat_info(cat):
    return {
        "Radiology":     {"icon":"🩻","color":"#4d9fff","bg":"linear-gradient(135deg,rgba(77,159,255,0.18),rgba(30,28,94,0.95))","border":"rgba(77,159,255,0.5)","dept":"Radiology & Imaging","bc":"b-rad"},
        "Lab Report":    {"icon":"🔬","color":"#00d4b4","bg":"linear-gradient(135deg,rgba(0,212,180,0.18),rgba(30,28,94,0.95))","border":"rgba(0,212,180,0.5)","dept":"Pathology Lab","bc":"b-lab"},
        "Cardiology":    {"icon":"❤️","color":"#ff6b9d","bg":"linear-gradient(135deg,rgba(255,107,157,0.18),rgba(30,28,94,0.95))","border":"rgba(255,107,157,0.5)","dept":"Cardiology Department","bc":"b-card"},
        "Clinical Notes":{"icon":"🩺","color":"#f5a623","bg":"linear-gradient(135deg,rgba(245,166,35,0.18),rgba(30,28,94,0.95))","border":"rgba(245,166,35,0.5)","dept":"General Medicine","bc":"b-clin"},
    }.get(cat,{"icon":"🏥","color":"#9b98cc","bg":"rgba(155,152,204,0.12)","border":"rgba(155,152,204,0.3)","dept":"General","bc":"b-clin"})

def get_severity(conf):
    if conf>=80: return "Critical","sev-critical","🔴","Immediate attention required"
    elif conf>=55: return "Moderate","sev-moderate","🟡","Close monitoring needed"
    else: return "Normal","sev-normal","🟢","Routine follow-up"

def get_explanation(cat,conf):
    return {
        "Radiology":f"AI detected imaging keywords (scan, X-ray, MRI, bone, fracture) with {conf}% confidence. Report indicates a radiological finding requiring specialist review.",
        "Lab Report":f"AI detected lab test keywords (blood levels, glucose, enzymes, cell counts) with {conf}% confidence. Abnormal values identified — pathologist review recommended.",
        "Cardiology":f"AI detected cardiac keywords (ECG, blood pressure, heart, chest pain, palpitations) with {conf}% confidence. Cardiology consultation {'urgently required' if conf>75 else 'recommended'}.",
        "Clinical Notes":f"AI detected clinical symptom keywords (fever, pain, cough, infection) with {conf}% confidence. General medicine treatment {'actively required' if conf>75 else 'recommended'}.",
    }.get(cat,"AI analyzed the report and matched medical keyword patterns.")

def extract_text_from_image(img):
    try: return pytesseract.image_to_string(img,config='--psm 6').strip()
    except: return ""

@st.cache_resource
def train_model():
    csv_path="medical_data.csv"
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path)
        df.columns=[c.strip().lower() for c in df.columns]
        df=df.dropna()
        texts=df['text'].tolist(); labels=df['label'].tolist()
    else:
        texts=["X-ray shows fracture femur bone","MRI scan herniated disc lumbar","CT scan chest pneumonia","Ultrasound gallstones abdomen","MRI brain tumor temporal lobe","Mammogram calcification","CT liver lesion","Blood test high glucose diabetes","Hemoglobin low anemia","White blood cell elevated infection","Urine protein bacteria","Thyroid hypothyroidism TSH","Liver enzymes elevated hepatitis","Platelet count low","Cholesterol LDL critical high","ECG irregular heartbeat atrial fibrillation","High blood pressure chest pain hypertension","Echocardiogram reduced ejection fraction","Shortness breath palpitations cardiac","Heart failure diagnosed cardiology","Hypertension uncontrolled blood pressure","Troponin elevated cardiac episode","Patient fever cough body pain","Headache vomiting morning","Follow up diabetes management","Antibiotics throat infection","Patient recovering surgery","Child fever skin rash","Fatigue loss appetite","Routine checkup vitals normal","Patient high bp diabetes cholesterol","Blood pressure hypertension cardiac risk","Chest pain shortness breath heart","ECG changes ischemia cardiac","Diabetes blood sugar high glucose insulin"]
        labels=["Radiology","Radiology","Radiology","Radiology","Radiology","Radiology","Radiology","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Lab Report","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Cardiology","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Clinical Notes","Cardiology","Cardiology","Cardiology","Cardiology","Lab Report"]
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

model,model_acc,total_samples=train_model()

SHIELD="""<svg width="46" height="52" viewBox="0 0 80 90" fill="none"><defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/></linearGradient></defs><path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg)"/><rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/><rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/></svg>"""
SHIELD_BIG="""<svg width="88" height="98" viewBox="0 0 80 90" fill="none"><defs><linearGradient id="sg2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/></linearGradient></defs><path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg2)"/><rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/><rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/></svg>"""

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
.stButton>button:hover{box-shadow:0 8px 30px rgba(245,166,35,0.6) !important;}
.glass-card{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:22px;padding:26px 30px;margin-bottom:18px;}
.sec-title{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;color:#f5a623;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #3d3a8a;display:flex;align-items:center;gap:8px;}
.sec-title::before{content:'';width:4px;height:14px;background:linear-gradient(180deg,#f5a623,#d4881a);border-radius:2px;flex-shrink:0;}
.stat-box{background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;padding:16px 18px;text-align:center;margin-bottom:14px;}
.stat-num{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f5a623;}
.stat-lbl{color:#8886c8;font-size:12px;margin-top:4px;}
.sev-pill{display:inline-flex;align-items:center;gap:6px;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;}
.sev-critical{background:rgba(255,70,70,0.2);color:#ff6b6b;border:1px solid rgba(255,70,70,0.4);}
.sev-moderate{background:rgba(255,179,71,0.2);color:#ffb347;border:1px solid rgba(255,179,71,0.4);}
.sev-normal{background:rgba(0,212,100,0.2);color:#00d464;border:1px solid rgba(0,212,100,0.4);}
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
.pulse{display:inline-block;width:7px;height:7px;background:#f5a623;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.7);}}
.page-title{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#f0f0ff;margin-bottom:6px;}
.page-sub{font-size:14px;color:#8886c8;margin-bottom:24px;}
.hist-card{background:rgba(255,255,255,0.03);border:1px solid #3d3a8a;border-radius:14px;padding:13px 16px;margin-bottom:10px;transition:border-color 0.2s;}
.hist-card:hover{border-color:rgba(245,166,35,0.4);}
.hist-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#f0f0ff;}
.hist-meta{font-size:11px;color:#8886c8;margin:3px 0;}
.hist-rep{font-size:12px;color:rgba(240,240,255,0.7);margin:4px 0;}
.hist-time{font-size:10px;color:#8886c8;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<div style="text-align:center;padding:24px 0 16px;"><div style="display:inline-block;filter:drop-shadow(0 0 22px rgba(245,166,35,0.4))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:42px;font-weight:800;color:#f5a623;margin-top:8px;letter-spacing:0.05em;">MEDICLASSIFY</div><div style="font-size:15px;color:#8886c8;font-style:italic;margin:6px 0 12px;">Diagnose Faster. Treat Better.</div><div style="font-size:14px;color:rgba(240,240,255,0.75);max-width:540px;margin:0 auto;line-height:1.7;">Welcome back, <strong style="color:#f5a623">'+st.session_state.username+'</strong>! 👋<br>AI classification with auto-prescription and advanced charts.</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    with m1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total_samples)+'</div><div class="stat-lbl">CSV Training samples</div></div>',unsafe_allow_html=True)
    with m2: st.markdown('<div class="stat-box"><div class="stat-num">'+str(model_acc)+'%</div><div class="stat-lbl">Model accuracy</div></div>',unsafe_allow_html=True)
    with m3: st.markdown('<div class="stat-box"><div class="stat-num">4</div><div class="stat-lbl">Medical categories</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    features=[("🤖","AI Classification","Naive Bayes + TF-IDF on CSV"),("🖼️","OCR Support","Auto reads prescription images"),("💊","Auto Prescription","Smart medicines from report"),("🔴🟡🟢","Severity Detection","Critical Moderate Normal"),("📊","Gauge + Bar Charts","Advanced visualizations"),("📋","History Log","Complete patient records")]
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
    st.markdown('<div class="glass-card"><div class="sec-title">Project overview</div><p style="font-size:14px;color:rgba(220,220,255,0.85);line-height:1.85;"><strong style="color:#f5a623">MediClassify v3.0</strong> is an AI-powered Medical Report Classification System built for placement mini project. It reads medical reports or prescription images, classifies into the correct department, auto-generates prescription, detects severity, and shows advanced gauge and bar chart visualizations.</p></div>',unsafe_allow_html=True)
    techs=[("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes + TF-IDF"),("📊","Streamlit","Web UI"),("🖼️","Tesseract OCR","Image to text"),("📈","Plotly","Gauge and bar charts"),("📁","CSV Dataset","Training data source")]
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

def render_result(category,confidence,all_proba,severity,sev_class,sev_icon,sev_msg,explanation,info,rx_data,name,meta,p_date):
    color=info["color"]; bg=info["bg"]; border=info["border"]

    # ── BIG RESULT CARD ──────────────────────────────────────
    st.markdown(
        '<div style="background:'+bg+';border:2px solid '+border+';border-radius:22px;overflow:hidden;margin-bottom:20px;">'
        '<div style="padding:24px 28px;border-bottom:1px solid '+border+'30;display:flex;align-items:center;justify-content:space-between;">'
        '<div>'
        '<div style="font-size:10px;color:'+color+';text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px;opacity:0.8;">✦ AI Classification Result</div>'
        '<div style="font-family:Syne,sans-serif;font-size:30px;font-weight:800;color:'+color+';">'+category+'</div>'
        '<div style="font-size:13px;color:rgba(255,255,255,0.7);margin-top:5px;">'+name+((" &nbsp;·&nbsp; "+meta) if meta else "")+'</div>'
        '<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">'
        '<span class="sev-pill '+sev_class+'">'+sev_icon+' '+severity+' — '+sev_msg+'</span>'
        '<span style="background:rgba(255,255,255,0.08);padding:5px 14px;border-radius:20px;font-size:11px;color:rgba(255,255,255,0.8);">🏥 '+info["dept"]+'</span>'
        '</div></div>'
        '<div style="font-size:64px;filter:drop-shadow(0 0 16px '+color+'80);">'+info["icon"]+'</div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;">'
        '<div style="padding:16px 20px;text-align:center;border-right:1px solid '+border+'30;">'
        '<div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;">Confidence</div>'
        '<div style="font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:'+color+';">'+str(confidence)+'%</div></div>'
        '<div style="padding:16px 20px;text-align:center;border-right:1px solid '+border+'30;">'
        '<div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;">Severity</div>'
        '<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:'+color+';">'+sev_icon+' '+severity+'</div></div>'
        '<div style="padding:16px 20px;text-align:center;">'
        '<div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;">Follow-up</div>'
        '<div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:'+color+';">'+rx_data["followup"]+'</div></div>'
        '</div>'
        '<div style="padding:18px 24px;border-top:1px solid '+border+'30;">'
        '<div style="font-size:10px;color:'+color+';font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🤖 AI Explanation</div>'
        '<div style="font-size:13px;color:rgba(255,255,255,0.85);line-height:1.75;">'+explanation+'</div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    # ── CHARTS SIDE BY SIDE ───────────────────────────────────
    ch1,ch2=st.columns(2)
    with ch1:
        fig_gauge=go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text":"Confidence Score","font":{"color":color,"size":13,"family":"DM Sans"}},
            number={"suffix":"%","font":{"color":color,"size":40,"family":"Syne"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8886c8","tickfont":{"color":"#8886c8","size":10}},
                "bar":{"color":color,"thickness":0.28},
                "bgcolor":"rgba(26,24,78,0.9)",
                "borderwidth":1,"bordercolor":border,
                "steps":[
                    {"range":[0,40],"color":"rgba(0,212,100,0.12)"},
                    {"range":[40,70],"color":"rgba(255,179,71,0.12)"},
                    {"range":[70,100],"color":"rgba(255,70,70,0.12)"},
                ],
                "threshold":{"line":{"color":color,"width":4},"thickness":0.78,"value":confidence}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"white","family":"DM Sans"},height=230,margin=dict(t=44,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge,use_container_width=True)

    with ch2:
        cats=list(all_proba.keys())
        vals=list(all_proba.values())
        bar_colors={"Radiology":"#4d9fff","Lab Report":"#00d4b4","Cardiology":"#ff6b9d","Clinical Notes":"#f5a623"}
        colors_list=[bar_colors.get(c,"#9b98cc") for c in cats]
        fig_bar=go.Figure(go.Bar(
            y=cats,x=vals,orientation='h',
            marker_color=colors_list,
            marker_line_width=0,
            text=[str(v)+"%" for v in vals],
            textposition="outside",
            textfont=dict(color="white",size=11)
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,24,78,0.5)",
            font=dict(color="white",family="DM Sans"),
            title=dict(text="Category breakdown",font=dict(color=color,size=13)),
            xaxis=dict(range=[0,120],ticksuffix="%",gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#8886c8",size=10)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color="white",size=11)),
            showlegend=False,height=230,
            margin=dict(t=36,b=10,l=10,r=50)
        )
        st.plotly_chart(fig_bar,use_container_width=True)

    # ── AUTO PRESCRIPTION CARD ────────────────────────────────
    rx_items="".join([
        '<div style="background:rgba(0,0,0,0.2);border:1px solid '+border+'40;border-radius:10px;padding:10px 14px;margin-bottom:8px;font-size:13px;color:rgba(255,255,255,0.9);display:flex;align-items:flex-start;gap:10px;">'
        '<span style="color:'+color+';font-size:18px;flex-shrink:0;">💊</span><span>'+m+'</span></div>'
        for m in rx_data["medicines"]
    ])
    advice_items="".join([
        '<div style="background:rgba(0,0,0,0.15);border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:12px;color:rgba(255,255,255,0.85);display:flex;gap:8px;">'
        '<span style="color:'+color+';flex-shrink:0;">✦</span><span>'+a+'</span></div>'
        for a in rx_data["advice"]
    ])
    diet_items="".join([
        '<div style="background:rgba(0,0,0,0.15);border-radius:8px;padding:7px 12px;margin-bottom:5px;font-size:12px;color:rgba(255,255,255,0.8);">🥗 '+d+'</div>'
        for d in rx_data["diet"]
    ])

    st.markdown(
        '<div style="background:'+bg+';border:2px solid '+border+';border-radius:20px;padding:22px 26px;margin-top:4px;">'
        '<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;color:'+color+';text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;display:flex;align-items:center;gap:10px;">'
        '💊 Auto-Generated Prescription'
        '<span style="font-size:10px;background:rgba(0,0,0,0.25);padding:3px 10px;border-radius:10px;color:rgba(255,255,255,0.5);font-weight:400;font-family:DM Sans;">Based on patient report keywords</span>'
        '</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
        '<div>'
        '<div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Medicines</div>'
        +rx_items+
        '</div>'
        '<div>'
        '<div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Medical Advice</div>'
        +advice_items+
        '<div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 10px;">Diet Plan</div>'
        +diet_items+
        '</div>'
        '</div>'
        '<div style="margin-top:16px;padding:10px 14px;background:rgba(0,0,0,0.25);border-radius:10px;font-size:12px;color:rgba(255,255,255,0.6);">'
        '⚠️ <strong style="color:'+color+';">Disclaimer:</strong> This is AI-generated. Always consult a qualified doctor before taking any medication.'
        '</div></div>',
        unsafe_allow_html=True
    )

def show_dashboard():
    show_navbar()
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:20px;padding:20px 28px;margin-bottom:20px;"><div style="display:flex;align-items:center;gap:14px;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:11px;color:#8886c8;">Accuracy: <strong style="color:#f5a623">'+str(model_acc)+'%</strong> &nbsp;|&nbsp; Dataset: <strong style="color:#f5a623">'+str(total_samples)+' samples</strong></div></div></div><div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);color:#f5a623;padding:7px 18px;border-radius:20px;font-size:12px;font-weight:600;"><span class="pulse"></span>AI Active</div></div>',unsafe_allow_html=True)

    total=sum(st.session_state.counts.values())
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total)+'</div><div class="stat-lbl">Total classified</div></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#4d9fff">'+str(st.session_state.counts["Radiology"])+'</div><div class="stat-lbl">Radiology</div></div>',unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#00d4b4">'+str(st.session_state.counts["Lab Report"])+'</div><div class="stat-lbl">Lab Report</div></div>',unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#ff6b9d">'+str(st.session_state.counts["Cardiology"])+'</div><div class="stat-lbl">Cardiology</div></div>',unsafe_allow_html=True)

    # ── INPUT FORM (LEFT) + HISTORY (RIGHT) ──────────────────
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
        p_report=""

        if input_type in ["📝 Type Report","📝 + 🖼️ Both"]:
            p_report=st.text_area("Report / symptoms",placeholder="e.g. Patient has high BP and diabetes with cholesterol\nMRI scan shows fracture in left leg\nBlood test shows high glucose",height=110)

        if input_type in ["🖼️ Upload Image","📝 + 🖼️ Both"]:
            uploaded_img=st.file_uploader("Upload prescription image",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uploaded_img is not None:
                img=Image.open(uploaded_img)
                st.image(img,caption="Uploaded image",use_container_width=True)
                with st.spinner("🔍 Reading text from image (OCR)..."):
                    ocr_text=extract_text_from_image(img)
                if ocr_text.strip():
                    st.markdown('<div class="ocr-box"><div style="font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">📄 OCR Extracted Text</div><div style="font-size:13px;color:#e0e0ff;line-height:1.7;">'+ocr_text+'</div></div>',unsafe_allow_html=True)
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

                # Save to session
                st.session_state.last_result={
                    "category":category,"confidence":confidence,"all_proba":all_proba,
                    "severity":severity,"sev_class":sev_class,"sev_icon":sev_icon,"sev_msg":sev_msg,
                    "explanation":explanation,"info":info,"rx_data":rx_data,
                    "name":name,"meta":meta,"p_date":str(p_date)
                }
                st.session_state.counts[category]+=1
                now=datetime.datetime.now()
                st.session_state.history.insert(0,{
                    "name":name,"age":p_age,"gender":p_gender,"doc":p_doc,
                    "date":str(p_date),"time":now.strftime("%I:%M %p"),
                    "report":p_report[:65]+("..." if len(p_report)>65 else ""),
                    "cat":category,"conf":confidence,"severity":severity,
                    "sev_icon":sev_icon,"color":info["color"],"bc":info["bc"],
                    "followup":rx_data["followup"]
                })
                st.rerun()

    with right:
        with st.expander("📊 Project flow",expanded=False):
            for i,(icon,label) in enumerate([("📄","Input report text or image"),("🖼️","OCR reads image automatically"),("📁","CSV dataset trained model"),("🤖","Naive Bayes classifies"),("💊","Auto prescription generated"),("📊","Gauge + bar charts shown")]):
                st.markdown('<div class="flow-step"><span style="font-size:16px;">'+icon+'</span> '+label+'</div>',unsafe_allow_html=True)
                if i<5: st.markdown('<div style="text-align:center;color:#f5a623;font-size:18px;margin:-2px 0">↓</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:16px">Classification history</div>',unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:32px 16px;color:#8886c8;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:36px;margin-bottom:10px">📋</div><div>No reports classified yet.</div></div>',unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:8]:
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
                    '<div class="hist-time">'+h['date']+' at '+h['time']+' &nbsp;|&nbsp; <span style="color:'+color+'">'+str(h['conf'])+'% '+h.get('sev_icon','')+'</span> &nbsp;|&nbsp; '+h.get('followup','—')+'</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            if st.button("🗑️ Clear history"):
                st.session_state.history=[]
                st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}
                st.session_state.last_result=None
                st.rerun()

    # ── OUTPUT SHOWN BELOW FULL WIDTH ─────────────────────────
    if st.session_state.last_result:
        r=st.session_state.last_result
        st.markdown("<hr style='border:none;border-top:2px solid #3d3a8a;margin:24px 0'>",unsafe_allow_html=True)
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;color:#f5a623;margin-bottom:18px;">📋 Classification Result & Prescription</div>',unsafe_allow_html=True)
        render_result(r["category"],r["confidence"],r["all_proba"],r["severity"],r["sev_class"],r["sev_icon"],r["sev_msg"],r["explanation"],r["info"],r["rx_data"],r["name"],r["meta"],r["p_date"])

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project</div>',unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page=="home": show_home()
    elif st.session_state.page=="about": show_about()
    elif st.session_state.page=="contact": show_contact()
    elif st.session_state.page=="dashboard": show_dashboard()
    else: show_home()

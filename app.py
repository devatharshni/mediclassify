import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"])

for pkg in ["scikit-learn","Pillow","pytesseract","plotly","numpy","pandas"]:
    try: __import__(pkg.replace("-","_").split("==")[0])
    except ImportError: install(pkg)

import streamlit as st
import re, datetime, os, json
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

# ── SESSION STATE ────────────────────────────────────────────
for key,val in [('logged_in',False),('username',''),('page','login'),('auth_mode','login'),
                ('history',[]),('counts',{"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0,"Neurology":0,"Orthopedics":0,"Dermatology":0,"Pediatrics":0}),
                ('last_result',None),('users',{"admin":"admin123","doctor":"medi2024","student":"project123"})]:
    if key not in st.session_state: st.session_state[key]=val

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background-color:#16144a !important;}
.stTextInput input{background:#1a184e !important;color:#fff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;padding:12px 16px !important;font-size:14px !important;}
.stTextInput input::placeholder{color:#7875b5 !important;}
.stTextInput input:focus{border-color:#f5a623 !important;}
.stTextInput label,.stTextArea label,.stSelectbox label,.stDateInput label,.stRadio label{color:#b0aee8 !important;font-size:13px !important;font-weight:500 !important;}
.stTextArea textarea{background:#1a184e !important;color:#fff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stTextArea textarea::placeholder{color:#7875b5 !important;}
.stSelectbox>div>div{background:#1a184e !important;color:#fff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
.stDateInput input{background:#1a184e !important;color:#fff !important;border:1.5px solid #4a47a3 !important;border-radius:12px !important;}
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
.b-neuro{background:rgba(180,100,255,0.15);color:#b464ff;border:1px solid rgba(180,100,255,0.3);}
.b-ortho{background:rgba(100,200,255,0.15);color:#64c8ff;border:1px solid rgba(100,200,255,0.3);}
.b-derm{background:rgba(255,150,100,0.15);color:#ff9664;border:1px solid rgba(255,150,100,0.3);}
.b-peds{background:rgba(100,255,150,0.15);color:#64ff96;border:1px solid rgba(100,255,150,0.3);}
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
/* AUTH TABS */
.auth-tab-active{background:linear-gradient(135deg,#f5a623,#d4881a);color:#1a0f00;border:none;border-radius:12px;font-family:'Syne',sans-serif;font-weight:800;font-size:14px;padding:10px 32px;cursor:pointer;letter-spacing:0.04em;}
.auth-tab-inactive{background:rgba(255,255,255,0.05);color:#b0aee8;border:1px solid #4a47a3;border-radius:12px;font-family:'Syne',sans-serif;font-weight:600;font-size:14px;padding:10px 32px;cursor:pointer;}
</style>
""", unsafe_allow_html=True)

SHIELD="""<svg width="46" height="52" viewBox="0 0 80 90" fill="none"><defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/></linearGradient></defs><path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg)"/><rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/><rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/></svg>"""
SHIELD_BIG="""<svg width="88" height="98" viewBox="0 0 80 90" fill="none"><defs><linearGradient id="sg2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/></linearGradient></defs><path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg2)"/><rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/><rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/></svg>"""

# ── TRAIN MODEL (CACHED — loads once, stays fast) ────────────
@st.cache_resource(show_spinner=False)
def train_model():
    csv_path="medical_data.csv"
    if os.path.exists(csv_path):
        df=pd.read_csv(csv_path)
        df.columns=[c.strip().lower() for c in df.columns]
        df=df.dropna()
        texts=df['text'].tolist(); labels=df['label'].tolist()
    else:
        # 8 departments with rich training data
        data=[
            # RADIOLOGY
            ("X-ray shows fracture in left femur bone","Radiology"),
            ("MRI scan reveals herniated disc in lumbar region","Radiology"),
            ("CT scan chest shows pneumonia consolidation","Radiology"),
            ("Ultrasound abdomen shows gallstones","Radiology"),
            ("MRI brain shows tumor in temporal lobe","Radiology"),
            ("Mammogram shows dense tissue calcification","Radiology"),
            ("CT scan liver reveals hypodense lesion","Radiology"),
            ("X-ray spine shows mild scoliosis curvature","Radiology"),
            ("PET scan shows metastatic lymph node activity","Radiology"),
            ("Ultrasound thyroid shows 1.2cm nodule","Radiology"),
            ("CT angiography shows pulmonary embolism","Radiology"),
            ("MRI knee shows torn ACL ligament","Radiology"),
            ("X-ray wrist shows hairline fracture distal radius","Radiology"),
            ("MRI pelvis shows ovarian cyst right side","Radiology"),
            ("CT head no intracranial hemorrhage detected","Radiology"),
            # LAB REPORT
            ("Blood test shows high glucose level diabetes","Lab Report"),
            ("Hemoglobin low patient has iron deficiency anemia","Lab Report"),
            ("White blood cell count elevated infection likely","Lab Report"),
            ("Urine shows protein and bacteria UTI","Lab Report"),
            ("Thyroid TSH very high hypothyroidism confirmed","Lab Report"),
            ("Liver enzymes ALT AST elevated hepatitis","Lab Report"),
            ("Platelet count critically low thrombocytopenia","Lab Report"),
            ("Creatinine BUN elevated chronic kidney disease","Lab Report"),
            ("Blood culture positive bacterial sepsis","Lab Report"),
            ("Cholesterol LDL 240 critically high dyslipidemia","Lab Report"),
            ("HbA1c 9.2 uncontrolled type 2 diabetes","Lab Report"),
            ("Serum ferritin low iron deficiency","Lab Report"),
            ("Vitamin D severely deficient 12 ng/ml","Lab Report"),
            ("Urine culture E coli infection positive","Lab Report"),
            ("Sodium hyponatremia critically low electrolyte","Lab Report"),
            # CARDIOLOGY
            ("ECG shows irregular heartbeat atrial fibrillation","Cardiology"),
            ("High blood pressure 180 over 110 hypertension","Cardiology"),
            ("Echocardiogram reduced ejection fraction 30 percent","Cardiology"),
            ("Chest pain shortness of breath palpitations","Cardiology"),
            ("Coronary angiogram 70 percent LAD blockage","Cardiology"),
            ("Congestive heart failure diagnosed BNP elevated","Cardiology"),
            ("Hypertension uncontrolled despite antihypertensives","Cardiology"),
            ("Troponin elevated myocardial infarction suspected","Cardiology"),
            ("Pericardial effusion detected echocardiogram","Cardiology"),
            ("Deep vein thrombosis left leg doppler positive","Cardiology"),
            ("Stress test ST depression ischemia detected","Cardiology"),
            ("Aortic stenosis severe valve replacement needed","Cardiology"),
            ("Holter monitor frequent PVCs arrhythmia","Cardiology"),
            ("Peripheral vascular disease claudication legs","Cardiology"),
            ("Patient rheumatic heart disease mitral regurgitation","Cardiology"),
            # CLINICAL NOTES
            ("Patient fever cough body pain 3 days viral","Clinical Notes"),
            ("Headache vomiting since morning migraine","Clinical Notes"),
            ("Follow up diabetes management diet counseling","Clinical Notes"),
            ("Throat infection tonsillitis antibiotics prescribed","Clinical Notes"),
            ("Post operative care appendix removal recovery","Clinical Notes"),
            ("Child high fever 104F skin rash dengue suspected","Clinical Notes"),
            ("Seasonal allergies rhinitis runny nose sneezing","Clinical Notes"),
            ("Fatigue loss of appetite weakness general malaise","Clinical Notes"),
            ("Acute gastroenteritis vomiting diarrhea dehydration","Clinical Notes"),
            ("Asthma exacerbation wheezing nebulization given","Clinical Notes"),
            ("Urinary tract infection burning micturition frequency","Clinical Notes"),
            ("Anxiety disorder panic attacks referred psychiatry","Clinical Notes"),
            ("Type 2 diabetes newly diagnosed fasting sugar 280","Clinical Notes"),
            ("Hypertension follow up BP 160 over 100 controlled","Clinical Notes"),
            ("Patient has high BP and diabetes with cholesterol","Clinical Notes"),
            # NEUROLOGY
            ("Patient has recurrent seizures epilepsy EEG abnormal","Neurology"),
            ("Stroke right sided weakness facial drooping FAST","Neurology"),
            ("Parkinson disease tremor rigidity bradykinesia","Neurology"),
            ("Multiple sclerosis MRI white matter lesions demyelination","Neurology"),
            ("Severe migraine with aura photophobia vomiting","Neurology"),
            ("Alzheimer dementia memory loss cognitive decline","Neurology"),
            ("Meningitis neck stiffness fever photophobia CSF","Neurology"),
            ("Guillain Barre syndrome ascending paralysis weakness","Neurology"),
            ("Brain hemorrhage subarachnoid bleed CT scan","Neurology"),
            ("Neuropathy peripheral tingling numbness feet hands","Neurology"),
            ("Vertigo BPPV dizziness balance disorder","Neurology"),
            ("Encephalitis viral brain inflammation confusion","Neurology"),
            ("Trigeminal neuralgia facial pain shooting electric","Neurology"),
            ("Cerebral palsy spasticity motor delay child","Neurology"),
            ("Bell palsy facial nerve weakness one side","Neurology"),
            # ORTHOPEDICS
            ("Knee osteoarthritis joint pain stiffness swelling","Orthopedics"),
            ("Hip replacement surgery post operative fracture neck","Orthopedics"),
            ("Lumbar disc herniation sciatica back pain radiating leg","Orthopedics"),
            ("Cervical spondylosis neck pain radiculopathy","Orthopedics"),
            ("Rheumatoid arthritis multiple joints swollen morning stiffness","Orthopedics"),
            ("Ankle sprain ligament tear lateral instability","Orthopedics"),
            ("Rotator cuff tear shoulder pain overhead difficulty","Orthopedics"),
            ("Bone tuberculosis Pott disease spine","Orthopedics"),
            ("Carpal tunnel syndrome wrist pain tingling median nerve","Orthopedics"),
            ("Gout uric acid crystal deposition big toe painful","Orthopedics"),
            ("Scoliosis spine curvature adolescent brace treatment","Orthopedics"),
            ("Tendinitis achilles heel pain inflammation","Orthopedics"),
            ("Stress fracture metatarsal foot runner athlete","Orthopedics"),
            ("Club foot congenital deformity infant casting","Orthopedics"),
            ("Osteomyelitis bone infection fever bone pain","Orthopedics"),
            # DERMATOLOGY
            ("Psoriasis plaques scaly red patches elbows knees","Dermatology"),
            ("Eczema atopic dermatitis itchy rash dry skin","Dermatology"),
            ("Acne vulgaris comedones pustules face back","Dermatology"),
            ("Fungal infection ringworm tinea corporis antifungal","Dermatology"),
            ("Urticaria hives allergic reaction wheals itching","Dermatology"),
            ("Melanoma skin cancer suspicious mole biopsy","Dermatology"),
            ("Vitiligo depigmentation white patches autoimmune","Dermatology"),
            ("Rosacea facial redness flushing papules","Dermatology"),
            ("Contact dermatitis allergic rash nickel soap","Dermatology"),
            ("Alopecia hair loss scalp autoimmune","Dermatology"),
            ("Seborrheic dermatitis dandruff scalp oily","Dermatology"),
            ("Scabies mite infestation itching night worse","Dermatology"),
            ("Herpes zoster shingles painful rash vesicles","Dermatology"),
            ("Drug rash adverse reaction skin eruption medication","Dermatology"),
            ("Cellulitis bacterial skin infection warmth redness","Dermatology"),
            # PEDIATRICS
            ("Child fever convulsion febrile seizure 2 years old","Pediatrics"),
            ("Infant not gaining weight failure to thrive feeding","Pediatrics"),
            ("Child developmental delay speech motor milestone","Pediatrics"),
            ("Neonatal jaundice bilirubin phototherapy newborn","Pediatrics"),
            ("Child asthma wheezing recurrent bronchospasm","Pediatrics"),
            ("Measles koplik spots rash fever child unvaccinated","Pediatrics"),
            ("Chickenpox varicella vesicular rash child","Pediatrics"),
            ("Child diarrhea dehydration ORS zinc rotavirus","Pediatrics"),
            ("ADHD attention deficit hyperactivity child school","Pediatrics"),
            ("Autism spectrum disorder communication social child","Pediatrics"),
            ("Tonsillitis recurrent child sore throat fever","Pediatrics"),
            ("Pneumonia child breathing difficulty chest indrawing","Pediatrics"),
            ("Meningitis child bulging fontanelle fever stiff neck","Pediatrics"),
            ("Kawasaki disease fever rash strawberry tongue child","Pediatrics"),
            ("Child malnutrition stunting wasting underweight","Pediatrics"),
        ]
        texts=[d[0] for d in data]; labels=[d[1] for d in data]

    X_train,X_test,y_train,y_test=train_test_split(texts,labels,test_size=0.15,random_state=42)
    model=Pipeline([
        ('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=10000,sublinear_tf=True)),
        ('clf',MultinomialNB(alpha=0.3))
    ])
    model.fit(X_train,y_train)
    acc=round(accuracy_score(y_test,model.predict(X_test))*100,1)
    model.fit(texts,labels)
    return model,acc,len(texts)

# ── HELPER FUNCTIONS ─────────────────────────────────────────
def clean_text(t):
    t=t.lower(); t=re.sub(r'[^a-z0-9\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

@st.cache_data(show_spinner=False)
def classify_cached(text):
    cat=model.predict([clean_text(text)])[0]
    proba=model.predict_proba([clean_text(text)])[0]
    conf=round(max(proba)*100)
    all_p={c:round(p*100,1) for c,p in zip(model.classes_,proba)}
    return cat,conf,all_p

def get_severity(conf):
    if conf>=80: return "Critical","sev-critical","🔴","Immediate attention required"
    elif conf>=55: return "Moderate","sev-moderate","🟡","Close monitoring needed"
    else: return "Normal","sev-normal","🟢","Routine follow-up"

def get_cat_info(cat):
    d={
        "Radiology":     {"icon":"🩻","color":"#4d9fff","bg":"linear-gradient(135deg,rgba(77,159,255,0.18),rgba(30,28,94,0.95))","border":"rgba(77,159,255,0.5)","dept":"Radiology & Imaging","bc":"b-rad"},
        "Lab Report":    {"icon":"🔬","color":"#00d4b4","bg":"linear-gradient(135deg,rgba(0,212,180,0.18),rgba(30,28,94,0.95))","border":"rgba(0,212,180,0.5)","dept":"Pathology Lab","bc":"b-lab"},
        "Cardiology":    {"icon":"❤️","color":"#ff6b9d","bg":"linear-gradient(135deg,rgba(255,107,157,0.18),rgba(30,28,94,0.95))","border":"rgba(255,107,157,0.5)","dept":"Cardiology Department","bc":"b-card"},
        "Clinical Notes":{"icon":"🩺","color":"#f5a623","bg":"linear-gradient(135deg,rgba(245,166,35,0.18),rgba(30,28,94,0.95))","border":"rgba(245,166,35,0.5)","dept":"General Medicine","bc":"b-clin"},
        "Neurology":     {"icon":"🧠","color":"#b464ff","bg":"linear-gradient(135deg,rgba(180,100,255,0.18),rgba(30,28,94,0.95))","border":"rgba(180,100,255,0.5)","dept":"Neurology Department","bc":"b-neuro"},
        "Orthopedics":   {"icon":"🦴","color":"#64c8ff","bg":"linear-gradient(135deg,rgba(100,200,255,0.18),rgba(30,28,94,0.95))","border":"rgba(100,200,255,0.5)","dept":"Orthopedics Department","bc":"b-ortho"},
        "Dermatology":   {"icon":"🧬","color":"#ff9664","bg":"linear-gradient(135deg,rgba(255,150,100,0.18),rgba(30,28,94,0.95))","border":"rgba(255,150,100,0.5)","dept":"Dermatology Department","bc":"b-derm"},
        "Pediatrics":    {"icon":"👶","color":"#64ff96","bg":"linear-gradient(135deg,rgba(100,255,150,0.18),rgba(30,28,94,0.95))","border":"rgba(100,255,150,0.5)","dept":"Pediatrics Department","bc":"b-peds"},
    }
    return d.get(cat,{"icon":"🏥","color":"#9b98cc","bg":"rgba(155,152,204,0.12)","border":"rgba(155,152,204,0.3)","dept":"General","bc":"b-clin"})

def get_explanation(cat,conf):
    e={
        "Radiology":f"AI detected imaging keywords (scan, X-ray, MRI, fracture) with {conf}% confidence. Radiologist review recommended.",
        "Lab Report":f"AI detected lab test values (blood levels, glucose, enzymes) with {conf}% confidence. Pathologist review needed.",
        "Cardiology":f"AI detected cardiac keywords (ECG, BP, chest pain, heart) with {conf}% confidence. {'Urgent cardiology consult required.' if conf>75 else 'Cardiologist evaluation recommended.'}",
        "Clinical Notes":f"AI detected general clinical symptoms (fever, pain, cough) with {conf}% confidence. General medicine care recommended.",
        "Neurology":f"AI detected neurological keywords (seizure, stroke, tremor, brain) with {conf}% confidence. Neurology specialist consultation required.",
        "Orthopedics":f"AI detected musculoskeletal keywords (joint, bone, fracture, pain) with {conf}% confidence. Orthopedic evaluation recommended.",
        "Dermatology":f"AI detected skin-related keywords (rash, skin, itch, lesion) with {conf}% confidence. Dermatologist consultation advised.",
        "Pediatrics":f"AI detected pediatric keywords (child, infant, fever, developmental) with {conf}% confidence. Pediatrician review required.",
    }
    return e.get(cat,"AI analyzed report and matched medical keyword patterns.")

PRESCRIPTIONS={
    "Radiology":{"default":{"medicines":["Tab. Analgesic 400mg — Twice daily after food","Tab. Calcium + D3 — Once daily","Topical anti-inflammatory gel — Twice daily"],"advice":["Rest the affected area completely","Follow radiologist recommendations","Repeat imaging if symptoms worsen","Attend all follow-up appointments"],"diet":["Calcium rich: milk, yogurt, cheese","Vitamin D: eggs, fish, sunlight","Avoid alcohol and smoking","Stay hydrated 2-3L daily"],"followup":"2-4 weeks"},"fracture":{"medicines":["Tab. Ibuprofen 400mg — Twice daily after food","Tab. Calcium + D3 — Once daily","Oint. Diclofenac Gel — Apply 3x daily"],"advice":["Immobilize with splint or cast","Ice pack 20 min every 4 hours","No weight bearing on fracture site","X-ray follow-up after 4 weeks"],"diet":["High calcium: milk, yogurt, cheese, broccoli","Vitamin D: eggs, fish, 15 min sunlight","Protein rich foods for bone healing","Avoid alcohol and smoking"],"followup":"4 weeks"},"tumor":{"medicines":["URGENT — Refer Oncology immediately","Tab. Dexamethasone 4mg — As directed","Tab. Pantoprazole 40mg — Before breakfast"],"advice":["Urgent oncology consultation TODAY","MRI with contrast recommended","Biopsy needed for confirmation","Do NOT delay — time sensitive"],"diet":["High protein for strength","Antioxidants: berries, turmeric, green tea","Avoid processed foods","3L water daily"],"followup":"URGENT — 1 week"}},
    "Lab Report":{"default":{"medicines":["Tab. Multivitamin — Once daily after food","Tab. Probiotic — Twice daily","Consult doctor for specific medicines"],"advice":["Repeat lab after treatment course","Maintain healthy lifestyle","Stay hydrated 2-3L daily","Regular health checkups"],"diet":["Balanced diet all nutrients","Fresh fruits and vegetables","Avoid junk food","Reduce salt and sugar"],"followup":"2-4 weeks"},"diabetes":{"medicines":["Tab. Metformin 500mg — Twice daily after food","Tab. Glimepiride 1mg — Before breakfast","Tab. Vitamin B12 500mcg — Once daily"],"advice":["Monitor blood sugar daily","HbA1c every 3 months","Regular foot and eye checkup","Exercise 30 min daily"],"diet":["Avoid sugar white rice maida","Whole grains vegetables dal","Small frequent meals 5-6/day","No alcohol no fruit juices"],"followup":"1 month"},"anemia":{"medicines":["Tab. Ferrous Sulfate 200mg — Twice daily","Tab. Folic Acid 5mg — Once daily","Tab. Vitamin B12 500mcg — Once daily"],"advice":["Iron on empty stomach","Avoid tea with iron tablets","Check Hb after 4 weeks","Transfusion if Hb below 7"],"diet":["Spinach beetroot pomegranate dates","Vitamin C: lemon orange amla","Avoid tea after meals","Jaggery raisins dried figs"],"followup":"4 weeks"},"cholesterol":{"medicines":["Tab. Atorvastatin 10mg — Once daily night","Tab. Omega-3 1000mg — Twice daily","Tab. Aspirin 75mg — Once daily"],"advice":["Lipid profile every 3 months","Exercise 45 min daily","Quit smoking and alcohol","Weight management BMI below 25"],"diet":["Avoid fried fatty foods","Oats nuts olive oil","Fruits: berries citrus apples","No egg yolk no red meat"],"followup":"3 months"}},
    "Cardiology":{"default":{"medicines":["Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily night","Tab. Metoprolol 25mg — Twice daily"],"advice":["Cardiology consultation urgent","ECG and Echo recommended","Avoid strenuous exercise","Report chest pain immediately"],"diet":["Heart healthy low fat diet","Low salt low fat strictly","Avoid fried oily foods","Plenty fruits and vegetables"],"followup":"2 weeks"},"hypertension":{"medicines":["Tab. Amlodipine 5mg — Once daily morning","Tab. Telmisartan 40mg — Once daily","Tab. Aspirin 75mg — Once daily after food","Tab. Atorvastatin 10mg — Once daily night"],"advice":["Monitor BP twice daily","Target below 130/80 mmHg","EMERGENCY if BP above 180/120","Walk 30 min daily","Avoid all stress"],"diet":["DASH diet fruits vegetables grains","Avoid salt pickles sauces completely","No alcohol no smoking","Limit caffeine 1 cup/day","Banana watermelon beetroot"],"followup":"2 weeks"},"heart":{"medicines":["Tab. Metoprolol 25mg — Twice daily","Tab. Ramipril 5mg — Once daily","Tab. Furosemide 40mg — Morning","Tab. Spironolactone 25mg — Once daily"],"advice":["URGENT cardiology OPD immediately","Avoid all physical exertion","Sleep head elevated 30 degrees","Report chest pain — EMERGENCY","Daily weight monitoring"],"diet":["Low sodium less than 2g/day","Fluid restriction as advised","Heart healthy fish nuts olive oil","No caffeine no alcohol"],"followup":"1 week — URGENT"}},
    "Clinical Notes":{"default":{"medicines":["Tab. Paracetamol 500mg — SOS","Tab. Vitamin C 500mg — Once daily","Tab. Zinc 20mg — Once daily","ORS sachet as needed"],"advice":["Rest at home 2-3 days","Monitor symptoms closely","Drink warm fluids throughout day","Return if no improvement in 3 days"],"diet":["Light digestible food khichdi soup","Warm soups and broths","Fresh fruits for immunity","Avoid junk cold drinks"],"followup":"3-5 days"},"fever":{"medicines":["Tab. Paracetamol 500mg — Three times daily","Tab. Cetirizine 10mg — Once night","Syp. Benadryl 10ml — Three times daily","ORS sachet — 1L water all day"],"advice":["Complete bed rest 3-5 days","Tepid sponging if fever above 102F","Drink 3-4 litres fluids daily","Return if fever above 104F"],"diet":["Light: khichdi soup boiled rice","Coconut water ORS lemon juice","Avoid spicy oily food","Small frequent meals"],"followup":"3-5 days"},"infection":{"medicines":["Tab. Amoxicillin 500mg — 3x daily 5 days","Tab. Ibuprofen 400mg — Twice daily","Tab. Probiotic — Twice daily","Tab. Paracetamol 500mg — SOS"],"advice":["Complete full antibiotic course","Never stop antibiotics early","Rest avoid exertion","Maintain strict hand hygiene"],"diet":["Turmeric milk ginger tea daily","Vitamin C: oranges lemons amla","Avoid cold drinks ice cream","Plenty warm fluids and soups"],"followup":"5-7 days"}},
    "Neurology":{"default":{"medicines":["Consult Neurologist immediately","Tab. as prescribed by neurologist","Do NOT self-medicate for neurological conditions"],"advice":["Urgent neurology consultation required","MRI brain may be needed","Do not drive until cleared by doctor","Take all medicines on time without fail"],"diet":["Brain healthy: fish walnuts berries","Omega-3 rich foods daily","Avoid alcohol completely","Stay hydrated 2-3L daily"],"followup":"1-2 weeks — URGENT"},"seizure":{"medicines":["Tab. Levetiracetam 500mg — Twice daily (as prescribed)","Tab. Valproate — As per neurologist","Tab. Clonazepam — SOS only","Never stop AEDs without doctor advice"],"advice":["Never miss a dose of medication","Avoid swimming alone driving heights","Seizure first aid: side position, do not restrain","Wear medical alert bracelet","Regular EEG monitoring"],"diet":["Regular meals no fasting","Ketogenic diet if prescribed","Avoid alcohol completely","Adequate sleep 8 hours minimum"],"followup":"1 month"},"migraine":{"medicines":["Tab. Sumatriptan 50mg — At onset of migraine","Tab. Naproxen 500mg — Twice daily during attack","Tab. Propranolol 40mg — Daily preventive","Tab. Amitriptyline 10mg — Night preventive"],"advice":["Identify and avoid triggers: bright light stress","Rest in dark quiet room during attack","Sleep and wake same time daily","Stress management yoga meditation","Keep migraine diary"],"diet":["Avoid triggers: cheese chocolate wine caffeine","Regular meals avoid skipping","Stay hydrated 2-3L water","Magnesium rich: nuts seeds leafy greens"],"followup":"3-4 weeks"}},
    "Orthopedics":{"default":{"medicines":["Tab. Ibuprofen 400mg — Twice daily after food","Tab. Calcium + D3 — Once daily","Oint. Diclofenac Gel — Apply 3x daily","Tab. Pantoprazole 40mg — Before breakfast"],"advice":["Rest the affected joint","Hot or cold compress as needed","Physiotherapy recommended","Avoid heavy lifting and strain"],"diet":["Calcium rich: milk yogurt cheese","Vitamin D: eggs fish sunlight","Anti-inflammatory: turmeric ginger","Avoid processed and junk food"],"followup":"2-3 weeks"},"arthritis":{"medicines":["Tab. Diclofenac 50mg — Twice daily after food","Tab. Methotrexate — As per rheumatologist","Tab. Folic Acid 5mg — Once daily","Oint. Volini — Apply 3x daily"],"advice":["Physiotherapy 3x per week","Warm water exercises swimming","Avoid joint overuse and strain","Use walking aids if needed","Regular rheumatology review"],"diet":["Anti-inflammatory: fish turmeric","Avoid red meat processed food","Calcium vitamin D essential","Omega-3: walnuts flaxseeds fish"],"followup":"4-6 weeks"},"gout":{"medicines":["Tab. Allopurinol 100mg — Once daily","Tab. Colchicine 0.5mg — During flare","Tab. Indomethacin 25mg — During flare","Drink 3L water daily"],"advice":["Avoid purine rich foods strictly","Keep uric acid below 6 mg/dL","Stay well hydrated always","Elevate affected joint during flare"],"diet":["Avoid: red meat seafood beer alcohol","Eat: cherries low fat dairy vegetables","Drink 3L water minimum daily","Coffee in moderation may help"],"followup":"4 weeks"}},
    "Dermatology":{"default":{"medicines":["Topical steroid cream — Apply twice daily","Tab. Cetirizine 10mg — Once night","Moisturizer — Apply after bath","Consult dermatologist for prescription"],"advice":["Avoid scratching the rash","Use mild soap and lukewarm water","Avoid known allergens and triggers","Wear loose cotton clothing"],"diet":["Avoid spicy oily foods","Drink 2-3L water daily","Fresh fruits and vegetables","Avoid dairy if dairy allergy"],"followup":"2 weeks"},"eczema":{"medicines":["Topical Betamethasone cream — Twice daily affected area","Tab. Cetirizine 10mg — Once night","Moisturizer Cetaphil — Apply 3x daily","Oint. Tacrolimus — As prescribed"],"advice":["Avoid hot showers use lukewarm","Wear cotton avoid wool synthetic","Cut nails short to prevent scratching","Avoid known triggers allergens","Humidifier in dry weather"],"diet":["Identify food triggers diary","Avoid dairy eggs nuts if allergic","Omega-3: fish flaxseeds walnuts","Probiotic yogurt helps skin"],"followup":"2-3 weeks"},"fungal":{"medicines":["Tab. Fluconazole 150mg — Once weekly 4 weeks","Oint. Clotrimazole — Apply twice daily","Antifungal dusting powder — Daily","Keep affected area dry always"],"advice":["Keep skin clean and dry always","Wear loose breathable cotton","Do not share towels clothes","Complete full course of treatment","Avoid sweating in affected area"],"diet":["Avoid sugar and yeast products","Probiotics: yogurt helps","Stay hydrated 2-3L water","Garlic turmeric natural antifungal"],"followup":"4 weeks"}},
    "Pediatrics":{"default":{"medicines":["Syp. Paracetamol — As per weight SOS","Syp. Cetirizine — As per weight once night","ORS — As per dehydration","Consult pediatrician for prescription"],"advice":["Monitor temperature every 4 hours","Keep child well hydrated","Rest at home avoid school","Return if fever above 104F or worsening"],"diet":["Breastfeed infants continue","ORS for hydration if diarrhea","Light easily digestible food","Plenty of warm fluids"],"followup":"3-5 days"},"fever":{"medicines":["Syp. Paracetamol 15mg/kg — 4-6 hourly","Syp. Ibuprofen — Alternate with Paracetamol","ORS sachets — Frequent sips","Syp. Antibiotic if bacterial infection"],"advice":["Tepid sponge if temp above 102F","Do not cover with heavy blankets","Keep child hydrated with ORS","Hospitalize if no improvement 3 days","Watch for danger signs: convulsion difficulty breathing"],"diet":["Continue breastfeeding infants","Light food: khichdi soup rice","Plenty of fluids coconut water","Avoid solid if vomiting present"],"followup":"2-3 days"},"respiratory":{"medicines":["Syp. Salbutamol nebulization — 4-6 hourly","Syp. Montelukast — Once night","Budesonide inhaler — Twice daily","Syp. Amoxicillin — If bacterial"],"advice":["Elevate head of bed slightly","Steam inhalation twice daily","Avoid smoke dust allergens","Count respiratory rate — above 60 hospital","Pulse oximeter check oxygen level"],"diet":["Warm soups broths honey lemon","Continue breastfeeding infants","Avoid cold drinks ice cream","Vitamin C for immunity"],"followup":"3-5 days"}}
}

def get_auto_prescription(cat,text):
    t=text.lower(); db=PRESCRIPTIONS.get(cat,PRESCRIPTIONS["Clinical Notes"])
    kmap={
        "Cardiology":{"hypertension":["high bp","blood pressure","hypertension","htn"],"heart":["heart failure","cardiac failure","ejection fraction","congestive"]},
        "Lab Report":{"diabetes":["diabetes","glucose","sugar","hba1c","diabetic"],"anemia":["anemia","hemoglobin","iron deficiency","hb low"],"cholesterol":["cholesterol","ldl","lipid","triglyceride"]},
        "Radiology":{"fracture":["fracture","broken","crack","dislocation"],"tumor":["tumor","cancer","malignant","metastatic","mass","lesion"]},
        "Clinical Notes":{"fever":["fever","temperature","viral","flu","cold"],"infection":["infection","bacteria","antibiotic","sepsis"]},
        "Neurology":{"seizure":["seizure","epilepsy","convulsion","fits"],"migraine":["migraine","headache","aura","photophobia"]},
        "Orthopedics":{"arthritis":["arthritis","rheumatoid","joint pain","osteoarthritis"],"gout":["gout","uric acid","tophi"]},
        "Dermatology":{"eczema":["eczema","atopic","dermatitis","itchy rash"],"fungal":["fungal","ringworm","tinea","candida"]},
        "Pediatrics":{"fever":["child fever","infant fever","febrile","high temperature child"],"respiratory":["wheezing","asthma child","breathing difficulty","nebulization"]},
    }
    for key,words in kmap.get(cat,{}).items():
        if any(w in t for w in words):
            return db.get(key,db["default"])
    return db["default"]

def extract_text_from_image(img):
    try: return pytesseract.image_to_string(img,config='--psm 6').strip()
    except: return ""

# ── LOAD MODEL ONCE ──────────────────────────────────────────
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
            if st.button("🚪 Logout"): st.session_state.logged_in=False; st.session_state.page="login"; st.session_state.auth_mode="login"; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #3d3a8a;margin:0 0 20px 0'>",unsafe_allow_html=True)

# ── AUTH PAGE (LOGIN + SIGNUP) ────────────────────────────────
def show_auth():
    st.markdown('<div style="text-align:center;padding:30px 0 6px;"><div style="display:inline-block;filter:drop-shadow(0 0 28px rgba(245,166,35,0.55))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:#f5a623;letter-spacing:0.05em;margin-top:8px;">MEDICLASSIFY</div><div style="font-size:13px;color:#b0aee8;font-style:italic;margin-top:5px;">Diagnose Faster. Treat Better.</div></div>',unsafe_allow_html=True)

    _,mid,_=st.columns([1,1.1,1])
    with mid:
        # Tab toggle
        t1,t2=st.columns(2)
        with t1:
            if st.button("🔐  Login",key="tab_login"):
                st.session_state.auth_mode="login"; st.rerun()
        with t2:
            if st.button("📝  Sign Up",key="tab_signup"):
                st.session_state.auth_mode="signup"; st.rerun()

        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

        if st.session_state.auth_mode=="login":
            st.markdown('<div style="background:#1e1c5e;border:1.5px solid #4a47a3;border-radius:20px;padding:28px 32px;"><div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#f5a623;margin-bottom:16px;text-align:center;">Welcome Back 👋</div></div>',unsafe_allow_html=True)
            username=st.text_input("👤  Username",placeholder="Enter your username",key="li_u")
            password=st.text_input("🔑  Password",placeholder="Enter your password",type="password",key="li_p")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Login to MediClassify  →",key="login_btn"):
                if username in st.session_state.users and st.session_state.users[username]==password:
                    st.session_state.logged_in=True; st.session_state.username=username; st.session_state.page="home"; st.rerun()
                else: st.error("❌ Wrong username or password!")
            st.markdown('<div style="margin-top:14px;padding:12px 16px;background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);border-radius:12px;font-size:12px;color:#b0aee8;"><strong style="color:#f5a623;">Demo:</strong> admin/admin123 &nbsp;|&nbsp; doctor/medi2024 &nbsp;|&nbsp; student/project123</div>',unsafe_allow_html=True)

        else:
            st.markdown('<div style="background:#1e1c5e;border:1.5px solid #4a47a3;border-radius:20px;padding:28px 32px;"><div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:#f5a623;margin-bottom:16px;text-align:center;">Create Account 🏥</div></div>',unsafe_allow_html=True)
            su_name=st.text_input("👤  Full name",placeholder="Enter your full name",key="su_n")
            su_user=st.text_input("🪪  Username",placeholder="Choose a username",key="su_u")
            su_email=st.text_input("📧  Email",placeholder="Enter your email",key="su_e")
            su_role=st.selectbox("🏥  Role",["","Doctor","Nurse","Lab Technician","Radiologist","Student","Other"],key="su_r")
            su_pass=st.text_input("🔑  Password",placeholder="Choose a password (min 6 chars)",type="password",key="su_p")
            su_pass2=st.text_input("🔑  Confirm password",placeholder="Re-enter your password",type="password",key="su_p2")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Create Account  →",key="signup_btn"):
                if not su_name or not su_user or not su_email or not su_role or not su_pass:
                    st.error("❌ Please fill all fields!")
                elif "@" not in su_email:
                    st.error("❌ Enter a valid email address!")
                elif len(su_pass)<6:
                    st.error("❌ Password must be at least 6 characters!")
                elif su_pass!=su_pass2:
                    st.error("❌ Passwords do not match!")
                elif su_user in st.session_state.users:
                    st.error("❌ Username already exists! Choose another.")
                else:
                    st.session_state.users[su_user]=su_pass
                    st.session_state.logged_in=True
                    st.session_state.username=su_user
                    st.session_state.page="home"
                    st.success("✅ Account created! Welcome to MediClassify!")
                    st.rerun()
            st.markdown('<div style="margin-top:14px;padding:10px 14px;background:rgba(100,255,150,0.06);border:1px solid rgba(100,255,150,0.2);border-radius:10px;font-size:12px;color:#b0aee8;">Already have an account? Click <strong style="color:#f5a623;">Login</strong> above.</div>',unsafe_allow_html=True)

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; 8 Departments &nbsp;|&nbsp; Placement Mini Project</div>',unsafe_allow_html=True)

# ── HOME ─────────────────────────────────────────────────────
def show_home():
    show_navbar()
    st.markdown('<div style="text-align:center;padding:20px 0 14px;"><div style="display:inline-block;filter:drop-shadow(0 0 22px rgba(245,166,35,0.4))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:40px;font-weight:800;color:#f5a623;margin-top:8px;letter-spacing:0.05em;">MEDICLASSIFY</div><div style="font-size:14px;color:#8886c8;font-style:italic;margin:5px 0 10px;">Diagnose Faster. Treat Better.</div><div style="font-size:13px;color:rgba(240,240,255,0.75);max-width:520px;margin:0 auto;line-height:1.7;">Welcome back, <strong style="color:#f5a623">'+st.session_state.username+'</strong>! 👋<br>8-department AI classification with auto-prescription.</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    features=[("🤖","AI Classification","Naive Bayes TF-IDF 8 departments"),("🖼️","OCR Support","Auto reads prescription images"),("💊","Auto Prescription","Smart medicines from report"),("🔴🟡🟢","Severity Detection","Critical Moderate Normal"),("📊","Gauge + Bar Charts","Advanced Plotly visualizations"),("📋","History Log","Complete patient records")]
    cols=st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:14px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+title+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec-title">8 Medical departments supported</div>',unsafe_allow_html=True)
    depts=[("🩻","Radiology"),("🔬","Lab Report"),("❤️","Cardiology"),("🩺","Clinical Notes"),("🧠","Neurology"),("🦴","Orthopedics"),("🧬","Dermatology"),("👶","Pediatrics")]
    dcols=st.columns(8)
    for col,(icon,name) in zip(dcols,depts):
        with col:
            st.markdown('<div style="background:#1e1c5e;border:1px solid #3d3a8a;border-radius:14px;padding:14px 8px;text-align:center;"><div style="font-size:28px;margin-bottom:6px;">'+icon+'</div><div style="font-size:11px;color:#b0aee8;font-weight:500;">'+name+'</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    _,mid,_=st.columns([1,1,1])
    with mid:
        if st.button("🏥  Open Dashboard →"): st.session_state.page="dashboard"; st.rerun()

def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About MediClassify</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Project overview, 8 departments, and technologies</div>',unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div class="sec-title">Project overview</div><p style="font-size:14px;color:rgba(220,220,255,0.85);line-height:1.85;"><strong style="color:#f5a623">MediClassify v3.0</strong> is an AI-powered Medical Report Classification System trained on 8 departments. It reads medical reports or prescription images, classifies into the correct department, auto-generates smart prescription, detects severity, and shows advanced gauge and bar chart visualizations — all powered by NLP and Naive Bayes ML.</p></div>',unsafe_allow_html=True)
    techs=[("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes TF-IDF"),("📊","Streamlit","Web UI"),("🖼️","Tesseract OCR","Image to text"),("📈","Plotly","Gauge bar charts"),("📁","CSV Dataset","Training data")]
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

def render_result(r):
    category=r["category"]; confidence=r["confidence"]; all_proba=r["all_proba"]
    severity=r["severity"]; sev_class=r["sev_class"]; sev_icon=r["sev_icon"]; sev_msg=r["sev_msg"]
    explanation=r["explanation"]; info=r["info"]; rx_data=r["rx_data"]
    name=r["name"]; meta=r["meta"]
    color=info["color"]; bg=info["bg"]; border=info["border"]

    # RESULT CARD
    st.markdown(
        '<div style="background:'+bg+';border:2px solid '+border+';border-radius:22px;overflow:hidden;margin-bottom:18px;">'
        '<div style="padding:22px 26px;border-bottom:1px solid '+border+'30;display:flex;align-items:center;justify-content:space-between;">'
        '<div><div style="font-size:10px;color:'+color+';text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px;opacity:0.8;">✦ AI Classification Result</div>'
        '<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:'+color+';">'+category+'</div>'
        '<div style="font-size:12px;color:rgba(255,255,255,0.7);margin-top:4px;">'+name+((" &nbsp;·&nbsp; "+meta) if meta else "")+'</div>'
        '<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">'
        '<span class="sev-pill '+sev_class+'">'+sev_icon+' '+severity+' — '+sev_msg+'</span>'
        '<span style="background:rgba(255,255,255,0.08);padding:5px 14px;border-radius:20px;font-size:11px;color:rgba(255,255,255,0.8);">🏥 '+info["dept"]+'</span>'
        '</div></div><div style="font-size:60px;filter:drop-shadow(0 0 14px '+color+'80);">'+info["icon"]+'</div></div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;">'
        '<div style="padding:14px 18px;text-align:center;border-right:1px solid '+border+'30;"><div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Confidence</div><div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:'+color+';">'+str(confidence)+'%</div></div>'
        '<div style="padding:14px 18px;text-align:center;border-right:1px solid '+border+'30;"><div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Severity</div><div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;color:'+color+';">'+sev_icon+' '+severity+'</div></div>'
        '<div style="padding:14px 18px;text-align:center;"><div style="font-size:10px;color:'+color+';opacity:0.8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Follow-up</div><div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;color:'+color+';">'+rx_data["followup"]+'</div></div>'
        '</div>'
        '<div style="padding:16px 22px;border-top:1px solid '+border+'30;">'
        '<div style="font-size:10px;color:'+color+';font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:7px;">🤖 AI Explanation</div>'
        '<div style="font-size:13px;color:rgba(255,255,255,0.85);line-height:1.75;">'+explanation+'</div>'
        '</div></div>',unsafe_allow_html=True)

    # CHARTS
    ch1,ch2=st.columns(2)
    with ch1:
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number",value=confidence,
            title={"text":"Confidence Score","font":{"color":color,"size":12,"family":"DM Sans"}},
            number={"suffix":"%","font":{"color":color,"size":36,"family":"Syne"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#8886c8","tickfont":{"color":"#8886c8","size":9}},"bar":{"color":color,"thickness":0.28},"bgcolor":"rgba(26,24,78,0.9)","borderwidth":1,"bordercolor":border,
                   "steps":[{"range":[0,40],"color":"rgba(0,212,100,0.1)"},{"range":[40,70],"color":"rgba(255,179,71,0.1)"},{"range":[70,100],"color":"rgba(255,70,70,0.1)"}],
                   "threshold":{"line":{"color":color,"width":4},"thickness":0.78,"value":confidence}}
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"white","family":"DM Sans"},height=220,margin=dict(t=40,b=8,l=16,r=16))
        st.plotly_chart(fig_g,use_container_width=True)

    with ch2:
        cats=list(all_proba.keys()); vals=list(all_proba.values())
        bcolors={"Radiology":"#4d9fff","Lab Report":"#00d4b4","Cardiology":"#ff6b9d","Clinical Notes":"#f5a623","Neurology":"#b464ff","Orthopedics":"#64c8ff","Dermatology":"#ff9664","Pediatrics":"#64ff96"}
        cl=[bcolors.get(c,"#9b98cc") for c in cats]
        fig_b=go.Figure(go.Bar(y=cats,x=vals,orientation='h',marker_color=cl,marker_line_width=0,text=[str(v)+"%" for v in vals],textposition="outside",textfont=dict(color="white",size=10)))
        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(26,24,78,0.5)",font=dict(color="white",family="DM Sans"),title=dict(text="Department breakdown",font=dict(color=color,size=12)),xaxis=dict(range=[0,120],ticksuffix="%",gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#8886c8",size=9)),yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color="white",size=10)),showlegend=False,height=220,margin=dict(t=34,b=8,l=8,r=44))
        st.plotly_chart(fig_b,use_container_width=True)

    # PRESCRIPTION
    rx_items="".join(['<div style="background:rgba(0,0,0,0.2);border:1px solid '+border+'40;border-radius:10px;padding:9px 13px;margin-bottom:7px;font-size:13px;color:rgba(255,255,255,0.9);display:flex;gap:10px;"><span style="color:'+color+';font-size:16px;flex-shrink:0;">💊</span><span>'+m+'</span></div>' for m in rx_data["medicines"]])
    advice_items="".join(['<div style="background:rgba(0,0,0,0.15);border-radius:8px;padding:7px 11px;margin-bottom:6px;font-size:12px;color:rgba(255,255,255,0.85);display:flex;gap:8px;"><span style="color:'+color+';flex-shrink:0;">✦</span><span>'+a+'</span></div>' for a in rx_data["advice"]])
    diet_items="".join(['<div style="background:rgba(0,0,0,0.15);border-radius:8px;padding:6px 11px;margin-bottom:5px;font-size:12px;color:rgba(255,255,255,0.8);">🥗 '+d+'</div>' for d in rx_data["diet"]])
    st.markdown(
        '<div style="background:'+bg+';border:2px solid '+border+';border-radius:20px;padding:20px 24px;margin-top:4px;">'
        '<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;color:'+color+';text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;display:flex;align-items:center;gap:10px;">💊 Auto-Generated Prescription <span style="font-size:10px;background:rgba(0,0,0,0.25);padding:3px 10px;border-radius:10px;color:rgba(255,255,255,0.5);font-weight:400;font-family:DM Sans;">Based on patient report</span></div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">'
        '<div><div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:9px;">Medicines</div>'+rx_items+'</div>'
        '<div><div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:9px;">Medical Advice</div>'+advice_items+'<div style="font-size:11px;color:'+color+';font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 8px;">Diet Plan</div>'+diet_items+'</div>'
        '</div>'
        '<div style="margin-top:14px;padding:9px 13px;background:rgba(0,0,0,0.25);border-radius:10px;font-size:12px;color:rgba(255,255,255,0.55);">⚠️ <strong style="color:'+color+';">Disclaimer:</strong> AI-generated suggestion. Always consult a qualified doctor.</div>'
        '</div>',unsafe_allow_html=True)

def show_dashboard():
    show_navbar()
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:20px;padding:18px 26px;margin-bottom:18px;"><div style="display:flex;align-items:center;gap:14px;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:11px;color:#8886c8;">Accuracy: <strong style="color:#f5a623">'+str(model_acc)+'%</strong> &nbsp;|&nbsp; 8 Departments</div></div></div><div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);color:#f5a623;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600;"><span class="pulse"></span>AI Active</div></div>',unsafe_allow_html=True)

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
        p_report=""
        if input_type in ["📝 Type Report","📝 + 🖼️ Both"]:
            p_report=st.text_area("Report / symptoms",placeholder="e.g. Patient has seizures epilepsy EEG abnormal\nKnee osteoarthritis joint pain swelling\nPsoriasis scaly rash elbows\nChild fever convulsion 2 years old",height=110)
        if input_type in ["🖼️ Upload Image","📝 + 🖼️ Both"]:
            uploaded_img=st.file_uploader("Upload prescription image",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uploaded_img is not None:
                img=Image.open(uploaded_img)
                st.image(img,caption="Uploaded image",use_container_width=True)
                with st.spinner("🔍 Reading image (OCR)..."):
                    ocr_text=extract_text_from_image(img)
                if ocr_text.strip():
                    st.markdown('<div class="ocr-box"><div style="font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">📄 OCR Extracted Text</div><div style="font-size:13px;color:#e0e0ff;line-height:1.7;">'+ocr_text+'</div></div>',unsafe_allow_html=True)
                    p_report=(p_report+" "+ocr_text).strip() if input_type=="📝 + 🖼️ Both" else ocr_text
                else: st.warning("⚠️ Could not extract text. Type the report manually.")
        st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
        classify_btn=st.button("▶  Classify & Generate Prescription")
        st.markdown('</div>',unsafe_allow_html=True)

        if classify_btn:
            if not p_report.strip(): st.error("⚠️ Please enter a report or upload a readable image!")
            else:
                # Fast classify — cached
                category,confidence,all_proba=classify_cached(p_report)
                severity,sev_class,sev_icon,sev_msg=get_severity(confidence)
                explanation=get_explanation(category,confidence)
                info=get_cat_info(category)
                rx_data=get_auto_prescription(category,p_report)
                name=p_name.strip() or "Unknown patient"
                meta=" · ".join(filter(None,["Age "+p_age if p_age else "",p_gender,"Dr: "+p_doc if p_doc else ""]))
                st.session_state.last_result={"category":category,"confidence":confidence,"all_proba":all_proba,"severity":severity,"sev_class":sev_class,"sev_icon":sev_icon,"sev_msg":sev_msg,"explanation":explanation,"info":info,"rx_data":rx_data,"name":name,"meta":meta}
                st.session_state.counts[category]+=1
                now=datetime.datetime.now()
                st.session_state.history.insert(0,{"name":name,"age":p_age,"gender":p_gender,"doc":p_doc,"date":str(p_date),"time":now.strftime("%I:%M %p"),"report":p_report[:60]+("..." if len(p_report)>60 else ""),"cat":category,"conf":confidence,"severity":severity,"sev_icon":sev_icon,"color":info["color"],"bc":info["bc"],"followup":rx_data["followup"]})
                st.rerun()

    with right:
        with st.expander("📊 Project flow",expanded=False):
            for i,(icon,label) in enumerate([("📄","Input report or image"),("🖼️","OCR reads image"),("📁","CSV dataset loaded"),("🤖","Naive Bayes classifies (8 dept)"),("💊","Auto prescription"),("📊","Gauge + bar charts")]):
                st.markdown('<div class="flow-step"><span style="font-size:15px;">'+icon+'</span> '+label+'</div>',unsafe_allow_html=True)
                if i<5: st.markdown('<div style="text-align:center;color:#f5a623;font-size:17px;margin:-2px 0">↓</div>',unsafe_allow_html=True)
        st.markdown('<div class="sec-title" style="margin-top:14px">Classification history</div>',unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:28px 14px;color:#8886c8;background:#1e1c5e;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:34px;margin-bottom:8px">📋</div><div>No reports classified yet.</div></div>',unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:8]:
                meta=" · ".join(filter(None,["Age "+h['age'] if h['age'] else "",h['gender'],"Dr: "+h['doc'] if h['doc'] else ""]))
                color=h.get('color','#f5a623'); bc=h.get('bc','b-clin')
                meta_line='<div class="hist-meta">'+meta+'</div>' if meta else ""
                st.markdown('<div class="hist-card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;"><span class="hist-name">'+h['name']+'</span><span class="badge '+bc+'">'+h['cat']+'</span></div>'+meta_line+'<div class="hist-rep">'+h['report']+'</div><div class="hist-time">'+h['date']+' at '+h['time']+' | <span style="color:'+color+'">'+str(h['conf'])+'% '+h.get('sev_icon','')+'</span> | '+h.get('followup','—')+'</div></div>',unsafe_allow_html=True)
            if st.button("🗑️ Clear history"):
                st.session_state.history=[]; st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0,"Neurology":0,"Orthopedics":0,"Dermatology":0,"Pediatrics":0}; st.session_state.last_result=None; st.rerun()

    if st.session_state.last_result:
        st.markdown("<hr style='border:none;border-top:2px solid #3d3a8a;margin:22px 0'>",unsafe_allow_html=True)
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:17px;font-weight:800;color:#f5a623;margin-bottom:16px;">📋 Classification Result & Prescription</div>',unsafe_allow_html=True)
        render_result(st.session_state.last_result)

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY v3.0</strong> &nbsp;|&nbsp; 8 Departments &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project</div>',unsafe_allow_html=True)

# ── ROUTER ───────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth()
else:
    if st.session_state.page=="home": show_home()
    elif st.session_state.page=="about": show_about()
    elif st.session_state.page=="contact": show_contact()
    elif st.session_state.page=="dashboard": show_dashboard()
    else: show_home()

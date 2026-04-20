import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"])

for pkg in ["scikit-learn","Pillow","pytesseract","plotly","numpy","pandas"]:
    try: __import__(pkg.replace("-","_").split("==")[0])
    except ImportError: install(pkg)

import streamlit as st
import re, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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

/* RESULT CARD */
.result-main{border-radius:18px;padding:24px 28px;margin-top:16px;}
.result-category{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;margin:8px 0 4px;}
.result-patient-info{font-size:13px;opacity:0.85;margin-bottom:12px;}
.severity-badge{display:inline-block;padding:5px 18px;border-radius:20px;font-size:12px;font-weight:700;font-family:'Syne',sans-serif;letter-spacing:0.05em;margin-right:8px;}
.sev-normal{background:rgba(0,212,100,0.15);color:#00d464;border:1px solid rgba(0,212,100,0.35);}
.sev-moderate{background:rgba(255,179,71,0.15);color:#ffb347;border:1px solid rgba(255,179,71,0.35);}
.sev-critical{background:rgba(255,70,70,0.15);color:#ff4646;border:1px solid rgba(255,70,70,0.35);}
.ai-explanation{background:rgba(255,255,255,0.04);border-left:3px solid #f5a623;border-radius:0 10px 10px 0;padding:12px 16px;margin-top:14px;font-size:13px;color:rgba(240,240,255,0.85);line-height:1.7;}
.ai-label{font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;}

/* HIST */
.hist-card{background:rgba(255,255,255,0.03);border:1px solid #3d3a8a;border-radius:13px;padding:13px 16px;margin-bottom:10px;}
.hist-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:#f0f0ff;}
.hist-meta{font-size:11px;color:#9b98cc;margin:3px 0;}
.hist-rep{font-size:12px;color:rgba(240,240,255,0.7);}
.hist-rx{font-size:11px;color:#9b98cc;font-style:italic;margin-top:3px;}
.hist-time{font-size:10px;color:#9b98cc;margin-top:4px;}
.badge{font-size:10px;font-weight:700;padding:4px 13px;border-radius:20px;font-family:'Syne',sans-serif;letter-spacing:0.04em;text-transform:uppercase;}
.b-rad{background:rgba(77,159,255,0.15);color:#4d9fff;border:1px solid rgba(77,159,255,0.3);}
.b-lab{background:rgba(0,212,180,0.15);color:#00d4b4;border:1px solid rgba(0,212,180,0.3);}
.b-card{background:rgba(255,107,157,0.15);color:#ff6b9d;border:1px solid rgba(255,107,157,0.3);}
.b-clin{background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3);}
.b-unk{background:rgba(155,152,204,0.15);color:#9b98cc;border:1px solid rgba(155,152,204,0.3);}
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
.divider{height:1px;background:#3d3a8a;margin:16px 0;}
.pulse{display:inline-block;width:7px;height:7px;background:#f5a623;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.7);}}
.ocr-box{background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.25);border-radius:14px;padding:14px 16px;margin-top:10px;font-size:13px;color:#f0f0ff;line-height:1.7;}
.ocr-label{font-size:10px;color:#f5a623;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background-color:rgba(255,255,255,0.05)!important;border:1px solid #3d3a8a!important;border-radius:11px!important;color:#f0f0ff!important;}
.stButton>button{background:linear-gradient(135deg,#f5a623,#d4881a)!important;color:#1e1c4a!important;border:none!important;border-radius:12px!important;font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:14px!important;padding:11px 24px!important;width:100%!important;box-shadow:0 4px 22px rgba(245,166,35,0.35)!important;}
label{color:#9b98cc!important;font-size:12px!important;}
</style>
""",unsafe_allow_html=True)

SHIELD="""<svg width="48" height="54" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

SHIELD_BIG="""<svg width="90" height="100" viewBox="0 0 80 90" fill="none">
  <defs><linearGradient id="sg2" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#f5c842"/><stop offset="100%" stop-color="#d4881a"/>
  </linearGradient></defs>
  <path d="M40 4L8 18V44C8 62 22 78 40 86C58 78 72 62 72 44V18L40 4Z" fill="url(#sg2)"/>
  <rect x="32" y="24" width="16" height="42" rx="5" fill="#2d2b6b"/>
  <rect x="19" y="36" width="42" height="16" rx="5" fill="#2d2b6b"/>
</svg>"""

@st.cache_resource
def train_model():
    data=[
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
    texts,labels=zip(*data)
    m=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=5000)),('clf',MultinomialNB())])
    m.fit(texts,labels)
    return m

def clean_text(t):
    t=t.lower()
    t=re.sub(r'[^a-z0-9\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

def classify_report(model,text):
    cleaned=clean_text(text)
    cat=model.predict([cleaned])[0]
    proba=model.predict_proba([cleaned])[0]
    conf=round(max(proba)*100)
    all_proba={c:round(p*100) for c,p in zip(model.classes_,proba)}
    return cat,conf,all_proba

def get_severity(cat,conf):
    if conf>=80: return "Critical","sev-critical","🔴"
    elif conf>=55: return "Moderate","sev-moderate","🟡"
    else: return "Normal","sev-normal","🟢"

def get_explanation(cat,conf,severity):
    explanations={
        "Radiology": f"The report contains imaging-related terminology suggesting a radiological finding. The AI model identified scan, imaging, or bone-related keywords with {conf}% confidence. Severity is marked {severity} — immediate radiologist review is {'recommended' if conf>75 else 'suggested'}.",
        "Lab Report": f"The report contains laboratory test result keywords such as blood levels, glucose, or cell counts. The AI model classified this as a Lab Report with {conf}% confidence. Severity {severity} indicates {'abnormal values requiring attention' if conf>75 else 'values that may need monitoring'}.",
        "Cardiology": f"The report contains cardiac-related terms such as ECG, blood pressure, or heart rate. The AI model classified this as a Cardiology case with {conf}% confidence. Severity {severity} — {'urgent cardiology consultation recommended' if conf>75 else 'cardiologist follow-up suggested'}.",
        "Clinical Notes": f"The report contains general clinical observation keywords such as fever, pain, or symptoms. The AI model classified this as Clinical Notes with {conf}% confidence. Severity {severity} — {'active treatment required' if conf>75 else 'routine monitoring recommended'}.",
    }
    return explanations.get(cat,"The AI model analyzed the report and made a classification based on medical keyword patterns.")

def extract_text_from_image(img):
    try:
        text=pytesseract.image_to_string(img,config='--psm 6')
        return text.strip()
    except Exception:
        return ""

cat_color={"Radiology":"#4d9fff","Lab Report":"#00d4b4","Cardiology":"#ff6b9d","Clinical Notes":"#f5a623","Unknown":"#9b98cc"}
cat_bg={"Radiology":"rgba(77,159,255,0.12)","Lab Report":"rgba(0,212,180,0.12)","Cardiology":"rgba(255,107,157,0.12)","Clinical Notes":"rgba(245,166,35,0.12)","Unknown":"rgba(155,152,204,0.12)"}
cat_border={"Radiology":"rgba(77,159,255,0.3)","Lab Report":"rgba(0,212,180,0.3)","Cardiology":"rgba(255,107,157,0.3)","Clinical Notes":"rgba(245,166,35,0.3)","Unknown":"rgba(155,152,204,0.3)"}
badge_css={"Radiology":"b-rad","Lab Report":"b-lab","Cardiology":"b-card","Clinical Notes":"b-clin","Unknown":"b-unk"}
model=train_model()

def show_navbar():
    col_logo,col_links=st.columns([1,2])
    with col_logo:
        st.markdown('<div style="display:flex;align-items:center;gap:12px;padding:12px 0;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:10px;color:#9b98cc;">Diagnose Faster. Treat Better.</div></div></div>',unsafe_allow_html=True)
    with col_links:
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
    st.markdown("<hr style='border:none;border-top:1px solid #3d3a8a;margin:0 0 24px 0'>",unsafe_allow_html=True)

def show_login():
    st.markdown('<div style="text-align:center;padding:40px 0 10px;"><div style="display:inline-block;filter:drop-shadow(0 0 30px rgba(245,166,35,0.5))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:40px;font-weight:800;color:#f5a623;margin-top:10px;">MEDICLASSIFY</div><div style="font-size:15px;color:#9b98cc;font-style:italic;margin-top:6px;">Diagnose Faster. Treat Better.</div></div>',unsafe_allow_html=True)
    _,mid,_=st.columns([1,1.2,1])
    with mid:
        st.markdown('<div class="glass-card"><div class="sec-title">Login to MediClassify</div>',unsafe_allow_html=True)
    with st.form("login_form"):
       username = st.text_input("Username", placeholder="Enter your username")
       password = st.text_input("Password", placeholder="Enter your password", type="password")
       st.markdown("<br>", unsafe_allow_html=True)
        login_btn = st.form_submit_button("🔐  Login")
        st.markdown('<div style="margin-top:16px;padding:12px;background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);border-radius:10px;font-size:12px;color:#9b98cc;"><strong style="color:#f5a623">Demo accounts:</strong><br>👤 admin / admin123 &nbsp;|&nbsp; 👤 doctor / medi2024 &nbsp;|&nbsp; 👤 student / project123</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
        if login_btn:
            if username in USERS and USERS[username]==password:
                st.session_state.logged_in=True; st.session_state.username=username; st.session_state.page="home"; st.rerun()
            else: st.error("❌ Wrong username or password!")

def show_home():
    show_navbar()
    st.markdown('<div style="text-align:center;padding:30px 0 20px;"><div style="display:inline-block;filter:drop-shadow(0 0 24px rgba(245,166,35,0.45))">'+SHIELD_BIG+'</div><div style="font-family:Syne,sans-serif;font-size:46px;font-weight:800;color:#f5a623;margin-top:8px;">MEDICLASSIFY</div><div style="font-size:16px;color:#9b98cc;font-style:italic;margin-top:6px;margin-bottom:16px;">Diagnose Faster. Treat Better.</div><div style="font-size:15px;color:rgba(240,240,255,0.7);max-width:560px;margin:0 auto;line-height:1.7;">Welcome back, <strong style="color:#f5a623">'+st.session_state.username+'</strong>! 👋<br>MediClassify uses AI to automatically classify medical reports.</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    features=[("🤖","AI Classification","Naive Bayes ML algorithm"),("⚡","Instant Results","Results with confidence score"),("📋","Patient Records","Complete history storage"),("🔒","Secure Login","Username and password auth"),("📊","Live Dashboard","Real-time statistics"),("🏥","4 Categories","Radiology, Lab, Cardiology, Clinical")]
    cols=st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:16px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+title+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    _,mid,_=st.columns([1,1,1])
    with mid:
        if st.button("🏥  Go to Dashboard →"): st.session_state.page="dashboard"; st.rerun()

def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About MediClassify</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Learn about our project, technology, and team</div>',unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div class="sec-title">Project overview</div><p style="font-size:14px;color:rgba(240,240,255,0.8);line-height:1.8;"><strong style="color:#f5a623">MediClassify</strong> is a Placement Mini Project built using Python and Machine Learning. It automatically classifies medical reports into Radiology, Lab Reports, Cardiology, and Clinical Notes with AI explanations, severity levels, and confidence scores.</p></div>',unsafe_allow_html=True)
    techs=[("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes ML"),("📊","Streamlit","Web UI"),("🔤","TF-IDF","NLP vectorization"),("🖼️","Tesseract OCR","Image to text"),("📈","Plotly","Charts & graphs")]
    cols=st.columns(3)
    for i,(icon,name,desc) in enumerate(techs):
        with cols[i%3]:
            st.markdown('<div class="feature-card" style="margin-bottom:16px;"><div class="feature-icon">'+icon+'</div><div class="feature-title">'+name+'</div><div class="feature-desc">'+desc+'</div></div>',unsafe_allow_html=True)

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
            elif "@" not in c_email: st.error("Enter valid email!")
            else: st.success("✅ Thank you "+c_name+"! We will reply to "+c_email+" soon.")
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        for icon,label,value in [("📧","Email","mediclassify@gmail.com"),("📱","Phone","+91 98765 43210"),("📍","Location","Chennai, Tamil Nadu"),("🕐","Hours","Mon-Fri, 9AM-6PM")]:
            st.markdown('<div class="contact-info-card"><div class="contact-icon">'+icon+'</div><div><div class="contact-label">'+label+'</div><div class="contact-value">'+value+'</div></div></div>',unsafe_allow_html=True)

def show_dashboard():
    show_navbar()
    st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;background:#252360;border:1px solid #3d3a8a;border-radius:20px;padding:22px 32px;margin-bottom:24px;"><div style="display:flex;align-items:center;gap:16px;">'+SHIELD+'<div><div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div><div style="font-size:12px;color:#9b98cc;">Diagnose Faster. Treat Better.</div></div></div><div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);color:#f5a623;padding:8px 20px;border-radius:20px;font-size:13px;"><span class="pulse"></span>AI Active</div></div>',unsafe_allow_html=True)

    total=sum(st.session_state.counts.values())
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-num">'+str(total)+'</div><div class="stat-lbl">Total</div></div>',unsafe_allow_html=True)
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
        st.markdown('<div class="divider"></div><div class="sec-title">Medical report</div>',unsafe_allow_html=True)

        input_type=st.radio("Input type",["📝 Type Report","🖼️ Upload Image","📝 + 🖼️ Both"],horizontal=True,label_visibility="collapsed")
        p_report=""
        uploaded_img=None
        ocr_text=""

        if input_type in ["📝 Type Report","📝 + 🖼️ Both"]:
            p_report=st.text_area("Report / symptoms",placeholder="e.g. MRI scan shows fracture\nBlood test shows high glucose\nPatient has chest pain",height=100)

        if input_type in ["🖼️ Upload Image","📝 + 🖼️ Both"]:
            uploaded_img=st.file_uploader("Upload prescription image",type=["jpg","jpeg","png"],label_visibility="collapsed")
            if uploaded_img is not None:
                img=Image.open(uploaded_img)
                st.image(img,caption="Uploaded prescription",use_container_width=True)
                with st.spinner("🔍 Reading text from image..."):
                    ocr_text=extract_text_from_image(img)
                if ocr_text:
                    st.markdown('<div class="ocr-box"><div class="ocr-label">📄 Text extracted from image (OCR)</div>'+ocr_text+'</div>',unsafe_allow_html=True)
                    if input_type=="🖼️ Upload Image":
                        p_report=ocr_text
                    else:
                        p_report=p_report+" "+ocr_text
                else:
                    st.warning("⚠️ Could not extract text from image. Please type the report manually.")

        st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
        p_rx=st.text_area("Prescription / notes",placeholder="e.g. Paracetamol 500mg twice daily",height=75)
        classify_btn=st.button("▶  Classify Report")
        st.markdown('</div>',unsafe_allow_html=True)

        if classify_btn:
            if not p_report.strip():
                st.error("⚠️ Please type report text or upload a readable image!")
            else:
                category,confidence,all_proba=classify_report(model,p_report)
                severity,sev_class,sev_icon=get_severity(category,confidence)
                explanation=get_explanation(category,confidence,severity)
                name=p_name.strip() or "Unknown patient"
                meta=" · ".join(filter(None,["Age "+p_age if p_age else "",p_gender,"Dr: "+p_doc if p_doc else ""]))
                color=cat_color[category]
                bg=cat_bg[category]
                border=cat_border[category]

                # ── RESULT CARD ──
                st.markdown(
                    '<div class="result-main" style="background:'+bg+';border:1px solid '+border+';">'
                    '<div class="result-patient-info" style="color:'+color+';">'+name+((" — "+meta) if meta else "")+'</div>'
                    '<div class="result-category" style="color:'+color+';">'+category+'</div>'
                    '<div style="margin:10px 0;">'
                    '<span class="severity-badge '+sev_class+'">'+sev_icon+' '+severity+'</span>'
                    '<span style="background:rgba(255,255,255,0.08);padding:5px 14px;border-radius:20px;font-size:12px;color:'+color+';">'+str(confidence)+'% confidence</span>'
                    '</div>'
                    '<div class="ai-label">🤖 AI Explanation</div>'
                    '<div class="ai-explanation">'+explanation+'</div>'
                    +(('<div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.1);font-size:12px;color:'+color+';">Rx: '+p_rx+'</div>') if p_rx.strip() else "")
                    +'</div>',
                    unsafe_allow_html=True
                )

                # ── CONFIDENCE CHART ──
                st.markdown("<div style='margin-top:16px;'>",unsafe_allow_html=True)
                fig=go.Figure()
                cats=list(all_proba.keys())
                vals=list(all_proba.values())
                colors_bar=["#4d9fff","#00d4b4","#ff6b9d","#f5a623"]
                fig.add_trace(go.Bar(
                    x=cats,y=vals,
                    marker_color=colors_bar,
                    text=[str(v)+"%" for v in vals],
                    textposition="outside",
                    textfont=dict(color="white",size=12)
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(37,35,96,0.5)",
                    font=dict(color="white",family="DM Sans"),
                    title=dict(text="Confidence scores by category",font=dict(color="#f5a623",size=14)),
                    yaxis=dict(range=[0,110],gridcolor="rgba(255,255,255,0.05)",ticksuffix="%"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    showlegend=False,
                    height=280,
                    margin=dict(t=40,b=20,l=20,r=20)
                )
                st.plotly_chart(fig,use_container_width=True)
                st.markdown("</div>",unsafe_allow_html=True)

                st.session_state.counts[category]+=1
                now=datetime.datetime.now()
                st.session_state.history.insert(0,{
                    "name":name,"age":p_age,"gender":p_gender,"doc":p_doc,
                    "date":str(p_date),"time":now.strftime("%I:%M %p"),
                    "report":p_report[:65]+("..." if len(p_report)>65 else ""),
                    "rx":(p_rx[:50]+"...") if len(p_rx)>50 else p_rx,
                    "cat":category,"conf":confidence,"severity":severity,"sev_icon":sev_icon
                })
                st.rerun()

    with right:
        with st.expander("📊 Project flow",expanded=False):
            for i,(icon,label) in enumerate([("📄","Medical report (text/image)"),("🖼️","OCR extracts image text"),("🔤","Clean text — NLP"),("🤖","Naive Bayes ML model"),("🎯","Predict + severity + chart"),("✅","Output with AI explanation")]):
                st.markdown('<div class="flow-step"><span style="font-size:18px">'+icon+'</span> '+label+'</div>',unsafe_allow_html=True)
                if i<5: st.markdown('<div style="text-align:center;color:#f5a623;font-size:20px;margin:-3px 0">↓</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:16px">Classification history</div>',unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:36px 20px;color:#9b98cc;background:#252360;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:38px;margin-bottom:10px">📋</div><div>No reports classified yet.</div></div>',unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:12]:
                meta=" · ".join(filter(None,["Age "+h['age'] if h['age'] else "",h['gender'],"Dr: "+h['doc'] if h['doc'] else ""]))
                color=cat_color[h['cat']]; bc=badge_css[h['cat']]
                rx_line='<div class="hist-rx">Rx: '+h['rx']+'</div>' if h['rx'] else ""
                meta_line='<div class="hist-meta">'+meta+'</div>' if meta else ""
                sev_line='<span style="font-size:10px;margin-left:4px;">'+h.get('sev_icon','')+" "+h.get('severity','')+'</span>'
                st.markdown(
                    '<div class="hist-card">'
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px">'
                    '<span class="hist-name">'+h['name']+'</span>'
                    '<span class="badge '+bc+'">'+h['cat']+'</span></div>'
                    +meta_line+
                    '<div class="hist-rep">'+h['report']+'</div>'
                    +rx_line+
                    '<div class="hist-time">'+h['date']+' at '+h['time']+' | '+str(h['conf'])+'% '+sev_line+'</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            if st.button("🗑️ Clear history"):
                st.session_state.history=[]; st.session_state.counts={"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}; st.rerun()

        # ── PIE CHART ──
        if sum(st.session_state.counts.values())>0:
            st.markdown('<div class="sec-title" style="margin-top:16px">Distribution chart</div>',unsafe_allow_html=True)
            labels=[k for k,v in st.session_state.counts.items() if v>0]
            values=[v for v in st.session_state.counts.values() if v>0]
            fig2=go.Figure(go.Pie(
                labels=labels,values=values,
                marker=dict(colors=["#4d9fff","#00d4b4","#ff6b9d","#f5a623"]),
                textfont=dict(color="white"),hole=0.4
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=True,
                legend=dict(font=dict(color="white")),
                height=260,
                margin=dict(t=10,b=10,l=10,r=10)
            )
            st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY</strong> &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project</div>',unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page=="home": show_home()
    elif st.session_state.page=="about": show_about()
    elif st.session_state.page=="contact": show_contact()
    elif st.session_state.page=="dashboard": show_dashboard()
    else: show_home()

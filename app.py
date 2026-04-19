import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except ImportError:
    install("scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

import streamlit as st
import re
import datetime

st.set_page_config(page_title="MediClassify", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

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
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def classify_report(model, text):
    cleaned = clean_text(text)
    category = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    return category, round(max(proba) * 100)

cat_color = {"Radiology":"#4d9fff","Lab Report":"#00d4b4","Cardiology":"#ff6b9d","Clinical Notes":"#f5a623","Unknown":"#9b98cc"}
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
        st.markdown(
            '<div style="margin-top:16px;padding:12px;background:rgba(245,166,35,0.08);'
            'border:1px solid rgba(245,166,35,0.2);border-radius:10px;font-size:12px;color:#9b98cc;">'
            '<strong style="color:#f5a623">Demo accounts:</strong><br>'
            '👤 admin / admin123 &nbsp;|&nbsp; 👤 doctor / medi2024 &nbsp;|&nbsp; 👤 student / project123</div>',
            unsafe_allow_html=True
        )
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
    st.markdown(
        '<div style="text-align:center;padding:30px 0 20px;">'
        '<div style="display:inline-block;filter:drop-shadow(0 0 24px rgba(245,166,35,0.45))">' + SHIELD_BIG + '</div>'
        '<div style="font-family:Syne,sans-serif;font-size:46px;font-weight:800;color:#f5a623;margin-top:8px;">MEDICLASSIFY</div>'
        '<div style="font-size:16px;color:#9b98cc;font-style:italic;margin-top:6px;margin-bottom:16px;">Diagnose Faster. Treat Better.</div>'
        '<div style="font-size:15px;color:rgba(240,240,255,0.7);max-width:560px;margin:0 auto;line-height:1.7;">'
        'Welcome back, <strong style="color:#f5a623">' + st.session_state.username + '</strong>! 👋<br>'
        'MediClassify uses AI to automatically classify medical reports.</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    features = [
        ("🤖","AI Classification","Naive Bayes ML algorithm"),
        ("⚡","Instant Results","Results with confidence score"),
        ("📋","Patient Records","Complete history storage"),
        ("🔒","Secure Login","Username and password auth"),
        ("📊","Live Dashboard","Real-time statistics"),
        ("🏥","4 Categories","Radiology, Lab, Cardiology, Clinical"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                '<div class="feature-card" style="margin-bottom:16px;">'
                '<div class="feature-icon">' + icon + '</div>'
                '<div class="feature-title">' + title + '</div>'
                '<div class="feature-desc">' + desc + '</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("🏥  Go to Dashboard →"):
            st.session_state.page = "dashboard"
            st.rerun()

def show_about():
    show_navbar()
    st.markdown('<div class="page-title">About MediClassify</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Learn about our project, technology, and team</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-card"><div class="sec-title">Project overview</div>'
        '<p style="font-size:14px;color:rgba(240,240,255,0.8);line-height:1.8;">'
        '<strong style="color:#f5a623">MediClassify</strong> is a Placement Mini Project built using '
        'Python and Machine Learning. It automatically classifies medical reports into '
        'Radiology, Lab Reports, Cardiology, and Clinical Notes.</p></div>',
        unsafe_allow_html=True
    )
    techs = [("🐍","Python 3","Main language"),("🤖","Scikit-Learn","Naive Bayes ML"),("📊","Streamlit","Web UI"),("🔤","TF-IDF","NLP vectorization")]
    cols = st.columns(4)
    for col, (icon, name, desc) in zip(cols, techs):
        with col:
            st.markdown(
                '<div class="feature-card" style="margin-bottom:16px;">'
                '<div class="feature-icon">' + icon + '</div>'
                '<div class="feature-title">' + name + '</div>'
                '<div class="feature-desc">' + desc + '</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('<div class="glass-card"><div class="sec-title">How it works</div>', unsafe_allow_html=True)
    steps = [("📄","Medical report entered"),("🔤","Text cleaned with NLP"),("📐","TF-IDF vectorization"),("🤖","Naive Bayes predicts"),("✅","Output with confidence")]
    for i, (icon, label) in enumerate(steps):
        st.markdown('<div class="flow-step"><span style="font-size:20px">' + icon + '</span> <strong style="color:#f5a623">Step ' + str(i+1) + ':</strong> ' + label + '</div>', unsafe_allow_html=True)
        if i < 4:
            st.markdown('<div style="text-align:center;color:#f5a623;font-size:22px;margin:-2px 0">↓</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_contact():
    show_navbar()
    st.markdown('<div class="page-title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Have a question? Send us a message!</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="glass-card"><div class="sec-title">Send a message</div>', unsafe_allow_html=True)
        c_name    = st.text_input("Your name", placeholder="e.g. Rahul Kumar")
        c_email   = st.text_input("Email", placeholder="e.g. rahul@email.com")
        c_subject = st.selectbox("Subject", ["", "General Inquiry", "Technical Support", "Feedback", "Bug Report"])
        c_msg     = st.text_area("Message", placeholder="Write your message here...", height=120)
        if st.button("📨  Send Message"):
            if not c_name or not c_email or not c_msg or not c_subject:
                st.error("Please fill all fields!")
            elif "@" not in c_email:
                st.error("Enter valid email!")
            else:
                st.success("✅ Thank you " + c_name + "! We will reply to " + c_email + " soon.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        contacts = [("📧","Email","mediclassify@gmail.com"),("📱","Phone","+91 98765 43210"),("📍","Location","Chennai, Tamil Nadu"),("🕐","Hours","Mon-Fri, 9AM-6PM")]
        for icon, label, value in contacts:
            st.markdown(
                '<div class="contact-info-card"><div class="contact-icon">' + icon + '</div>'
                '<div><div class="contact-label">' + label + '</div>'
                '<div class="contact-value">' + value + '</div></div></div>',
                unsafe_allow_html=True
            )

def show_dashboard():
    show_navbar()
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'background:#252360;border:1px solid #3d3a8a;border-radius:20px;padding:22px 32px;margin-bottom:24px;">'
        '<div style="display:flex;align-items:center;gap:16px;">' + SHIELD +
        '<div><div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#f5a623;">MEDICLASSIFY</div>'
        '<div style="font-size:12px;color:#9b98cc;">Diagnose Faster. Treat Better.</div></div></div>'
        '<div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);'
        'color:#f5a623;padding:8px 20px;border-radius:20px;font-size:13px;">'
        '<span class="pulse"></span>AI Active</div></div>',
        unsafe_allow_html=True
    )
    total = sum(st.session_state.counts.values())
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="stat-box"><div class="stat-num">' + str(total) + '</div><div class="stat-lbl">Total</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#4d9fff">' + str(st.session_state.counts["Radiology"]) + '</div><div class="stat-lbl">Radiology</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#00d4b4">' + str(st.session_state.counts["Lab Report"]) + '</div><div class="stat-lbl">Lab Report</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="stat-box"><div class="stat-num" style="color:#ff6b9d">' + str(st.session_state.counts["Cardiology"]) + '</div><div class="stat-lbl">Cardiology</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="glass-card"><div class="sec-title">Patient information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: p_name = st.text_input("Patient name", placeholder="e.g. Rahul Kumar")
        with col2: p_age  = st.text_input("Age", placeholder="e.g. 35")
        col3, col4 = st.columns(2)
        with col3: p_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        with col4: p_doc    = st.text_input("Doctor name", placeholder="e.g. Dr. Priya")
        p_date = st.date_input("Date of visit")
        st.markdown('<div class="divider"></div><div class="sec-title">Medical report</div>', unsafe_allow_html=True)
        input_type = st.radio("Input type", ["📝 Type Report", "🖼️ Upload Image", "📝 + 🖼️ Both"], horizontal=True, label_visibility="collapsed")
        p_report = ""
        uploaded_img = None
        if input_type in ["📝 Type Report", "📝 + 🖼️ Both"]:
            p_report = st.text_area("Report / symptoms", placeholder="e.g. MRI scan shows fracture\nBlood test shows high glucose\nPatient has chest pain", height=100)
        if input_type in ["🖼️ Upload Image", "📝 + 🖼️ Both"]:
            uploaded_img = st.file_uploader("Upload prescription image", type=["jpg","jpeg","png","pdf"], label_visibility="collapsed")
            if uploaded_img is not None and uploaded_img.type in ["image/jpeg","image/png","image/jpg"]:
                st.image(uploaded_img, caption="Uploaded prescription", use_container_width=True)
                st.markdown('<div style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);border-radius:10px;padding:10px;font-size:12px;color:#f5a623;margin-top:8px;">✅ Image uploaded! Type report text for AI classification.</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        p_rx = st.text_area("Prescription / notes", placeholder="e.g. Paracetamol 500mg twice daily", height=75)
        classify_btn = st.button("▶  Classify Report")
        st.markdown('</div>', unsafe_allow_html=True)

        if classify_btn:
            if not p_report.strip():
                st.error("Please type the report text to classify!")
            else:
                category, confidence = classify_report(model, p_report)
                name  = p_name.strip() or "Unknown patient"
                meta  = " · ".join(filter(None, [
                    "Age " + p_age if p_age else "",
                    p_gender,
                    "Dr: " + p_doc if p_doc else ""
                ]))
                color = cat_color[category]
                css   = cat_css[category]
                has_img = uploaded_img is not None
                img_badge = '<span style="background:rgba(245,166,35,0.15);color:#f5a623;border:1px solid rgba(245,166,35,0.3);padding:2px 10px;border-radius:10px;font-size:10px;margin-left:8px;">+ Image</span>' if has_img else ""
                rx_html = '<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.12);font-size:12px;color:' + color + '">Rx: ' + p_rx + '</div>' if p_rx.strip() else ""
                meta_html = " — " + meta if meta else ""
                st.markdown(
                    '<div class="' + css + '">'
                    '<div style="font-size:12px;color:' + color + ';margin-bottom:4px;">Classification for ' + name + meta_html + ' ' + img_badge + '</div>'
                    '<div class="result-cat" style="color:' + color + '">' + category + '</div>'
                    '<div class="result-info" style="color:' + color + '">' + str(confidence) + '% confidence | Naive Bayes ML</div>'
                    + rx_html + '</div>',
                    unsafe_allow_html=True
                )
                st.session_state.counts[category] += 1
                now = datetime.datetime.now()
                st.session_state.history.insert(0, {
                    "name": name, "age": p_age, "gender": p_gender, "doc": p_doc,
                    "date": str(p_date), "time": now.strftime("%I:%M %p"),
                    "report": p_report[:65] + ("..." if len(p_report) > 65 else ""),
                    "rx": (p_rx[:50] + "...") if len(p_rx) > 50 else p_rx,
                    "cat": category, "conf": confidence, "has_img": has_img
                })
                st.rerun()

    with right:
        with st.expander("📊 Project flow", expanded=False):
            steps = [("📄","Medical report (text/image)"),("🔤","Clean text — NLP"),("🤖","Naive Bayes ML model"),("🎯","Predict category"),("✅","Output result")]
            for i, (icon, label) in enumerate(steps):
                st.markdown('<div class="flow-step"><span style="font-size:18px">' + icon + '</span> ' + label + '</div>', unsafe_allow_html=True)
                if i < 4:
                    st.markdown('<div style="text-align:center;color:#f5a623;font-size:20px;margin:-3px 0">↓</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:16px">Classification history</div>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:36px 20px;color:#9b98cc;background:#252360;border:1px solid #3d3a8a;border-radius:16px;"><div style="font-size:38px;margin-bottom:10px">📋</div><div>No reports classified yet.</div></div>', unsafe_allow_html=True)
        else:
            for h in st.session_state.history[:12]:
                meta = " · ".join(filter(None, [
                    "Age " + h['age'] if h['age'] else "",
                    h['gender'],
                    "Dr: " + h['doc'] if h['doc'] else ""
                ]))
                color    = cat_color[h['cat']]
                bc       = badge_css[h['cat']]
                img_tag  = " 🖼️" if h.get('has_img') else ""
                rx_line  = '<div class="hist-rx">Rx: ' + h['rx'] + '</div>' if h['rx'] else ""
                meta_line = '<div class="hist-meta">' + meta + '</div>' if meta else ""
                st.markdown(
                    '<div class="hist-card">'
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px">'
                    '<span class="hist-name">' + h['name'] + img_tag + '</span>'
                    '<span class="badge ' + bc + '">' + h['cat'] + '</span></div>'
                    + meta_line +
                    '<div class="hist-rep">' + h['report'] + '</div>'
                    + rx_line +
                    '<div class="hist-time">' + h['date'] + ' at ' + h['time'] + ' | ' + str(h['conf']) + '% confidence</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.session_state.counts  = {"Radiology":0,"Lab Report":0,"Cardiology":0,"Clinical Notes":0}
                st.rerun()

    st.markdown('<div class="footer-bar"><strong style="color:#f5a623">MEDICLASSIFY</strong> &nbsp;|&nbsp; Diagnose Faster. Treat Better. &nbsp;|&nbsp; Placement Mini Project</div>', unsafe_allow_html=True)

# ROUTER
if not st.session_state.logged_in:
    show_login()
else:
    if st.session_state.page == "home":       show_home()
    elif st.session_state.page == "about":    show_about()
    elif st.session_state.page == "contact":  show_contact()
    elif st.session_state.page == "dashboard":show_dashboard()
    else: show_home()

import streamlit as st
import requests
import json

# Page Layout
st.set_page_config(page_title="MedPal AI V5", layout="wide", page_icon="🏥")
API_URL = "http://127.0.0.1:8000"

# Custom CSS for compact spacing
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .stMetric {background-color: #0E1117; padding: 10px; border-radius: 5px; border: 1px solid #262730;}
    .streamlit-expanderHeader {font-weight: bold; color: #00FFAA;}
</style>
""", unsafe_allow_html=True)

st.title("🏥 MedPal AI: Clinical Dashboard")

# 3-Column Layout
col_left, col_mid, col_right = st.columns([1, 2, 1.2], gap="medium")

# ==========================================
# ⬅️ LEFT COLUMN: INPUT & CONTROL
# ==========================================
with col_left:
    st.subheader("📂 Patient Record")
    uploaded_file = st.file_uploader("Upload scanned PDF or digital report.", type=["pdf"])
    
    if uploaded_file:
        analyze_btn = st.button("🚀 Analyze Document", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Digitizing, Extracting & Searching Medical Knowledge..."):
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(f"{API_URL}/analyze_full", files=files)
                    if response.status_code == 200:
                        st.session_state['analysis_result'] = response.json()
                        st.success("Processing Complete")
                    else:
                        st.error("Server Error")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
    
    st.divider()
    st.caption("System Status")
    st.success("✅ Neural Engine: Online")
    st.success("✅ Llama 3.3: Connected")
    if st.session_state.get('analysis_result', {}).get('similar_cases'):
        st.success("✅ PMC Library: Active")

# ==========================================
# 📄 MIDDLE COLUMN: GENAI INSIGHTS
# ==========================================
with col_mid:
    if 'analysis_result' in st.session_state:
        res = st.session_state['analysis_result']
        
        # Tabs for Content
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "✉️ Patient Letter", "💰 Billing", "💬 AI Assistant"])
        
        with tab1:
            st.markdown(res.get("generated_summary", "No summary available."))
            
        with tab2:
            st.info("✉️ Draft Referral Letter")
            letter = res.get("patient_letter", "Letter generation failed.") # Updated key
            st.text_area("Copy/Edit Content:", value=letter, height=300)
            
            st.download_button(
                label="📥 Download Letter",
                data=letter,
                file_name=f"Referral_{res.get('extracted_vitals', {}).get('age', 'Patient')}.txt",
                mime="text/plain"
            )

        with tab3:
            st.success("💰 Suggested ICD-10 Codes")
            billing = res.get("billing_codes", {})
            # Handle list vs dict format
            codes = billing.get("codes", []) if isinstance(billing, dict) else billing
            
            if codes:
                for item in codes:
                    c = item.get('code', 'N/A')
                    d = item.get('description', 'Unknown')
                    st.markdown(f"**{c}** — *{d}*")
            else:
                st.warning("No specific billing codes detected.")
            
        with tab4:
            st.caption("Chat with this specific patient report.")
    
            # This container with a fixed height enables auto-scrolling to the bottom
            chat_container = st.container(height=500, border=True)
    
            with chat_container:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Render all existing messages
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Chat input should be OUTSIDE the container to stay pinned at the bottom
            if prompt := st.chat_input("Ask MedPal..."):
                # 1. Add user message to state
                st.session_state.messages.append({"role": "user", "content": prompt})
        
                # 2. Display user message immediately
                with chat_container.chat_message("user"):
                    st.markdown(prompt)

                # 3. Generate and display assistant response
                with chat_container.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = requests.post(f"{API_URL}/chat", json={"question": prompt}).json()
                            answer = str(resp) if not isinstance(resp, dict) else resp.get("ai_insight", str(resp))
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except:
                            st.error("Chat Error")
        
                # Rerun to ensure the container focuses on the bottom
                st.rerun()

        # EVIDENCE-BASED MEDICINE EXPANDER
        with st.expander("📚 Evidence-Based Medicine (PMC-Patients Match)", expanded=True):
            cases = res.get('similar_cases', [])
            if cases:
                st.info(f"MedPal found {len(cases)} similar cases in the Medical Library:")
                for i, case in enumerate(cases):
                    st.markdown(f"**{i+1}. {case['title']}**")
                    st.caption(case['summary'])
                    if i < len(cases) - 1: st.markdown("---")
            else:
                st.write("No similar cases found in local library.")

    else:
        st.markdown("### ⬅️ Action Required")
        st.info("Please upload a Patient PDF in the left panel to generate the dashboard.")

# ==========================================
# 🧠 RIGHT COLUMN: NEURO-SYMBOLIC SIMULATOR
# ==========================================
with col_right:
    st.subheader("🧠 Risk Engine")
    
    # Toggle Mode
    mode = st.radio("Mode", ["Static Analysis", "Interactive Simulator"], horizontal=True)
    
    if mode == "Static Analysis":
        if 'analysis_result' in st.session_state:
            res = st.session_state['analysis_result']
            risks = res.get("risk_assessment", {})
            vitals = res.get("extracted_vitals", {})
            
            # Vitals Card
            with st.container(border=True):
                st.caption("🧬 Extracted Vitals")
                c1, c2 = st.columns(2)
                sys = vitals.get('bp_systolic', -1)
                dia = vitals.get('bp_diastolic', -1)
                bp_disp = f"{sys}/{dia} mmHg" if sys > 0 else "N/A"
                
                gluc = vitals.get('glucose', -1)
                gluc_disp = f"{gluc} mg/dL" if gluc > 0 else "N/A"
                
                c1.metric("BP", bp_disp)
                c2.metric("Glucose", gluc_disp)
                
                c3, c4 = st.columns(2)
                c3.metric("Smoker", "Yes" if vitals.get('smoker')==1 else "No")
                c4.metric("Fam Hist", "Yes" if vitals.get('family_history')==1 else "No")

            st.markdown("---")
            
            # Risk Gauges
            d_score = float(risks.get('diabetes_probability', '0%').strip('%'))
            h_score = float(risks.get('heart_disease_probability', '0%').strip('%'))
            s_score = float(risks.get('stroke_probability', '0%').strip('%')) # New key
            
            st.write("**Diabetes Risk**")
            st.progress(min(d_score/100, 1.0))
            st.caption(f"Probability: {d_score}%")
            
            st.write("**Heart Disease Risk**")
            st.progress(min(h_score/100, 1.0))
            st.caption(f"Probability: {h_score}%")

            st.write("**Stroke Risk**")
            st.progress(min(s_score/100, 1.0))
            st.caption(f"Probability: {s_score}%")
            
            if h_score > 50:
                st.error("⚠️ High Risk Detected")
            else:
                st.success("✅ Low Clinical Risk")
        else:
            st.caption("Waiting for analysis...")

    else: # SIMULATOR MODE
        st.info("Adjust values to see real-time risk changes.")
        
        s_age = st.slider("Age", 20, 90, 55)
        s_bmi = st.slider("BMI", 15.0, 50.0, 32.5)
        s_gluc = st.slider("Glucose", 50, 300, 160)
        s_chol = st.slider("Cholesterol", 100, 400, 240)
        s_sys = st.slider("Systolic BP", 90, 200, 145)
        s_dia = st.slider("Diastolic BP", 60, 120, 90)
        
        c1, c2 = st.columns(2)
        s_smoker = c1.checkbox("Smoker?")
        s_fam = c2.checkbox("Family Hist?")
        
        if st.button("🔄 Recalculate Risk", type="primary"):
            payload = {
                "age": s_age, "glucose": s_gluc, "bp_systolic": s_sys,
                "bp_diastolic": s_dia, "bmi": s_bmi, "cholesterol": s_chol,
                "smoker": 1 if s_smoker else 0, "family_history": 1 if s_fam else 0
            }
            try:
                sim_res = requests.post(f"{API_URL}/predict_manual", json=payload).json()
                st.divider()
                st.metric("Diabetes Risk", f"{sim_res['diabetes_risk']}%")
                st.metric("Heart Risk", f"{sim_res['heart_risk']}%")
            except Exception as e: st.error(f"Sim Error: {e}")
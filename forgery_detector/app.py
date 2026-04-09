# app.py

import streamlit as st
import os
import numpy as np
from PIL import Image

from modules.ela import perform_ela, get_ela_regions
from modules.metadata import analyze_metadata
from modules.cnn_model import predict_image
from modules.pdf_handler import pdf_to_images

# ── Page Config ──
st.set_page_config(
    page_title="Document Forgery Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Styling ──
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .risk-high   { color: red;    font-weight: bold; font-size: 1.5rem; }
    .risk-medium { color: orange; font-weight: bold; font-size: 1.5rem; }
    .risk-low    { color: green;  font-weight: bold; font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="main-title">🔍 AI Document Forgery Detector</div>', 
            unsafe_allow_html=True)
st.write("Upload any document image or PDF to analyze it for signs of forgery.")
st.divider()

# ── File Upload ──
uploaded_file = st.file_uploader(
    "📎 Upload Document (JPG, PNG, PDF)",
    type=['jpg', 'jpeg', 'png', 'pdf']
)

os.makedirs('uploads', exist_ok=True)

if uploaded_file is not None:
    
    # Save uploaded file
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Handle PDF: convert first page to image
    if uploaded_file.name.endswith('.pdf'):
        st.info("📄 PDF detected — analyzing first page...")
        pages = pdf_to_images(file_path)
        analyze_path = pages[0]  # Analyze first page
    else:
        analyze_path = file_path
    
    # ── Show Original Image ──
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Original Document")
        st.image(analyze_path, use_column_width=True)
    
    # ── Analyze Button ──
    if st.button("🔬 Analyze for Forgery", type="primary"):
        
        with st.spinner("Analyzing document..."):
            
            # ── Run All 3 Analyses ──
            ela_image, ela_score = perform_ela(analyze_path)
            meta_result = analyze_metadata(analyze_path)
            cnn_label, cnn_confidence = predict_image(analyze_path)
        
        # ── Show ELA Result ──
        with col2:
            st.subheader("🔬 ELA Analysis")
            st.image(ela_image, use_column_width=True)
            st.caption("Bright areas = Potentially tampered regions")
        
        st.divider()
        
        # ── Results Dashboard ──
        st.subheader("📊 Analysis Results")
        
        r1, r2, r3 = st.columns(3)
        
        with r1:
            st.metric(
                label="ELA Suspicion Score",
                value=f"{ela_score}%",
                delta="Higher = More Suspicious"
            )
        
        with r2:
            st.metric(
                label="Metadata Risk",
                value=meta_result['risk_level'],
                delta=f"Score: {meta_result['risk_score']}/100"
            )
        
        with r3:
            st.metric(
                label="CNN Prediction",
                value=cnn_label.upper(),
                delta=f"Confidence: {cnn_confidence}%"
            )
        
        st.divider()
        
        # ── Final Verdict ──
        st.subheader("⚖️ Final Verdict")
        
        # Calculate combined score
        # ELA score + metadata risk + CNN result
        cnn_score = cnn_confidence if cnn_label == 'forged' else (100 - cnn_confidence)
        final_score = (ela_score * 0.4) + (meta_result['risk_score'] * 0.3) + (cnn_score * 0.3)
        
        if final_score > 60:
            st.markdown('<p class="risk-high">🚨 HIGH RISK — Document likely FORGED</p>',
                        unsafe_allow_html=True)
        elif final_score > 35:
            st.markdown('<p class="risk-medium">⚠️ MEDIUM RISK — Suspicious elements found</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p class="risk-low">✅ LOW RISK — Document appears Authentic</p>',
                        unsafe_allow_html=True)
        
        st.progress(int(final_score), text=f"Overall Forgery Score: {final_score:.1f}%")
        
        # ── Metadata Findings ──
        if meta_result['findings']:
            st.subheader("🔍 Detailed Findings")
            for finding in meta_result['findings']:
                st.warning(finding)
        
        # ── Raw Metadata Expander ──
        with st.expander("📋 View Raw Metadata"):
            st.json(meta_result['raw_metadata'])
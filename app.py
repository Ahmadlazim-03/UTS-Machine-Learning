import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add subdirectories to path
sys.path.append('MissingValue')
sys.path.append('Transformasi')
sys.path.append('EkstraksiFitur')
sys.path.append('ImbalancedData')

from missing_value import process_missing_values
from transformasi import process_data_transformation
from ektraksi_fitur import process_feature_extraction
from imbalanced_data import process_imbalanced_data

st.set_page_config(page_title="Cervical Cancer Preprocessing", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f7f9fa;
    }
    .stApp {
        background-color: #f7f9fa;
    }
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #34495e;
        margin-bottom: 1em;
    }
    .card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(44,62,80,0.08);
        padding: 1.5em;
        margin-bottom: 1.5em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Cervical Cancer Data Preprocessing Toolkit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Visualisasi & Analisis Data Faktor Risiko Kanker Serviks</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Pengaturan Preprocessing")
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    mv_strategy = st.selectbox("Strategi Missing Value", ["median", "mean", "mode", "drop"], index=0)
    scaler_type = st.selectbox("Scaler", ["minmax", "standard", "robust"], index=0)
    variance_threshold = st.slider("PCA Variance Threshold", 0.80, 0.99, 0.95, 0.01)
    resampling_method = st.selectbox("Resampling Method", ["ros", "smote", "adasyn"], index=0)
    run_pipeline = st.button("Jalankan Preprocessing")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Data berhasil diupload! Shape: {df.shape}")
    st.write(df.head())
else:
    st.info("Silakan upload file CSV untuk mulai analisis.")
    df = None

if run_pipeline and df is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1️⃣ Missing Value Analysis")
    cleaned_data, mv_summary = process_missing_values(df, output_type='web', strategy=mv_strategy)
    st.write("Summary:", mv_summary)
    st.write("Data setelah penanganan missing value:")
    st.dataframe(cleaned_data.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2️⃣ Data Transformation")
    transformed_data, transformer, tr_summary = process_data_transformation(cleaned_data, output_type='web', scaler_type=scaler_type)
    st.write("Summary:", tr_summary)
    st.write("Data setelah transformasi:")
    st.dataframe(transformed_data.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3️⃣ Feature Extraction (PCA)")
    pca_data, extractor, fe_summary = process_feature_extraction(transformed_data, output_type='web', variance_threshold=variance_threshold)
    st.write("Summary:", fe_summary)
    st.write("Data setelah PCA:")
    st.dataframe(pca_data.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4️⃣ Imbalanced Data Handling")
    resampled_data, handler, id_summary = process_imbalanced_data(pca_data, output_type='web', method=resampling_method)
    st.write("Summary:", id_summary)
    st.write("Data setelah resampling:")
    st.dataframe(resampled_data.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    st.success("Preprocessing selesai! Data siap untuk modeling.")
    st.download_button("Download Data Final (CSV)", resampled_data.to_csv(index=False), file_name="final_processed_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Tips:** Anda dapat mengubah parameter di sidebar dan jalankan ulang untuk eksperimen.")

else:
    st.markdown("<div class='card'><b>Petunjuk:</b> Upload data, pilih parameter di sidebar, lalu klik <b>Jalankan Preprocessing</b>.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center>Developed by: <b>UTS Machine Learning Project</b> | Powered by Streamlit</center>", unsafe_allow_html=True)

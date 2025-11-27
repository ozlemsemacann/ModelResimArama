import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Model Resmine En Yakın Ürün", layout="wide")

# CACHE
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-16", device="cpu")

@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")

@st.cache_data
def load_dataframe():
    return pd.read_excel("images.xlsx")

model = load_model()
embeddings = load_embeddings()
df = load_dataframe()

st.title("🔍 Model Resmine En Yakın Ürünü Bulma")

uploaded_file = st.file_uploader("Bir model resmi yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Yüklenen Resim", width=350)

    query_emb = model.encode(query_img)

    sims = cosine_similarity([query_emb], embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]

    best_path = f"images/img_{best_idx}.jpg"

    if os.path.exists(best_path):
        best_img = Image.open(best_path)
        st.subheader("📌 En Benzer Bulunan Ürün")
        st.image(best_img, caption=f"Benzerlik Skoru: {best_score:.3f}", width=350)
    else:
        st.error("Benzer resim bulunamadı.")

    st.write("📄 Excel Kaydı:")
    st.write(df.iloc[best_idx])

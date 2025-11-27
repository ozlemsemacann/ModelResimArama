import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- AYARLAR ---
MODEL_NAME = "clip-ViT-B-16"
EMBEDDING_FILE = "embeddings.npy"
EXCEL_FILE = "images.xlsx"

st.set_page_config(layout="wide", page_title="AI Görsel Arama")

@st.cache_resource
def load_resources():
    # 1. Modeli yükle (Sorgu resmini vektöre çevirmek için lazım)
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Kaydedilmiş vektörleri yükle
    try:
        stored_embeddings = np.load(EMBEDDING_FILE)
    except FileNotFoundError:
        st.error(f"'{EMBEDDING_FILE}' bulunamadı! Lütfen önce hazırlık kodunu çalıştırın.")
        return None, None, None

    # 3. Excel dosyasını yükle (Linkleri göstermek için)
    try:
        df = pd.read_excel(EXCEL_FILE)
    except FileNotFoundError:
        st.error(f"'{EXCEL_FILE}' bulunamadı!")
        return None, None, None

    return model, stored_embeddings, df

# --- URL Sütununu Bulma (Hazırlık koduyla aynı mantık) ---
def find_url_column(df):
    possible_cols = ['link', 'url', 'image_url', 'resim_link', 'gorsel_link', 'image', 'img_url']
    for col in df.columns:
        if str(col).lower() in possible_cols:
            return col
    # Bulamazsa 'http' içeren ilk sütunu al
    for col in df.columns:
        if df[col].astype(str).str.contains('http').any():
            return col
    return df.columns[0] # Hiçbiri yoksa ilk sütun

# --- UYGULAMA BAŞLANGICI ---
st.title("🔎 Akıllı Görsel Arama")
st.caption("CLIP Modeli ile önceden indekslenmiş verilerde arama yapın.")

model, stored_embeddings, df = load_resources()

if model is not None and stored_embeddings is not None:
    
    url_col = find_url_column(df)
    
    # KULLANICI RESİM YÜKLER
    uploaded_file = st.file_uploader("Bir resim yükleyin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Aranan Resim", width=250)
            
            # Aranan resmin vektörünü çıkar
            with st.spinner("Analiz ediliyor..."):
                query_embedding = model.encode(query_image, convert_to_numpy=True)
            
            # Benzerlik Hesapla (Cosine Similarity)
            # query_embedding -> (512,) bunu (1, 512) yapıyoruz
            similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings)[0]
            
            # En iyi 5 sonucu bul
            top_k = 10
            # argsort küçükten büyüğe sıralar, ters çeviriyoruz [::-1]
            top_indices = similarities.argsort()[-top_k:][::-1]
            
        with col2:
            st.subheader("📸 En Benzer Sonuçlar")
            
            # Sonuçları ızgara (grid) şeklinde göster
            cols = st.columns(3) # 3 sütunlu görünüm
            
            for i, idx in enumerate(top_indices):
                score = similarities[idx]
                row_data = df.iloc[idx]
                image_url = row_data[url_col]
                
                # Siyah kare kontrolü (Hatalı resimlerin skoru genelde düşük olur ama yine de filtreleyelim)
                # Buradaki mantık: Eğer hazırlık aşamasında siyah resim atandıysa, kullanıcı renkli resim arattığında zaten çıkmaz.
                
                with cols[i % 3]:
                    st.image(image_url, use_container_width=True)
                    st.caption(f"Benzerlik: **%{score*100:.1f}**")
                    st.markdown(f"[Ürüne Git]({image_url})")
                    # İsterseniz ürün adını da yazdırabilirsiniz:
                    # st.text(row_data['UrunAdi'])
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from io import BytesIO

st.set_page_config(page_title="Akıllı Görsel Arama", layout="wide")

# -----------------------------------------------------------------------------
# 1. YÜKLEME FONKSİYONLARI
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-16", device="cpu")

@st.cache_data
def load_data():
    if not os.path.exists("embeddings.npy") or not os.path.exists("images.xlsx"):
        return None, None, "Veri dosyaları eksik! Lütfen önce 'create_embeddings.py' dosyasını çalıştırın."
    
    try:
        embs = np.load("embeddings.npy")
        df = pd.read_excel("images.xlsx")
        # Veri temizliği
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return embs, df, None
    except Exception as e:
        return None, None, f"Dosya okuma hatası: {str(e)}"

def get_url_from_row(row):
    """Excel satırından URL'yi bulur."""
    possible_cols = ['link', 'url', 'image_url', 'resim_link', 'gorsel_link', 'image', 'resim']
    for col in row.index:
        if str(col).lower() in possible_cols or (isinstance(row[col], str) and str(row[col]).startswith('http')):
            return row[col]
    return None

def get_filterable_columns(df):
    """Filtre olmaya uygun sütunları bulur."""
    filter_cols = []
    priority_keywords = ['kategori', 'category', 'grup', 'group', 'cinsiyet', 'gender', 
                         'hedef', 'tip', 'type', 'stil', 'style', 'kalıp', 'fit']
    
    for col in df.columns:
        if "url" in str(col).lower() or "link" in str(col).lower(): continue
            
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            unique_count = df[col].nunique()
            is_priority = any(k in str(col).lower() for k in priority_keywords)
            if unique_count < 50 and (unique_count > 1 or is_priority):
                filter_cols.append(col)
    return filter_cols

# -----------------------------------------------------------------------------
# 2. ARAYÜZ VE MANTIK
# -----------------------------------------------------------------------------
st.title("🔍 Akıllı Ürün Eşleştirme")
st.markdown("Ürün tipi ve kalıbına göre benzer ürünleri bulun.")

# Önbellek Temizleme Butonu
if st.sidebar.button("🔄 Verileri ve Önbelleği Yenile"):
    st.cache_data.clear()
    st.rerun()

model = load_model()
embeddings, df, error = load_data()

if error:
    st.error(f"⚠️ {error}")
    st.stop()

# --- KENAR ÇUBUĞU ---
st.sidebar.header("⚙️ Arama Ayarları")

# 1. "RENK ÖNEMLİ DEĞİL" MODU
st.sidebar.subheader("🎯 Arama Modu")
ignore_color = st.sidebar.toggle("Renkleri Yoksay (Kalıp Odaklı)", value=True, 
                               help="Açık olduğunda: Resmin renklerini siler ve kontrastı artırarak sadece ürünün kalıbına/şekline odaklanır.")

# 2. FİLTRELER
st.sidebar.subheader("📂 Filtreler")
filter_columns = get_filterable_columns(df)
active_filters = {}
filtered_indices = df.index.tolist()

if filter_columns:
    for col in filter_columns:
        unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
        options = ["Tümü"] + unique_vals
        selection = st.sidebar.selectbox(f"{col}", options, key=col)
        if selection != "Tümü":
            active_filters[col] = selection

    if active_filters:
        mask = pd.Series([True] * len(df))
        for col, val in active_filters.items():
            mask = mask & (df[col].astype(str) == val)
        filtered_indices = df[mask].index.tolist()
        st.sidebar.success(f"✅ Filtrelenen: {len(filtered_indices)} ürün")
    else:
        st.sidebar.info(f"Tüm veritabanı taranıyor ({len(df)} ürün)")
else:
    st.sidebar.warning("Excel'de kategori sütunu bulunamadı.")

top_k = st.sidebar.slider("Benzerlik Sayısı", 1, 10, 3)

# --- RESİM YÜKLEME ---
uploaded_file = st.file_uploader("Referans Görsel Yükle", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col_left, col_right = st.columns([1, 2])
    
    # 1. RESİM İŞLEME VE GÖSTERİM
    with col_left:
        st.subheader("📤 Aranan")
        original_img = Image.open(uploaded_file).convert("RGB")
        
        # Görüntü İşleme Mantığı
        if ignore_color:
            # 1. Grayscale yap (Rengi at)
            gray_img = ImageOps.grayscale(original_img)
            # 2. Histogram Eşitleme (Kontrastı patlat - Şekli/Doku'yu öne çıkar)
            # Bu işlem, modelin "siyah pantolon" ile "mavi pantolon" arasındaki renk farkını görmesini engeller,
            # bunun yerine "bacak şekli", "cep detayları" gibi yapısal özelliklere odaklanmasını sağlar.
            enhanced_img = ImageOps.equalize(gray_img)
            query_img = enhanced_img.convert("RGB")
            
            st.image(original_img, caption="Orijinal", use_container_width=True)
            st.caption("👀 Yapay Zeka Bunu Görüyor (Şekil Odaklı):")
            st.image(query_img, use_container_width=True)
        else:
            query_img = original_img
            st.image(query_img, use_container_width=True)
        
    # 2. HESAPLAMA
    with st.spinner("Kalıp ve tip analizi yapılıyor..."):
        if len(filtered_indices) == 0:
            st.error("Filtre sonucunda ürün kalmadı. Lütfen filtreleri gevşetin.")
            st.stop()

        query_emb = model.encode(query_img)
        filtered_embeddings = embeddings[filtered_indices]
        
        sims = cosine_similarity([query_emb], filtered_embeddings)[0]
        sorted_local_indices = np.argsort(sims)[-top_k:][::-1]
        
    # 3. SONUÇLAR
    with col_right:
        st.subheader(f"📌 Tip Olarak En Benzer {top_k} Sonuç")
        
        for local_idx in sorted_local_indices:
            score = sims[local_idx]
            global_idx = filtered_indices[local_idx]
            row = df.iloc[global_idx]
            url = get_url_from_row(row)
            
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                
                with c1:
                    if url:
                        st.image(url, width=120)
                    else:
                        st.image("https://placehold.co/120x150?text=Resim+Yok", width=120)
                
                with c2:
                    st.markdown(f"**Benzerlik:** %{score*100:.1f}")
                    if active_filters:
                        st.caption("Filtreye Uygun")
                    
                    details = {k:v for k,v in row.to_dict().items() if str(v) != str(url) and pd.notna(v)}
                    st.json(details, expanded=False)
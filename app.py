import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Görsel Arama Motoru",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. MODEL VE VERİ YÜKLEME (CACHE MEKANİZMASI)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """CLIP Modelini önbelleğe alır ve yükler."""
    with st.spinner('Yapay zeka modeli (CLIP) yükleniyor...'):
        return SentenceTransformer("clip-ViT-B-16", device="cpu")

@st.cache_data
def load_data():
    """Embedding verilerini ve Excel tablosunu yükler."""
    embeddings_path = "embeddings.npy"
    excel_path = "images.xlsx"
    
    # Dosya kontrolü
    if not os.path.exists(embeddings_path) or not os.path.exists(excel_path):
        return None, None, "Veri dosyaları eksik! (embeddings.npy veya images.xlsx)"
    
    try:
        embs = np.load(embeddings_path)
        df = pd.read_excel(excel_path)
        return embs, df, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------------------------------------------------------
# 3. ANA UYGULAMA AKIŞI
# -----------------------------------------------------------------------------

st.title("🔍 Model Resmine En Yakın Ürünü Bulma")
st.markdown("""
Bu uygulama, yüklediğiniz fotoğrafa görsel olarak en çok benzeyen ürünü veritabanından bulur.
""")

# Yan panel ayarları
st.sidebar.header("Ayarlar")
top_k = st.sidebar.slider("Kaç benzer ürün gösterilsin?", min_value=1, max_value=5, value=1)

# Verileri Yükle
model = load_model()
embeddings, df, error_msg = load_data()

# Hata varsa durdur
if error_msg:
    st.error(f"⚠️ Hata: {error_msg}")
    st.info("Lütfen proje klasörüne 'embeddings.npy', 'images.xlsx' dosyalarını ve 'images/' klasörünü eklediğinizden emin olun.")
    st.stop()

# Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Bir model/kıyafet resmi yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.divider()
    
    # İki sütunlu yapı: Sol (Yüklenen), Sağ (Bulunanlar)
    col1, col2 = st.columns([1, 2])

    # --- SOL SÜTUN: KULLANICI RESMİ ---
    with col1:
        st.subheader("📤 Yüklenen Resim")
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, use_container_width=True, caption="Aranan Görsel")

    # --- HESAPLAMA ---
    with st.spinner('Veritabanı taranıyor ve benzerlikler hesaplanıyor...'):
        # Resmi vektöre çevir
        query_emb = model.encode(query_img)

        # Benzerlik skoru hesapla
        sims = cosine_similarity([query_emb], embeddings)[0]

        # En yüksek skora sahip ilk 'top_k' indeksi al
        # argsort küçükten büyüğe sıralar, ters çevirip ilk k tanesini alıyoruz
        top_indices = np.argsort(sims)[-top_k:][::-1]

    # --- SAĞ SÜTUN: SONUÇLAR ---
    with col2:
        st.subheader(f"📌 En Benzer {top_k} Sonuç")
        
        for i, idx in enumerate(top_indices):
            score = sims[idx]
            match_row = df.iloc[idx]
            
            # Resim yolunu belirle (Kullanıcının yapısına göre: images/img_{index}.jpg)
            # NOT: Eğer Excel'de dosya adı sütunu varsa, best_path = match_row['dosya_adi'] şeklinde değiştirin.
            best_path = f"images/img_{idx}.jpg"
            
            with st.container(border=True):
                c_img, c_info = st.columns([1, 2])
                
                with c_img:
                    if os.path.exists(best_path):
                        st.image(best_path, caption=f"Sıra #{i+1}", width=150)
                    else:
                        st.warning(f"Resim bulunamadı: {best_path}")
                        st.image("https://placehold.co/150x200?text=No+Image", width=150)

                with c_info:
                    st.metric(label="Benzerlik Skoru", value=f"%{score*100:.1f}")
                    st.markdown("**Ürün Detayları:**")
                    # Excel verisini göster (boş olmayan sütunları)
                    clean_data = match_row.dropna().to_dict()
                    st.json(clean_data, expanded=False)
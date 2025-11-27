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
    
    # Dosya kontrolü (Sadece temel veri dosyalarını kontrol ediyoruz)
    if not os.path.exists(embeddings_path) or not os.path.exists(excel_path):
        return None, None, "Veri dosyaları eksik! (embeddings.npy veya images.xlsx)"
    
    try:
        embs = np.load(embeddings_path)
        # openpyxl motorunu açıkça belirterek olası motor hatalarını azaltıyoruz
        # Eğer 'Missing optional dependency openpyxl' hatası alırsanız: pip install openpyxl
        df = pd.read_excel(excel_path) # engine='openpyxl' varsayılan olarak denenir
        return embs, df, None
    except Exception as e:
        return None, None, str(e)

def get_image_source(row, index):
    """
    Görsel kaynağını belirler:
    1. Önce yerel 'images/' klasörüne bakar.
    2. Yoksa Excel'deki 'link', 'url', 'image' sütunlarına bakar.
    3. Hiçbiri yoksa None döner.
    """
    # 1. Yerel Dosya Kontrolü
    local_path = f"images/img_{index}.jpg"
    if os.path.exists(local_path):
        return local_path, "local"
    
    # 2. Excel URL Kontrolü (Olası sütun isimleri)
    possible_cols = ['link', 'url', 'image_url', 'gorsel_link', 'resim_link', 'image']
    for col in possible_cols:
        # Büyük/küçük harf duyarlılığını kaldırmak için sütun isimlerini kontrol et
        match_col = next((c for c in row.index if c.lower() == col), None)
        if match_col and isinstance(row[match_col], str) and row[match_col].startswith('http'):
            return row[match_col], "url"
            
    return None, "placeholder"

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
    if "openpyxl" in str(error_msg):
        st.warning("Excel dosyasını okumak için 'openpyxl' kütüphanesine ihtiyacınız var. Lütfen terminale `pip install openpyxl` yazarak yükleyin.")
    else:
        st.info("Lütfen proje klasörüne 'embeddings.npy' ve 'images.xlsx' dosyalarını eklediğinizden emin olun.")
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
        top_indices = np.argsort(sims)[-top_k:][::-1]

    # --- SAĞ SÜTUN: SONUÇLAR ---
    with col2:
        st.subheader(f"📌 En Benzer {top_k} Sonuç")
        
        for i, idx in enumerate(top_indices):
            score = sims[idx]
            match_row = df.iloc[idx]
            
            # Görsel kaynağını belirle
            img_src, src_type = get_image_source(match_row, idx)
            
            with st.container(border=True):
                c_img, c_info = st.columns([1, 2])
                
                with c_img:
                    if src_type == "local":
                        st.image(img_src, caption=f"Sıra #{i+1}", width=150)
                    elif src_type == "url":
                        st.image(img_src, caption=f"Sıra #{i+1} (Web)", width=150)
                    else:
                        # Görsel yoksa placeholder göster
                        st.image("https://placehold.co/150x200/png?text=Gorsel+Yok", caption="Görsel Bulunamadı", width=150)

                with c_info:
                    st.metric(label="Benzerlik Skoru", value=f"%{score*100:.1f}")
                    st.markdown("**Ürün Detayları:**")
                    # Excel verisini göster (URL sütunları ve boşlar hariç daha temiz görünüm)
                    clean_data = {k: v for k, v in match_row.to_dict().items() if pd.notna(v) and not str(v).startswith('http')}
                    st.json(clean_data, expanded=False)
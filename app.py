import streamlit as st
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="BookMatch AI", page_icon="📚", layout="wide")
with st.sidebar:
    st.title("📖 Proje Hakkında")
    st.info("""
    Bu proje, **NLP ** teknikleri kullanılarak geliştirilmiştir.
    - **Algoritma:** Kosinüs Benzerliği
    - **Vektörizasyon:** TF-IDF
    - **Veri Seti:** 50.000+ Goodreads Yorumu
    """)
    st.write("Geliştiren: [İmge Su Çakmak]")
st.title("📚 BookMatch: Akıllı Kitap Tavsiye Sistemi")
st.markdown("Okuduğun ve sevdiğin bir kitabı seç, yapay zeka sana en uygun 5 kitabu bulsun! ")

@st.cache_data
def load_data():
    df_titles = pd.read_pickle("book_titles.pkl")
    book_profiles = pd.read_pickle("book_profiles.pkl")
    unique_books=df_titles.drop_duplicates(subset=["title"])
    return unique_books, book_profiles

@st.cache_resource
def load_matrix():
    return scipy.sparse.load_npz("tfidf_book_matrix.npz")

unique_books, book_profiles = load_data()
book_matrix = load_matrix()

selected_book_title=st.selectbox("🔍 Kütüphanede bir kitap ara veya listeden seç:",unique_books["title"].values)

if st.button("Tavsiye Bul 🚀"):
    with st.spinner("Tavsiye kitaplar bulunuyor..."):
        selected_book_id=unique_books[unique_books["title"]==selected_book_title]["book_id"].values[0]

        try:
            book_idx=book_profiles[book_profiles["book_id"].astype(str)==str(selected_book_id)].index[0]

            sim_scores=linear_kernel(book_matrix[book_idx], book_matrix).flatten()

            related_docs_indices=sim_scores.argsort()[:-7:-1][1:]
            recommended_ids=book_profiles.iloc[related_docs_indices]["book_id"].astype(str)

            recommendations=unique_books[unique_books["book_id"].astype(str).isin(recommended_ids)]
            recommendations = recommendations.drop_duplicates(subset=['title'])

            st.success("Tavsiye kitaplar bulundu! Aşağıda listelenmiştir:")
            st.divider()

            for idx, row in recommendations.iterrows():
                with st.container(border=True): # Her kitap için şık çerçeve
                    col1, col2 = st.columns([1, 4]) # Kutuyu ikiye böl
                
                with col1:
                    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=60)
                
                with col2:
                    st.markdown(f"#### 📖 {row['title']}")
                    st.metric(label="Okur Puanı", value=f"⭐ {row['average_rating']}")

        except IndexError:
            st.error("Seçilen kitap için tavsiye bulunamadı. Lütfen başka bir kitap seçin.")

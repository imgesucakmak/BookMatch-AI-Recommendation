# 📚 BookMatch AI: NLP Tabanlı Akıllı Kitap Tavsiye Sistemi

Bu proje, Goodreads veri seti kullanılarak geliştirilmiş, kullanıcıların okuma tercihlerine göre yeni kitaplar keşfetmesini sağlayan içerik tabanlı bir Yapay Zeka uygulamasıdır. 

## 🚀 Proje Özeti
Kullanıcının seçtiği kitabın içeriğini ve okur yorumlarını analiz eden sistem, **Doğal Dil İşleme** teknikleri kullanarak kütüphanedeki diğer kitaplarla matematiksel bir benzerlik ölçümü yapar ve en uygun 5 kitabı saniyeler içinde önerir.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python
- **Makine Öğrenmesi & NLP:** Scikit-learn (TF-IDF Vectorizer, Cosine Similarity)
- **Veri İşleme:** Pandas, NumPy, SciPy (Sparse Matrices)
- **Web Arayüzü:** Streamlit

## ⚙️ Teknik Mimari ve Model Detayları
1. **Veri Ön İşleme:** 50.000'den fazla Goodreads kullanıcı yorumu temizlendi, noktalama işaretleri ve anlamsız bağlaçlar filtrelendi.
2. **Vektörizasyon:** Yorumlar `TfidfVectorizer` ile 5000 özellikli bir matematiksel matrise dönüştürüldü.
3. **Optimizasyon:** 10 GB'lık devasa benzerlik matrisini bellekte tutmak yerine, SciPy ile sadece kitapların sıkıştırılmış vektörler kaydedildi. Sistem, benzerliği kullanıcının arama yaptığı anda dinamik olarak hesaplayarak RAM tasarrufu sağlar.

## 🚧 Karşılaşılan Mühendislik Zorlukları
* **Veri Seyrekliği ve Soğuk Başlangıç:** Spesifik veya az yoruma sahip kitap baskılarında, TF-IDF vektörleri zayıf kalabilmekte ve model jenerik tavsiyeler üretebilmektedir. İlerleyen versiyonlarda sadece belirli bir yorum eşiğini geçen (örn. >100 yorum) popüler `book_id`'lerin sisteme dahil edilmesi planlanmaktadır.

## 💻 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

**1. Repoyu bilgisayarınıza klonlayın:**
```bash
git clone https://github.com/imgesucakmak/BookMatch-AI-Recommendation.git
pip install -r requirements.txt
streamlit run app.py

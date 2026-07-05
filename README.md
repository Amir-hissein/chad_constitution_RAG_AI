# 🇹🇩 Constitution of Chad — RAG Assistant

> A Retrieval-Augmented Generation (RAG) assistant that answers questions about the
> **Constitution of Chad (2023, revised 2025)** using only the official text — with
> article-level citations and no hallucinations.

**Languages / Diller:** [English](#-english) · [Türkçe](#-türkçe)

**Stack:** Python · LangChain · ChromaDB · BGE-M3 (embeddings) · Ollama (Llama 3.1) · LangGraph

---

## 🇬🇧 English

### Overview

Large Language Models do not know the Constitution of Chad, and if asked directly they
tend to **invent** answers (hallucinate). This project solves that with **RAG**: before
answering, the system **retrieves the most relevant articles** from the real document and
feeds them to the LLM with a strict instruction — *"answer only from this context and cite
the article numbers."*

The result is a trustworthy legal assistant that responds in French (the official language
of the text), grounds every answer in real articles, and refuses when the information is not
in the Constitution.

### How RAG Works (Architecture)

The pipeline is split into two phases:

```
PHASE A — INDEXING (run once, offline)
  1. Load      load_pdf.py          PDF  → clean raw text
  2. Split     split_documents.py   text → chunks (one per article + metadata)
  3. Embed     build_index.py       chunk → vector (BGE-M3)
  4. Index     build_index.py       vectors → ChromaDB (persisted on disk)

PHASE B — QUERYING (run on every question)
  5. Retrieve  rag_chain.py         question → 6 closest articles (semantic search)
  6. Generate  rag_chain.py         articles + question → grounded answer (Llama 3.1)
  7. Orchestrate  graph.py          retrieve → generate as a LangGraph state machine

INTERFACE & EVALUATION
  8. Web UI    app.py               ask questions in the browser (Streamlit)
  9. Evaluate  evaluate.py          score retrieval & answers on eval/questions.json
```

**Key idea — embeddings.** An embedding turns a text into a list of ~1024 numbers (a
*vector*) such that texts with similar *meaning* have close vectors. Searching for *"voting
age"* therefore matches the article that says *"âgés de dix-huit ans"* (eighteen years old)
even though the word "age" never appears — semantic search, not keyword search.

**Golden rule.** The *same* embedding model (BGE-M3) is used both to index and to query.
Otherwise the vectors are not comparable.

### Project Structure

```
chad_constitution_RAG/
├── data/
│   └── constitution_tchad.pdf     # source document (text PDF, not scanned)
├── src/
│   ├── load_pdf.py                # Step 1 — load & clean the PDF
│   ├── split_documents.py         # Step 2 — chunk by article + metadata
│   ├── build_index.py             # Steps 3-4 — embeddings + ChromaDB index
│   ├── rag_chain.py               # Steps 5-6 — retrieval + generation
│   ├── graph.py                   # Step 7 — LangGraph orchestration (optional)
│   ├── app.py                     # Step 8 — Streamlit web interface
│   └── evaluate.py                # Step 9 — automatic evaluation
├── eval/
│   └── questions.json             # Q&A test set used by evaluate.py
├── chroma_db/                     # persisted vector store (auto-generated)
├── requirements.txt
├── .vscode/settings.json          # points the editor to the venv
└── README.md
```

### Requirements

- **Python 3.12+**
- **Ollama** (local, free LLM runtime) — https://ollama.com
- ~7 GB free disk (BGE-M3 model ~2.3 GB + Llama 3.1 ~4.9 GB)

### Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the local LLM
ollama pull llama3.1
```

### Usage

```bash
# Build the vector index (run once — downloads BGE-M3 on first run, a few minutes)
python src/build_index.py

# Option A — Web interface (recommended): opens http://localhost:8501 in your browser
streamlit run src/app.py

# Option B — Command line
python src/rag_chain.py

# Optional — measure quality (retrieval & answers) on a fixed question set
PYTHONPATH=src python src/evaluate.py
```

The **web interface** (`app.py`) shows the generated answer plus an expandable panel listing
the exact articles that were retrieved, so you can check where each answer comes from. The
heavy model/index load is cached, so only the first question is slow.

Each step file is runnable on its own and prints a ✅ / ❌ self-test, so you can validate the
pipeline stage by stage.

### Example

```
Question: Quelles sont les conditions pour être candidat à la présidence ?

Retrieved: Article 68, Article 69, Article 72
Answer:    Pour être candidat à l'élection présidentielle, il faut … (Article 68).
```

If a question is out of scope (e.g. *"What is the capital of France?"*), the assistant
replies: *"Cette information ne figure pas dans la Constitution."*

### Design Notes

- **Chunking by article** keeps each legal unit intact instead of cutting blindly every N
  characters. Very long articles (> 1500 chars) are further split with a
  `RecursiveCharacterTextSplitter`.
- **Metadata** (`article`, `titre`, `source`) travels with each chunk all the way to the
  answer, enabling precise citations.
- **The LLM is a swappable module.** Retrieval, embeddings, and the vector store are
  independent of the generator — you can replace Llama 3.1 with any cloud model (Claude,
  GPT-4o, Gemini) by changing a single line in `rag_chain.py`.

### Known Limitations

- The document preamble (text before Article 1) is intentionally dropped by the article
  splitter.
- On the built-in evaluation set, retrieval is ~80%: a few "opening" articles of a title
  (e.g. 65, 110) are occasionally missed. Raising `k` in `rag_chain.py` helps.

---

## 🇹🇷 Türkçe

### Genel Bakış

Büyük Dil Modelleri (LLM) Çad Anayasası'nı bilmez ve doğrudan sorulduğunda genellikle
cevap **uydurur** (halüsinasyon). Bu proje bunu **RAG** ile çözer: sistem cevap vermeden
önce gerçek belgeden **en alakalı maddeleri getirir** ve bunları LLM'e katı bir talimatla
verir — *"yalnızca bu bağlamdan cevap ver ve madde numaralarını belirt."*

Sonuç, Fransızca (metnin resmî dili) yanıt veren, her cevabı gerçek maddelere dayandıran ve
bilgi Anayasa'da yoksa cevap vermeyi reddeden güvenilir bir hukuk asistanıdır.

### RAG Nasıl Çalışır (Mimari)

İşlem hattı iki aşamaya ayrılır:

```
AŞAMA A — İNDEKSLEME (bir kez, çevrimdışı)
  1. Yükle     load_pdf.py          PDF  → temiz ham metin
  2. Böl       split_documents.py   metin → parçalar (madde başına bir parça + üstveri)
  3. Gömme     build_index.py       parça → vektör (BGE-M3)
  4. İndeksle  build_index.py       vektörler → ChromaDB (diske kaydedilir)

AŞAMA B — SORGULAMA (her soruda)
  5. Getirme   rag_chain.py         soru → en yakın 6 madde (anlamsal arama)
  6. Üretme    rag_chain.py         maddeler + soru → dayanaklı cevap (Llama 3.1)
  7. Düzenleme graph.py             getirme → üretme, LangGraph durum makinesi olarak

ARAYÜZ VE DEĞERLENDİRME
  8. Web Arayüzü app.py             tarayıcıda soru sor (Streamlit)
  9. Değerlendir evaluate.py        eval/questions.json üzerinde getirme ve cevap skoru
```

**Ana fikir — gömmeler (embeddings).** Bir gömme, metni ~1024 sayıdan oluşan bir listeye
(*vektör*) dönüştürür; öyle ki *anlamı* benzer metinler birbirine yakın vektörlere sahip
olur. Bu yüzden *"oy verme yaşı"* aramak, "yaş" kelimesi hiç geçmese bile *"âgés de dix-huit
ans"* (on sekiz yaşında) diyen maddeyle eşleşir — anahtar kelime değil, anlamsal arama.

**Altın kural.** İndeksleme ve sorgulama için *aynı* gömme modeli (BGE-M3) kullanılır. Aksi
halde vektörler karşılaştırılabilir olmaz.

### Proje Yapısı

```
chad_constitution_RAG/
├── data/
│   └── constitution_tchad.pdf     # kaynak belge (taranmış değil, metin PDF)
├── src/
│   ├── load_pdf.py                # Adım 1 — PDF yükleme ve temizleme
│   ├── split_documents.py         # Adım 2 — maddeye göre parçalama + üstveri
│   ├── build_index.py             # Adım 3-4 — gömmeler + ChromaDB indeksi
│   ├── rag_chain.py               # Adım 5-6 — getirme + üretme
│   ├── graph.py                   # Adım 7 — LangGraph düzenlemesi (isteğe bağlı)
│   ├── app.py                     # Adım 8 — Streamlit web arayüzü
│   └── evaluate.py                # Adım 9 — otomatik değerlendirme
├── eval/
│   └── questions.json             # evaluate.py için soru-cevap test seti
├── chroma_db/                     # kalıcı vektör deposu (otomatik oluşturulur)
├── requirements.txt
├── .vscode/settings.json          # editörü venv'e yönlendirir
└── README.md
```

### Gereksinimler

- **Python 3.12+**
- **Ollama** (yerel, ücretsiz LLM çalıştırma ortamı) — https://ollama.com
- ~7 GB boş disk (BGE-M3 modeli ~2.3 GB + Llama 3.1 ~4.9 GB)

### Kurulum

```bash
# 1. Sanal ortam oluştur ve etkinleştir
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Yerel LLM'i indir
ollama pull llama3.1
```

### Kullanım

```bash
# Vektör indeksini oluştur (bir kez — ilk çalıştırmada BGE-M3 indirilir, birkaç dakika)
python src/build_index.py

# Seçenek A — Web arayüzü (önerilir): tarayıcıda http://localhost:8501 açılır
streamlit run src/app.py

# Seçenek B — Komut satırı
python src/rag_chain.py

# İsteğe bağlı — sabit bir soru setinde kaliteyi ölç (getirme ve cevaplar)
PYTHONPATH=src python src/evaluate.py
```

**Web arayüzü** (`app.py`), üretilen cevabın yanı sıra getirilen maddeleri listeleyen
açılabilir bir panel gösterir; böylece her cevabın nereden geldiğini kontrol edebilirsiniz.
Ağır model/indeks yüklemesi önbelleğe alınır, bu yüzden yalnızca ilk soru yavaştır.

Her adım dosyası tek başına çalıştırılabilir ve ✅ / ❌ öz-test yazdırır; böylece işlem
hattını adım adım doğrulayabilirsiniz.

### Örnek

```
Soru:      Quelles sont les conditions pour être candidat à la présidence ?
           (Cumhurbaşkanı adayı olmak için koşullar nelerdir?)

Getirilen: Madde 68, Madde 69, Madde 72
Cevap:     Cumhurbaşkanlığı seçimine aday olmak için … gereklidir (Madde 68).
```

Bir soru kapsam dışıysa (ör. *"Fransa'nın başkenti neresidir?"*), asistan şu yanıtı verir:
*"Cette information ne figure pas dans la Constitution."* (Bu bilgi Anayasa'da yer almıyor.)

### Tasarım Notları

- **Maddeye göre parçalama**, her N karakterde körü körüne kesmek yerine her hukuki birimi
  bütün tutar. Çok uzun maddeler (> 1500 karakter) `RecursiveCharacterTextSplitter` ile
  ayrıca bölünür.
- **Üstveri** (`article`, `titre`, `source`) her parçayla birlikte cevaba kadar taşınır ve
  kesin atıflara olanak tanır.
- **LLM değiştirilebilir bir modüldür.** Getirme, gömmeler ve vektör deposu üreticiden
  bağımsızdır — `rag_chain.py` içinde tek bir satırı değiştirerek Llama 3.1'i herhangi bir
  bulut modeliyle (Claude, GPT-4o, Gemini) değiştirebilirsiniz.

### Bilinen Kısıtlamalar

- Belge önsözü (Madde 1'den önceki metin), madde ayırıcı tarafından bilinçli olarak
  atılır.
- Yerleşik değerlendirme setinde getirme kalitesi ~%80'dir: bir başlığın "açılış" maddeleri
  (ör. 65, 110) bazen kaçırılır. `rag_chain.py` içindeki `k` değerini artırmak yardımcı olur.

---

*Built as a hands-on project to learn Retrieval-Augmented Generation from scratch.*
*Sıfırdan Retrieval-Augmented Generation öğrenmek için uygulamalı bir proje olarak
geliştirilmiştir.*

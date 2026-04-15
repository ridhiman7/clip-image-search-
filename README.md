# Semantic Image Search Engine

Search an image collection using **natural language** or a **reference image** — powered by OpenAI CLIP embeddings and FAISS vector indexing, with a live Streamlit web interface.

---

## Demo

| Text query: *"a bird"* | Image query: upload any photo |
|---|---|
| Returns top-k images ranked by cosine similarity | Finds visually similar images from the index |

---

## How It Works

```
Query (text or image)
        │
        ▼
  CLIP Encoder (ViT-B/32)
        │  512-d L2-normalised embedding
        ▼
  FAISS Index ──── stored embeddings of all indexed images
        │  nearest-neighbour search (inner product = cosine similarity)
        ▼
  Ranked Results (path + score)
        │
        ▼
  Streamlit UI
```

1. **All images** in the collection are encoded into 512-dimensional vectors using CLIP and stored in a FAISS index
2. At **query time**, the text or image query is encoded the same way
3. FAISS finds the closest vectors — images whose CLIP embeddings are nearest to the query
4. Results are ranked by **cosine similarity score** and displayed in a grid

Because CLIP is trained on 400M image-text pairs, text and images live in the same embedding space — so *"a cat sitting on a chair"* and a matching photo will be close together without any fine-tuning.

---

## Features

- **Text-to-image search** — describe what you're looking for in plain English
- **Image-to-image search** — upload a photo to find visually similar results
- **FAISS vector index** — supports Flat (exact), IVF, and IVFPQ index types
- **STL-10 dataset pipeline** — automated download, subset, and class-organised storage at 224×224
- **Streamlit web UI** — live result grid with similarity scores and class labels
- **Evaluation metrics** — Precision@k, Recall@k, mAP@k, Hit Rate@k

---

## Tech Stack

| Component | Library |
|---|---|
| Vision-language model | OpenAI CLIP (ViT-B/32) via HuggingFace Transformers |
| Vector index | FAISS (faiss-cpu) |
| Deep learning | PyTorch |
| Dataset | STL-10 via torchvision (96×96, ImageNet-derived) |
| Web interface | Streamlit |
| Image processing | Pillow |

---

## Project Structure

```
├── app.py                # Streamlit web interface
├── main.py               # CLI: build index and run demo queries
├── prepare_stl10.py      # Download and subset STL-10 into ./images/ (224×224)
├── prepare_cifar10.py    # Legacy: CIFAR-10 pipeline (32×32, kept for reference)
├── embedding.py          # CLIP image and text encoder (CLIPEmbedder)
├── index.py              # FAISS index wrapper (build, search, save, load)
├── search.py             # High-level search engine (text + image queries)
├── data_loader.py        # PyTorch Dataset and DataLoader for image folders
├── evaluate.py           # IR metrics: P@k, R@k, mAP@k, HitRate@k
├── colab_notebook.ipynb  # End-to-end Colab notebook
├── requirements.txt
├── images/               # Dataset images (created by prepare_stl10.py)
│   ├── airplane/
│   ├── bird/
│   ├── car/
│   └── ...
└── index/                # Saved FAISS index (created by main.py --mode build)
    ├── faiss.index
    └── metadata.pkl
```

---

## Deployment — Streamlit Community Cloud (public URL)

Get a permanent public URL anyone can visit — free, no server needed.

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set main file to `app.py` → click **Deploy**
4. On first launch the app automatically downloads STL-10 and builds the index (takes ~3 min)
5. You get a permanent URL: `https://yourname-yourrepo-app.streamlit.app`

No manual setup needed — the app bootstraps itself.

---

## Setup

### Local

```bash
git clone https://github.com/ridhiman7/<your-repo-name>.git
cd <your-repo-name>

pip install -r requirements.txt

# 1. Download STL-10 and save a subset as images (224x224, sharp)
python prepare_stl10.py --per_class 10

# 2. Build the FAISS index
python main.py --image_dir ./images --mode build

# 3. Launch the web app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

### Google Colab

```python
# Cell 1 — clone and install
!git clone https://github.com/ridhiman7/<your-repo-name>.git
%cd <your-repo-name>
!pip install -r requirements.txt -q

# Cell 2 — prepare dataset (STL-10, 10 images/class)
!python prepare_stl10.py --per_class 10

# Cell 3 — build index
!python main.py --image_dir ./images --mode build

# Cell 4 — install localtunnel
!npm install -g localtunnel -q

# Cell 5 — launch Streamlit and get public URL
import subprocess, time

subprocess.Popen(
    ['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(3)

# Print your external IP — enter this as the password if localtunnel asks
!curl -s https://ipv4.icanhazip.com

# Get your public URL
!lt --port 8501
```

---

## Dataset

Uses a subset of **STL-10** — 10 classes, 96×96 native resolution (ImageNet-derived), saved at 224×224 using LANCZOS upscaling for sharp display.

| Class | Description |
|---|---|
| airplane, bird, ship | transport and wildlife |
| car, truck | vehicles |
| cat, dog, horse, deer, monkey | animals |

```bash
# 10 images per class (100 total) — default
python prepare_stl10.py --per_class 10

# 50 images per class (500 total)
python prepare_stl10.py --per_class 50

# Keep native 96×96 resolution (no upscale)
python prepare_stl10.py --size 96

# All in one flat folder instead of subfolders
python prepare_stl10.py --per_class 10 --flat
```

---

## Evaluation

The `evaluate.py` module computes standard information retrieval metrics against ground-truth class labels:

| Metric | Description |
|---|---|
| **Precision@k** | Fraction of top-k results that match the query class |
| **Recall@k** | Fraction of all relevant images found in top-k |
| **mAP@k** | Mean Average Precision — rewards early relevant results |
| **Hit Rate@k** | Whether at least one relevant result appears in top-k |

---

## Index Types

| Type | Speed | Accuracy | When to use |
|---|---|---|---|
| `flat` | Fast | Exact | Default — up to ~100k images |
| `ivf` | Faster | ~Exact | 100k+ images |
| `ivfpq` | Fastest | Approximate | Millions of images, memory-constrained |

```bash
python main.py --image_dir ./images --mode build --index_type flat
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0,<4.46.0
faiss-cpu>=1.7.4
Pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
streamlit>=1.32.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

# 🔍 Fingerprint Recognition — Deep Learning Project

A PyTorch-based fingerprint verification system using Siamese Networks with Contrastive or Triplet Loss, trained on the **SOCOFing dataset** from Kaggle.

---

## 📁 Project Structure

```
fingerprint_recognition/
├── data/
│   ├── download_dataset.py   ← Download SOCOFing from Kaggle
│   └── dataset.py            ← Pair & Triplet dataset loaders
├── models/
│   └── siamese.py            ← CNN Encoder, SiameseNet, TripletNet, losses
├── utils/
│   └── metrics.py            ← EER, AUC, cosine similarity
├── train.py                  ← Training script (contrastive or triplet)
├── inference.py              ← Verify two fingerprint images
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API

1. Go to https://www.kaggle.com/settings/account
2. Click **"Create New API Token"** → downloads `kaggle.json`
3. Place it at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<user>\.kaggle\kaggle.json`

### 3. Download the dataset

```bash
python data/download_dataset.py
```

This downloads the **SOCOFing** dataset (Sokoto Coventry Fingerprint dataset):
- 6,000 real fingerprint images
- 600 subjects × 10 fingers
- Also includes altered versions (Easy / Medium / Hard)

---

## 🚀 Training

### Triplet Loss (recommended)

```bash
python train.py \
  --data_dir ./data/raw \
  --mode triplet \
  --epochs 30 \
  --batch_size 64 \
  --embed_dim 128 \
  --num_pairs 20000
```

### Contrastive Loss (Siamese)

```bash
python train.py \
  --data_dir ./data/raw \
  --mode contrastive \
  --epochs 30 \
  --margin 1.0
```

### Include altered fingerprints (harder training)

```bash
python train.py --data_dir ./data/raw --mode triplet --use_altered
```

Checkpoints are saved to `./checkpoints/`.

---

## 🔬 Evaluation

The training loop automatically reports **validation loss** each epoch.

For full metrics (EER, AUC), use the `evaluate_verification` function from `utils/metrics.py` in a notebook or evaluation script:

```python
from utils.metrics import evaluate_verification
results = evaluate_verification(model, val_loader, device)
print(f"EER: {results['eer']:.4f}  |  AUC: {results['auc']:.4f}")
```

> **EER (Equal Error Rate)** is the standard metric for biometric systems.  
> Lower EER = better. State-of-the-art systems achieve < 1% EER.

---

## 🧪 Inference

Verify if two fingerprint images belong to the same person:

```bash
python inference.py \
  --img1 path/to/finger1.BMP \
  --img2 path/to/finger2.BMP \
  --checkpoint checkpoints/encoder_best.pt \
  --threshold 0.5
```

Output:
```
  Image 1    : finger1.BMP
  Image 2    : finger2.BMP
  Similarity : 0.8732  (threshold=0.5)
  Decision   : ✅ SAME PERSON
```

> **Threshold**: tune based on your EER threshold from evaluation. Higher = stricter matching.

---

## 🏗️ Architecture

```
Input (1×96×96 grayscale)
        ↓
ConvBlock(1→32)   + MaxPool   →  32×48×48
ConvBlock(32→64)  + MaxPool   →  64×24×24
ConvBlock(64→128) + MaxPool   → 128×12×12
ConvBlock(128→256)+ MaxPool   → 256×6×6
ConvBlock(256→256) no pool    → 256×6×6
        ↓
Flatten → FC(512) → ReLU → Dropout(0.3) → FC(128)
        ↓
L2 Normalize → 128-dim embedding
```

Two images pass through the **same encoder** (shared weights).  
Their embeddings are compared by cosine similarity.

---

## 📈 Expected Results

| Mode        | Epochs | EER (approx) |
|-------------|--------|--------------|
| Contrastive | 30     | ~8–12%       |
| Triplet     | 30     | ~4–8%        |
| Triplet + altered data | 50 | ~3–6% |

---

## 🔧 Next Steps / Improvements

- [ ] Add **ArcFace loss** for stronger embeddings
- [ ] Use **SE blocks** (Squeeze-and-Excitation) in the CNN
- [ ] Apply **CLAHE preprocessing** for better ridge contrast
- [ ] Train with **hard negative mining** in triplets
- [ ] Export to **ONNX** for deployment

## 🚀 Project Overview

The project is designed to process the dataset, transforming raw data into structured feature vectors.

### Core Features
- **Text Extraction**: Processes plot summaries using state-of-the-art transformer models (MiniLM, MPNet, CLIP).
- **Image Extraction**: Generates poster embeddings using VGG16, ViT, and CLIP.
- **Audio Extraction**: Extracts deep features from trailer audio using Whisper, Audio Spectrogram Transformer (AST), and VGGish.
- **Video Extraction**: Processes trailer visuals with spatiotemporal models like SlowFast, MViT, and R(2+1)D.

---

## 🛠 Installation & Environments

Because each modality has specific (and sometimes conflicting) dependencies, it is recommended to create **4 separate virtual environments**.

### 1. Text Environment
```bash
# Create and activate env (e.g., venv_text)
pip install -r text_reqs.txt
```

### 2. Image Environment
```bash
# Create and activate env (e.g., venv_image)
pip install -r image_reqs.txt
```

### 3. Audio Environment
```bash
# Create and activate env (e.g., venv_audio)
pip install -r audio_reqs.txt
```

### 4. Video Environment
```bash
# Create and activate env (e.g., venv_video)
pip install -r video_reqs.txt
```

*Note: A CUDA-enabled GPU is strongly recommended for video and audio processing.*

---

## 💻 Usage

All extraction scripts are located in the `src/` directory.

### 1. Text Feature Extraction
Extracts embeddings for movie plots.
```bash
python src/extract_text.py
```
*Models used:* `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `clip-ViT-B-32`.

### 2. Image Feature Extraction
Processes posters in the `data/poster/` directory.
```bash
python src/extract_images.py
```
*Models used:* `clip-ViT-B-32`, `vgg16`, `vit-base-patch16-224`.

### 3. Audio Feature Extraction
**Pre-step:** Download the required models (Whisper, AST, VGGish) to the local `models/` folder.
```bash
python src/download_audio_models.py
```

**Extraction:** Extracts audio from trailers (up to 30s) and generates embeddings.
```bash
python src/extract_audio.py
```
*Models used:* `Whisper`, `AST`, `VGGish`.
*Consolidation:* Results are merged into `.parquet` and `.h5` files automatically.

### 4. Video Feature Extraction
Processes trailer visuals using spatiotemporal architectures.
```bash
python src/extract_slowfast.py
python src/extract_mvit.py
python src/extract_r2p1d.py
```
*Strategy:* These scripts remove the final classification head to output raw feature vectors.

---

## 📊 Output Format

By default, the scripts generate individual JSON files for each movie ID in dedicated folders (e.g., `/minilm/123.json`). 

**JSON Structure:**
```json
{
    "123": [0.123, -0.456, 0.789, ...]
}
```

The audio pipeline also provides consolidated `audio_embeddings_{model}.parquet` and `audio_embeddings_{model}.h5` files for easier batch processing.

---

## 📝 Logs & Error Handling
Each script maintains a log file (e.g., `log_slowfast.txt`, `log_minilm.txt`) to track skipped files or errors during processing, allowing for easy resumption of interrupted tasks.

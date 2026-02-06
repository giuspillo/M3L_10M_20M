# Binge Watch: Reproducible Multimodal Benchmarks for MovieLens-10M and 20M

## Description
This repository provides **M³L-10M** and **M³L-20M**, large-scale and fully reproducible multimodal extensions of the MovieLens-10M and MovieLens-20M datasets. The datasets augment the original MovieLens interaction data with rich **textual, visual, acoustic, and video features** extracted from movie plots, posters, and trailers.

The data is curated with a strong emphasis on **scientific reproducibility**. All preprocessing and feature extraction steps follow a fully documented pipeline, and no aggressive item filtering is applied, preserving the original interaction distributions. The interaction files are formatted for **immediate use with the MMRec framework**, enabling straightforward benchmarking of multimodal recommender systems.

## Dataset Statistics
The datasets maintain the original MovieLens sparsity characteristics while providing high coverage of multimodal side information.

| Metric | M³L-10M | M³L-20M |
|------|--------:|--------:|
| **Total Users** | 69,878 | 138,493 |
| **Total Items** | 9,031 | 19,009 |
| **Total Ratings** | 9,409,884 | 18,777,965 |
| **Sparsity** | 98.66% | 99.29% |
| **Interaction Coverage** | 94.10% | 93.89% |

## Included Multimodal Features
For each movie item, high-dimensional latent embeddings are provided, extracted using widely adopted, state-of-the-art encoders:

- **Textual (Movie plots):**
  - MiniLM  
  - MPNet  
  - CLIP-Text  

- **Visual (Movie posters):**
  - VGG16  
  - Vision Transformer (ViT)  
  - CLIP-Image  

- **Acoustic (Movie trailers):**
  - VGGish  
  - Whisper  
  - Audio Spectrogram Transformer (AST)  

- **Video (Movie trailers):**
  - SlowFast (R50)  
  - R(2+1)D  
  - Multiscale Vision Transformer (MViT)

## Resource Contents
1. **Interaction Data (MMRec format)**  
   - `train.tsv`, `val.tsv`, and `test.tsv` files  
   - Standard 8:1:1 train/validation/test split  

2. **Multimodal Feature Files**  
   - JSON files for each encoder (e.g., `text_mpnet.json`, `audio_whisper.json`)  
   - Each file maps MovieLens item IDs to fixed-length embedding vectors  

## Intended Use
These datasets are designed for **benchmarking and reproducible research** in multimodal recommendation, representation learning, and cross-modal modeling, particularly on large-scale implicit and explicit feedback scenarios.

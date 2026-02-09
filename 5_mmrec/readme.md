# Quantitative analysis with MMRec

This documentation provides a comprehensive guide for running experiments using the [MMRec](https://github.com/enoche/MMRec/tree/master/src) framework with the provided entry scripts: `main_bpr.py`, `main_lattice.py`, `main_freedom.py`, and `main_vbpr.py`.

## 1. Required Packages
To set up the environment for MMRec, creare a new virtual environment and install the following dependencies:

```
# create virtual env
python -m venv _env
source _env/bin/activate

# install dipendencied
pip install torch torchvision torchaudio torchscale numpy pandas scipy pyyaml tqdm scikit-learn
```

We also release our `requirements.txt` file, that can be installed as:
```
pip install -r req.txt
```

---

## 2. Data download
Due to the GitHub file size limit, we cannot publish in this repo the post-processed dataset, so please find them on our [Zenodo](https://zenodo.org/records/18499145) resource. They are already formatted in the MMRec format.

---

## 3. Run Configuration
The framework is pre-configured with the following settings across all scripts:
- Hardware: Configured to run on GPU 0 (`gpu_id: 0`).
- Dataset: Defaults to `m3l-10m`.
- Persistence: The `save_model` flag is enabled (True) to automatically store the best model checkpoints in the `saved/` directory.

### Model Configuration
In `src/configs/model` it is possible to customize the various parameters and hyperparameters of each model. For our analysis, we set:

- BPR
```
embedding_size: 64
reg_weight: [1e-02, 1e-03, 1e-04]

hyper_parameters: ["reg_weight"]
```

- VBPR
```
embedding_size: 64
reg_weight: [1e-02, 1e-03, 1e-04]

hyper_parameters: ["reg_weight"]
```

- LATTICE
```
embedding_size: 64
feat_embed_dim: 64
weight_size: [64, 64]
learning_rate_scheduler: [0.96, 50]
lambda_coeff: 0.9
reg_weight: [1e-02, 1e-03, 1e-04]
cf_model: lightgcn
mess_dropout: [0.1, 0.1]
n_layers: [1,2,3]
knn_k: 10

hyper_parameters: ["reg_weight", "n_layers"]
```

- FREEDOM
```
embedding_size: 64
feat_embed_dim: 64
weight_size: [64, 64]
lambda_coeff: 0.9
reg_weight: [1e-02, 1e-03, 1e-04]
n_mm_layers: [1,2,3]
n_ui_layers: [1,2,3]
knn_k: 10
mm_image_weight: 0.1
dropout: 0.8

hyper_parameters: ["reg_weight", "n_mm_layers", "n_ui_layers"]
```

Moreover, for each model, it is possible to select the modalities it will use as follows:

```
# let's use AST for audio and MViT for video
text_feature_file: "audio/ast.npy"
vision_feature_file: "video/mvit.npy"
```

We can also include them as hyperparameters to be optimized. 
```
# now let's find the best text modality and the best image modality 
text_feature_file: ["text/minilm.npy", "text/mpnet.npy", "text/clip_text.npy"]
vision_feature_file: ["image/vit.npy", "image/vgg.npy", "image/clip_image.npy"] 

# include them among the hyperparameters for the VBPR model
hyper_parameters: ["reg_weight", "n_layers", "text_feature_file", "vision_feature_file"]
```

---

## 4. Usage & Execution Commands
Use the following commands to execute experiments within the MMRec framework. You can use the `-m` and `-d` flags to override model and dataset defaults.

### BPR (General Training)
- File: `main_bpr.py`
- Command: `python main_bpr.py --dataset m3l-10m`

### LATTICE (Latent Structure Mining)
- File: `main_lattice.py`
- Default Model: LATTICE
- Command: `python main_lattice.py --model LATTICE --dataset m3l-10m`

### FREEDOM (Parameter-Efficient Graph Rec)
- File: `main_freedom.py`
- Default Model: FREEDOM
- Command: `python main_freedom.py --model FREEDOM --dataset m3l-10m`

### VBPR (Visual Bayesian Personalized Ranking)
- File: `main_vbpr.py`
- Default Model: VBPR
- Command: python `main_vbpr.py --model VBPR --dataset m3l-10m`

---
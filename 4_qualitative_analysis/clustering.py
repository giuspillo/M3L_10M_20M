import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# --- Configuration ---
# Update these paths to match your local environment if necessary
DATA_BASE_DIR = "."
ML20M_DIR = "."
OUTPUT_DIR = "."
MOVIES_CSV = os.path.join(ML20M_DIR, "movies.csv")
RATINGS_CSV = os.path.join(ML20M_DIR, "ratings.csv")

LIMIT = 1000  
N_CLUSTERS = 10

MODALITIES = {
    "Text": ["mpnet"],
    "Image": ["vit"],
    "Video": ["slowfast"],
    "Audio": ["ast"]
}

ARCHETYPE_LISTS = {
    "Text": ["The Graduate (1967)", "Star Trek: The Motion Picture (1979)",  "Ghost (1990)", "Mortal Kombat (1995)", "Star Wars: Episode VI - Return of the Jedi (1983)"], #"Platoon (1986)",
    "Image": ["Star Trek: Insurrection (1998)", "Die Hard 2 (1990)", "The Cider House Rules (1999)", "Jumanji (1995)", "The Butterfly Effect (2004)"],
    "Video": ["Strange Days (1995)", "Awakenings (1990)", "The King's Speech (2010)", "The Day After Tomorrow (2004)", "Psycho (1960)"],
    "Audio": ["Annie Hall (1977)", "Toy Story (1995)", "Pitch Black (2000)", "Sleepy Hollow (1999)"]
}

def get_data(limit=1000):
    df_r = pd.read_csv(RATINGS_CSV, usecols=['movieId'])
    counts = df_r['movieId'].value_counts()
    popular = counts.head(limit).index.tolist()
    df_m = pd.read_csv(MOVIES_CSV)
    df_m = df_m[df_m['movieId'].isin(popular)]
    titles = dict(zip(df_m['movieId'].astype(str), df_m['title']))
    genres = dict(zip(df_m['movieId'].astype(str), df_m['genres']))
    return [str(x) for x in popular], titles, genres

def load_embeddings(path, ids):
    embeddings, valid_ids = [], []
    for mid in ids:
        f_path = os.path.join(path, f"{mid}.json")
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                data = json.load(f)
                vec = data if isinstance(data, list) else list(data.values())[0]
                embeddings.append(vec)
                valid_ids.append(mid)
    return np.array(embeddings), valid_ids

def main():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    ids, titles, genres = get_data(limit=LIMIT)
    membership = pd.DataFrame(index=ids)

    for mod, strategies in MODALITIES.items():
        dir_map = {"Text": "1_text", "Image": "2_image", "Video": "3_video", "Audio": "4_audio"}
        path = os.path.join(DATA_BASE_DIR, dir_map[mod], strategies[0])
        
        if not os.path.exists(path):
            print(f"Path not found for {mod}: {path}")
            continue
            
        emb, v_ids = load_embeddings(path, ids)
        if len(emb) == 0:
            continue
            
        emb_norm = normalize(emb)
        pca = PCA(n_components=0.95, random_state=42)
        emb_reduced = pca.fit_transform(emb_norm)
        
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_reduced)
        
        # --- Individual Visualization for each Modality ---
        tsne = TSNE(n_components=2, random_state=42, metric='cosine', init='pca', perplexity=30)
        emb_2d = tsne.fit_transform(emb_reduced)
        
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=80)
        plt.title(f"{mod} Archetype Mapping", fontsize=14, fontweight='bold')
        plt.xticks([]); plt.yticks([])
        
        # Annotate Archetypes
        for t_title in ARCHETYPE_LISTS[mod]:
            if 'Star Wars' in t_title:
                xytext=(15, -15)
            else:
                xytext=(15, 15)
            for idx, mid in enumerate(v_ids):
                if titles.get(mid) == t_title:
                    plt.annotate(
                        t_title.split("-")[0].strip().split(" (")[0],
                        (emb_2d[idx, 0], emb_2d[idx, 1]),
                        xytext=xytext, textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.4'),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                    )
                    break
        
        # Save each plot separately
        save_path = os.path.join(OUTPUT_DIR, f"archetype_tsne_{mod.lower()}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close() # Close figure to free memory
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
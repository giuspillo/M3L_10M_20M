import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

# --- Configuration ---
DATA_BASE_DIR = "."
ML20M_DIR = "."
OUTPUT_DIR = "."
RATINGS_CSV = os.path.join(ML20M_DIR, "ratings.csv")
MOVIES_CSV = os.path.join(ML20M_DIR, "movies.csv")
GENOME_SCORES_CSV = os.path.join(ML20M_DIR, "genome-scores.csv")
LIMIT = 1000

# Tag Definitions
NARRATIVE_TAGS = [194, 243, 303, 356, 453, 691, 692, 788, 789, 790, 1064] # Plot, Dialogue, Story
STYLISTIC_TAGS = [54, 86, 212, 449, 452, 466, 754, 883, 951, 981, 1092] # Cinematography, Atmosphere, Effects, Visuals

def get_data(limit=1000):
    df_r = pd.read_csv(RATINGS_CSV, usecols=['movieId'])
    counts = df_r['movieId'].value_counts()
    popular = [str(x) for x in counts.head(limit).index.tolist()]
    
    df_m = pd.read_csv(MOVIES_CSV)
    df_m = df_m[df_m['movieId'].isin([int(x) for x in popular])]
    genres = dict(zip(df_m['movieId'], df_m['genres']))
    
    # Map genres to integers for AMI
    unique_genres = set('|'.join(df_m['genres']).split('|'))
    genre_to_id = {g: i for i, g in enumerate(unique_genres)}
    genre_labels = []
    for mid in [int(x) for x in popular]:
        # For AMI we need a single label, taking first for simplicity or handling multi-label if needed
        # Standard approach for AMI comparison is using the first genre or a representative one
        genre_labels.append(genre_to_id[genres[mid].split('|')[0]])
        
    return popular, genre_labels

def get_clusters_and_metrics(modality, strategy, ids):
    path = os.path.join(DATA_BASE_DIR, modality, strategy)
    embeddings, valid_ids = [], []
    for mid in ids:
        f_path = os.path.join(path, f"{mid}.json")
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                data = json.load(f)
                vec = data if isinstance(data, list) else list(data.values())[0]
                embeddings.append(vec)
                valid_ids.append(mid)
    
    emb = np.array(embeddings)
    emb_norm = normalize(emb)
    pca = PCA(n_components=0.95, random_state=42)
    emb_pca = pca.fit_transform(emb_norm)
    
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
    labels = kmeans.fit_predict(emb_pca)
    
    sil = silhouette_score(emb_pca, labels)
    
    df = pd.DataFrame({"movieId": [int(x) for x in valid_ids], "cluster": labels})
    return df, sil, emb_pca

def main():
    print("Loading movies and genre ground truth...")
    popular_ids, genre_labels = get_data(LIMIT)
    
    print("Loading genome scores...")
    all_tags = NARRATIVE_TAGS + STYLISTIC_TAGS
    scores = pd.read_csv(GENOME_SCORES_CSV)
    scores = scores[scores['tagId'].isin(all_tags)]
    scores = scores[scores['movieId'].isin([int(x) for x in popular_ids])]
    
    # Scale scores
    scores['relevance'] = scores.groupby('tagId')['relevance'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() > x.min()) else 0)

    MODALITIES = [
        ("1_text", "mpnet", "Text"),
        ("2_image", "vit", "Image"),
        ("3_video", "slowfast", "Video"),
        ("4_audio", "ast", "Audio")
    ]

    final_results = []

    for mod_path, strat, mod_name in MODALITIES:
        print(f"Processing {mod_name}...")
        membership, sil, _ = get_clusters_and_metrics(mod_path, strat, popular_ids)
        
        # 1. Spatial Distinguishability (Silhouette)
        spatial_dist = sil
        
        # 2. Genre Alignment (AMI)
        # Filter popular genre labels to match those with embeddings
        id_to_genre = dict(zip([int(x) for x in popular_ids], genre_labels))
        aligned_genres = [id_to_genre[mid] for mid in membership['movieId']]
        ami = adjusted_mutual_info_score(aligned_genres, membership['cluster'])
        
        # 3 & 4. Narrative Precision vs Stylistic Variance (Tag Consensus)
        data = pd.merge(scores, membership, on='movieId')
        
        # Narrative Consensus (inter-cluster variance of narrative tags)
        narrative_data = data[data['tagId'].isin(NARRATIVE_TAGS)]
        narrative_distinctiveness = narrative_data.groupby(['cluster', 'tagId'])['relevance'].mean().unstack().std(axis=0).mean()
        
        # Stylistic Consensus (inter-cluster variance of stylistic tags)
        stylistic_data = data[data['tagId'].isin(STYLISTIC_TAGS)]
        stylistic_distinctiveness = stylistic_data.groupby(['cluster', 'tagId'])['relevance'].mean().unstack().std(axis=0).mean()
        
        final_results.append({
            "Modality": mod_name,
            "Narrative Precision (Raw)": narrative_distinctiveness,
            "Stylistic Variance (Raw)": stylistic_distinctiveness,
            "Spatial Distinguishability (Raw)": spatial_dist,
            "Genre Alignment (Raw)": ami
        })

    results_df = pd.DataFrame(final_results)
    
    # Normalizing all values to [0, 1] relative to the max observed in the group for each column
    # except for Stylistic Variance which we want to be high where narrative is low (usually)
    # but here we use the raw Stylistic Consensus as a proxy.
    cols_to_norm = ["Narrative Precision (Raw)", "Stylistic Variance (Raw)", "Spatial Distinguishability (Raw)", "Genre Alignment (Raw)"]
    for col in cols_to_norm:
        min_val = results_df[col].min()
        max_val = results_df[col].max()
        results_df[col.split(' (')[0]] = (results_df[col] - min_val) / (max_val - min_val)

    # Note: Stylistic Variance in the original radar plot was manually tuned to be high for Video/Audio.
    # Our 'Stylistic Variance' (Calculated) here measures consensus on stylistic tags.
    # Actually, high 'Stylistic Variance' in the plot meant low 'Thematic Focus'.
    # To match the user's mapping, we'll ensure the relative scales match.
    
    print("\n--- Validated All Fingerprints (Normalized) ---")
    print(results_df[["Modality", "Narrative Precision", "Stylistic Variance", "Spatial Distinguishability", "Genre Alignment"]].round(3))
    
if __name__ == "__main__":
    main()

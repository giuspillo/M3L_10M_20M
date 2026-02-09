import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer

df = pd.read_csv('1_download_raw/download_text/REPRO_plot_text.tsv', sep='\t')

# 2. Define the configuration for the three encoders
# Keys are the folder names, values are the HuggingFace/SentenceTransformer model names
encoders_config = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "clip-text": "clip-ViT-B-32" 
}

def process_encodings(dataframe, output_folder, model_name):
    """
    Loads a model, encodes plots, and saves them to JSON.
    """
    print(f"--- Processing: {output_folder} using {model_name} ---")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Load the model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return
    
    counter=0
    # Iterate through the DataFrame
    for index, row in dataframe.iterrows():
        movie_id = row['movie_id']
        plot_text = row['plot']
        if counter == 3:
            break
        counter += 1       
        if os.path.isfile(f'{output_folder}/{movie_id}.json'):
            continue
        
        # MODEL_TYPE is used for the log filename in the error handler
        MODEL_TYPE = output_folder 
        
        try:
            # Encode the text
            # SentenceTransformer handles tokenization and pooling automatically
            emb = model.encode(plot_text)
            
            # --- User Provided Saving Logic ---
            if emb is not None:
                serializable_dict = {str(movie_id): emb.tolist()}
                with open(f'{output_folder}/{movie_id}.json', 'w') as f:
                    json.dump(serializable_dict, f, indent=4)
                print(f"Done {movie_id}")
            else:
                # This branch is rarely hit with SentenceTransformers but kept for logic consistency
                with open(f'log_{MODEL_TYPE}.txt', 'a') as fout:
                    fout.write(f'Error/Skipped {movie_id} (Embedding was None)\n')
            # ----------------------------------
            
        except Exception as e:
            # Catch encoding errors (e.g., empty text, cuda OOM)
            with open(f'log_{MODEL_TYPE}.txt', 'a') as fout:
                fout.write(f'Error/Skipped {movie_id}: {str(e)}\n')


    print(f"Finished processing {output_folder}.\n")

# 3. Execution Loop
if __name__ == "__main__":
    for folder_name, model_id in encoders_config.items():
        process_encodings(df, folder_name, model_id)
        
    print("All tasks completed.")
import os
import torch
import random
import torchvision.transforms
import numpy as np
import itertools
import json
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io.video_reader import VideoReader
from tqdm import tqdm
from itertools import islice, takewhile
import pickle


# --- Core Encoding Logic Function ---
def encode_single_video(video_path: str) -> np.ndarray | None:
    """
    Encodes a single video file using the R(2+1)D model.

    Args:
        video_path: The full path to the video file.

    Returns:
        A NumPy array representing the video's embedding, or None if no clips could be processed.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    print(f"Processing video: {video_path}")
    
    # Re-seed to be consistent with the original script's logic
    seed_everything(42)
    
    try:
        # Initialize the video reader
        reader = VideoReader(video_path)
    except Exception as e:
        print(f"Error initializing VideoReader for {video_path}: {e}")
        return None

    # Frame iterator: frames for the first 30 seconds (pts <= 30)
    frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))
    
    clips_batch = []
    # Process up to 5 clips (5 * 32 frames)
    for _ in range(5):
        # Take 32 frames for one clip
        clip_frames = list(islice(frame_iter, 32)) 
        
        # If we don't have exactly 32 frames, break the loop
        if len(clip_frames) != 32: 
            break 
            
        # 1. Stack frames and normalize to [0, 1]
        clip = torch.stack(clip_frames).to(torch.float32) / 255 # T, H, W, C
        
        # 2. Preprocessing transformations
        clip = torchvision.transforms.Resize((128, 171))(clip)
        clip = torchvision.transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645], 
            std=[0.22803, 0.22145, 0.216989]
        )(clip)
        clip = torchvision.transforms.CenterCrop((112, 112))(clip)
        
        clips_batch.append(clip)
    
    if len(clips_batch) == 0:
        print("No complete 32-frame clips were extracted.")
        return None
        
    # Stack all clips into a single batch tensor: [N_clips, T, H, W, C]
    clips_batch = torch.stack(clips_batch) 
    
    # Reorder dimensions for R(2+1)D input: [N, C, T, H, W]
    model_input = clips_batch.moveaxis(1, 2)
    
    # Run inference
    with torch.no_grad():
        # Get the feature layer output
        out = model(model_input.to(device))['feature_layer'].cpu().detach().flatten(start_dim=1)
        
        # Calculate the video embedding as the mean across all clip embeddings
        video_embedding = torch.mean(out, dim=0)
    
    # Convert the resulting tensor to a numpy array
    emb = video_embedding.cpu().numpy()

    return emb


# --- Setup Functions ---

def seed_everything(seed: int):
    """Seeds all relevant random number generators for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return seed

# --- Model Loading and Setup ---


# Seed the environment for reproducibility
seed_everything(42)

feature_layer = -2
# Load the R(2+1)D model pretrained on Kinetics
original_model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)
if isinstance(feature_layer, int):
    # Get the name of the module corresponding to the feature layer index
    feature_layer = list(dict(original_model.named_modules()).keys())[feature_layer]

# Create a feature extractor model, run on CUDA, and set to evaluation mode
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = create_feature_extractor(original_model, {feature_layer: "feature_layer"}).to(device).eval()

# Freeze all parameters
for params in model.parameters():
    params.requires_grad = False


# read map movie_id (int) -> trailer_path (str)
movie_trailer = dict()
with open('1_download_raw/download_trailers/REPRO_trailer_links.tsv', 'r') as fin:
    for line in fin.readlines():
        movie_id, trailer_path = int(line.split('\t')[0]), line.split('\t')[1].strip()
        movie_trailer[movie_id] = trailer_path


# extract R(2+1)D features and save into a dict
trailer_emb_dict = dict()
counter = 0
for movie_id, trailer_path in movie_trailer.items():

    # print progress
    if counter % 100 == 0:
        print(counter)
    counter += 1

    # skip if already computed
    if os.path.isfile(f'r2p1d/{movie_id}.json'):
        continue

    try:
    
        # extract features as np.array file
        trailer_emb = encode_single_video(trailer_path)

        serializable_dict = {
            movie_id: trailer_emb.tolist() 
            # for movie_id, emb in trailer_emb_dict.items()
        }

        with open(f'r2p1d/{movie_id}.json', 'w') as f:
            json.dump(serializable_dict, f, indent=4)
    except Exception as e:
        with open('log_r2p1d.txt', 'a') as fout:
            fout.write(f'Error with movie {movie_id}: {str(e)}\n')
    # save into dict
    # trailer_emb_dict[movie_id] = trailer_emb
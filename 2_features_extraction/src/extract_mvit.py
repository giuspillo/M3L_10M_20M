import os
import torch
import torch.nn as nn
import random
import torchvision.transforms
import numpy as np
import json
from torchvision.io.video_reader import VideoReader
from itertools import islice, takewhile

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

# --- Core Encoding Logic Function ---
def encode_single_video(video_path: str) -> np.ndarray | None:
    """
    Encodes a single video file using the MViT model.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    # Re-seed to be consistent for every video
    seed_everything(42)
    
    try:
        reader = VideoReader(video_path)
    except Exception as e:
        print(f"Error initializing VideoReader for {video_path}: {e}")
        return None

    # Frame iterator: frames for the first 30 seconds (pts <= 30)
    frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))
    
    clips_batch = []
    # Process up to 5 clips
    for _ in range(5):
        # MViT_32x3 expects exactly 32 frames
        target_frames = 32
        clip_frames = list(islice(frame_iter, target_frames)) 
        
        # Padding logic: If we have > 16 frames but < 32, pad with zeros
        # If < 16, we assume the clip is too short and stop.
        if len(clip_frames) < target_frames:
            if len(clip_frames) > 16:
                # Create padding frames (zeros) matching the shape of the first frame
                padding = [torch.zeros_like(clip_frames[0]) for _ in range(target_frames - len(clip_frames))]
                clip_frames.extend(padding)
            else:
                break 
            
        # 1. Stack frames and convert to float [0, 1]
        clip = torch.stack(clip_frames).to(torch.float32) / 255.0 # T, C, H, W
        
        # 2. Preprocessing transformations for MViT
        # Resize to 224x224 (Standard for MViT)
        clip = torchvision.transforms.Resize((224, 224))(clip)
        
        # Normalize using ImageNet mean/std
        clip = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(clip)
        
        # Center Crop (Standard evaluation practice)
        clip = torchvision.transforms.CenterCrop(224)(clip)
        
        clips_batch.append(clip)
    
    if len(clips_batch) == 0:
        print(f"No complete {target_frames}-frame clips were extracted for {video_path}.")
        return None
        
    # Stack all clips: [N_clips, T, C, H, W]
    clips_batch = torch.stack(clips_batch) 
    
    # Reorder dimensions for MViT input: [Batch, Channels, Time, Height, Width]
    # Current is [Batch, Time, Channels, Height, Width]
    model_input = clips_batch.permute(0, 2, 1, 3, 4)
    
    # Run inference
    with torch.no_grad():
        # MViT Output with Identity Head is [Batch, 768] (Feature Vector)
        # It internally handles the Sequence Pooling
        out_features = model(model_input.to(device))
        
        # Calculate the final video embedding as the mean across all 5 clips
        video_embedding = torch.mean(out_features, dim=0)
    
    # Convert to numpy
    emb = video_embedding.cpu().numpy()
    return emb


# --- Model Loading and Setup ---

seed_everything(42)

print("Loading MViT Base 32x3 Model...")
# Load via Torch Hub to ensure correct architecture and weights
original_model = torch.hub.load("facebookresearch/pytorchvideo", "mvit_base_32x3", pretrained=True)

# Modify the head for feature extraction
# MViT has a 'head' module containing pooling, dropout, and projection.
# We replace the final projection layer ('proj') with Identity to get the 768-dim embedding.
if hasattr(original_model, 'head') and hasattr(original_model.head, 'proj'):
    original_model.head.proj = nn.Identity()
else:
    raise AttributeError("Could not find 'head.proj' in model. Structure might differ.")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = original_model.to(device).eval()

# Freeze all parameters
for params in model.parameters():
    params.requires_grad = False

print(f"Model loaded on {device}.")


# --- Main Execution ---

# Ensure output directory exists
os.makedirs('mvit', exist_ok=True)

# Read map movie_id (int) -> trailer_path (str)
movie_trailer = dict()
if os.path.exists('1_download_raw/download_trailers/REPRO_trailer_links.tsv'):
    with open('1_download_raw/download_trailers/REPRO_trailer_links.tsv', 'r') as fin:
        for line in fin.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                movie_id, trailer_path = int(parts[0]), parts[1]
                movie_trailer[movie_id] = trailer_path
else:
    print("Warning: 'map_movieid_trailerpath.tsv' not found.")

# Extract features and save individually
counter = 0
for movie_id, trailer_path in movie_trailer.items():

    # Print progress
    if counter % 100 == 0:
        print(f"Processed {counter} videos...")
    counter += 1

    # Skip if already computed
    if os.path.isfile(f'mvit/{movie_id}.json'):
        continue

    try:
        # Extract features
        print(f"Encoding {movie_id}...")
        trailer_emb = encode_single_video(trailer_path)

        if trailer_emb is not None:
            serializable_dict = {
                str(movie_id): trailer_emb.tolist() 
            }

            with open(f'mvit/{movie_id}.json', 'w') as f:
                json.dump(serializable_dict, f, indent=4)
        else:
            with open('log_mvit.txt', 'a') as fout:
                fout.write(f'Skipped movie {movie_id}: No valid clips found.\n')
    except Exception as e:
        print(f"Error processing {movie_id}: {e}")
        with open('log_mvit.txt', 'a') as fout:
            fout.write(f'Error with movie {movie_id}: {str(e)}\n')
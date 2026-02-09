import os
import torch
import torch.nn as nn
import random
import torchvision.transforms
import numpy as np
import json
from torchvision.io.video_reader import VideoReader
from itertools import islice, takewhile

import gc 
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

# --- SlowFast Helper: Pack Pathways ---
def pack_pathway_output(frames, alpha=4):
    """
    Prepares the input for SlowFast.
    Args:
        frames: Tensor of shape [C, T, H, W]
        alpha: The ratio between Fast and Slow pathways (usually 4).
    Returns:
        List of [slow_path, fast_path] tensors.
    """
    # Define indices for Fast and Slow pathways
    # Assuming input 'frames' is already subsampled to the model's expected "Fast" rate.
    # For SlowFast R50 8x8, we typically want 32 frames for Fast, and 8 for Slow.
    
    fast_pathway = frames
    
    # Slow pathway takes 1 frame for every 'alpha' frames in the fast pathway
    slow_pathway = torch.index_select(
        frames,
        1,
        torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long()
    )
    
    return [slow_pathway, fast_pathway]

# # --- Core Encoding Logic Function ---
# def encode_single_video(video_path: str) -> np.ndarray | None:
#     """
#     Encodes a single video file using the SlowFast model.
#     """
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found at {video_path}")
#         return None

#     seed_everything(42)
    
#     try:
#         reader = VideoReader(video_path)
#     except Exception as e:
#         print(f"Error initializing VideoReader for {video_path}: {e}")
#         return None

#     # Frame iterator: frames for the first 30 seconds
#     frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))
    
#     clips_batch_slow = []
#     clips_batch_fast = []
    
#     # Process up to 5 clips
#     for _ in range(5):
#         target_frames = 64 
#         clip_frames = list(islice(frame_iter, target_frames)) 
        
#         # Padding Logic
#         if len(clip_frames) < target_frames:
#             if len(clip_frames) > 32:
#                 padding = [torch.zeros_like(clip_frames[0]) for _ in range(target_frames - len(clip_frames))]
#                 clip_frames.extend(padding)
#             else:
#                 break 
            
#         # 1. Stack frames [T, C, H, W] and normalize to [0, 1]
#         clip = torch.stack(clip_frames).to(torch.float32) / 255.0
        
#         # 2. Subsample and Transforms
#         # IMPORTANT: Keep dimensions as [T, C, H, W] for torchvision transforms.
#         # 'Normalize' treats T as Batch, which correctly applies normalization to C.
        
#         # Subsample: 64 frames -> 32 frames (Stride 2)
#         clip = clip[::2] # [32, C, H, W]

#         # Define Transforms (Resize/Crop/Normalize)
#         transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(256),
#             torchvision.transforms.CenterCrop(256),
#             torchvision.transforms.Normalize(
#                 mean=[0.45, 0.45, 0.45], 
#                 std=[0.225, 0.225, 0.225]
#             )
#         ])
        
#         # Apply transforms
#         clip = transform(clip) # Output shape still [32, 3, 256, 256]
        
#         # 3. Permute to [C, T, H, W] for SlowFast Model
#         clip = clip.permute(1, 0, 2, 3) # [3, 32, 256, 256]
        
#         # 4. Pack Pathways
#         pathways = pack_pathway_output(clip, alpha=4)
        
#         clips_batch_slow.append(pathways[0])
#         clips_batch_fast.append(pathways[1])
    
#     if len(clips_batch_slow) == 0:
#         print(f"No complete clips extracted for {video_path}.")
#         return None
        
#     batch_slow = torch.stack(clips_batch_slow).to(device)
#     batch_fast = torch.stack(clips_batch_fast).to(device)
    
#     with torch.no_grad():
#         out_features = model([batch_slow, batch_fast])
#         video_embedding = torch.mean(out_features, dim=0)
    
#     emb = video_embedding.cpu().numpy()
#     return emb

def encode_single_video(video_path: str) -> np.ndarray | None:
    """
    Encodes a single video file using the SlowFast model.
    Optimized to process clips sequentially to prevent OOM.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    seed_everything(42)
    
    try:
        reader = VideoReader(video_path)
    except Exception as e:
        print(f"Error initializing VideoReader for {video_path}: {e}")
        return None

    # Frame iterator
    frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))
    
    # Store accumulated embeddings (small vectors), not video frames
    embeddings_accumulator = []
    
    # Process up to 5 clips sequentially
    for _ in range(5):
        target_frames = 64 
        clip_frames = list(islice(frame_iter, target_frames)) 
        
        # Padding Logic
        if len(clip_frames) < target_frames:
            if len(clip_frames) > 32:
                padding = [torch.zeros_like(clip_frames[0]) for _ in range(target_frames - len(clip_frames))]
                clip_frames.extend(padding)
            else:
                break 
            
        # --- Preprocessing (Same as before) ---
        clip = torch.stack(clip_frames).to(torch.float32) / 255.0
        
        # Subsample & Transform
        clip = clip[::2] # [32, C, H, W]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.Normalize(
                mean=[0.45, 0.45, 0.45], 
                std=[0.225, 0.225, 0.225]
            )
        ])
        clip = transform(clip) 
        clip = clip.permute(1, 0, 2, 3) # [C, T, H, W]
        
        # Pack Pathways
        pathways = pack_pathway_output(clip, alpha=4)
        
        # Prepare single-item batch [1, C, T, H, W]
        # Move to GPU immediately and delete CPU reference
        input_slow = pathways[0].unsqueeze(0).to(device)
        input_fast = pathways[1].unsqueeze(0).to(device)
        
        # --- Inference Per Clip ---
        with torch.no_grad():
            # Output shape: [1, 2304]
            clip_emb = model([input_slow, input_fast])
            # Move result to CPU and store
            embeddings_accumulator.append(clip_emb.cpu())
            
        # Explicitly clean up heavy tensors
        del clip, clip_frames, pathways, input_slow, input_fast
        torch.cuda.empty_cache() # Optional: keeps VRAM clean
    
    if len(embeddings_accumulator) == 0:
        print(f"No complete clips extracted for {video_path}.")
        return None
    

    # Stack and Average the embedding vectors (Cheap operation)
    # [5, 2304] -> [2304]
    all_embeddings = torch.vstack(embeddings_accumulator)
    video_embedding = torch.mean(all_embeddings, dim=0)
    
    return video_embedding.numpy()

# --- Model Loading and Setup ---

seed_everything(42)

print("Loading SlowFast R50 Model...")
# Load via Torch Hub
# 'slowfast_r50' is the standard ResNet50 backbone model
original_model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = original_model.to(device).eval()

# Freeze parameters
for params in model.parameters():
    params.requires_grad = False

# Modify the Head for Feature Extraction
# In SlowFast, the classification head is typically in 'blocks[-1]'
# It usually contains a Dropout and a Projection (Linear) layer.
# We replace the Projection layer 'proj' with Identity.
if hasattr(model, 'blocks'):
    # Access the last block (Head)
    head_block = model.blocks[-1]
    if hasattr(head_block, 'proj'):
        print(f"Replaced classification head (Output dim: {head_block.proj.out_features}) with Identity.")
        head_block.proj = nn.Identity()
    else:
        # Fallback: Depending on version, it might be just the block itself if it's a simple Linear
        print("Warning: Could not find 'proj' in blocks[-1]. Checking structure...")
else:
    raise AttributeError("Model structure differs from expected SlowFast format.")

print(f"Model loaded on {device}.")


# --- Main Execution ---

os.makedirs('slowfast', exist_ok=True)

# Read map
movie_trailer = dict()
if os.path.exists('1_download_raw/download_trailers/REPRO_trailer_links.tsv'):
    with open('1_download_raw/download_trailers/REPRO_trailer_links.tsv', 'r') as fin:
        for line in fin.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                movie_trailer[int(parts[0])] = parts[1]
else:
    print("Warning: Map file not found.")

counter = 0
for movie_id, trailer_path in movie_trailer.items():

    gc.collect()

    if os.path.isfile(f'slowfast/{movie_id}.json'):
        continue

    try:
        print(f"Encoding {movie_id}...")
        trailer_emb = encode_single_video(trailer_path)

        if trailer_emb is not None:
            serializable_dict = {
                str(movie_id): trailer_emb.tolist() 
            }
            with open(f'slowfast/{movie_id}.json', 'w') as f:
                json.dump(serializable_dict, f, indent=4)
        else:
            with open('log_slowfast.txt', 'a') as fout:
                fout.write(f'Skipped movie {movie_id}: No valid clips.\n')
        counter += 1
    except Exception as e:
        print(f"Error {movie_id}: {e}")
        with open('log_slowfast.txt', 'a') as fout:
            fout.write(f'Error {movie_id}: {str(e)}\n')
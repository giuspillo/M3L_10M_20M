import os
import torch
import random
import numpy as np
import json
import torchaudio
import av
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperModel, WhisperFeatureExtractor, ASTModel, AutoFeatureExtractor
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 1         
NUM_WORKERS = 4         # Ridotto per risparmiare RAM CPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LOCAL_ROOT = "models"
MAP_PATH = '1_download_raw/download_trailers/REPRO_trailer_links.tsv'
BASE_OUT_DIR = "."

# DURATION LIMIT (Fondamentale per evitare OOM)
MAX_DURATION_SEC = 30  # Whisper e AST usano max 30s (o 10s), inutile caricare di più.

torch.backends.cudnn.benchmark = True

# ==========================================
# 1. UTILITIES PER SALVATAGGIO
# ==========================================

def consolidate_results(model_type, output_dir):
    # If both parquet and h5 exist, skip
    if os.path.exists(os.path.join(BASE_OUT_DIR, f"audio_embeddings_{model_type}.parquet")) and os.path.exists(os.path.join(BASE_OUT_DIR, f"audio_embeddings_{model_type}.h5")):
        print(f"\n[Consolidation] Skipping {model_type} as both parquet and h5 exist.")
        return
    
    print(f"\n[Consolidation] Reading JSON files for {model_type}...")
    all_data = []
    files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not files:
        print("No JSON files found to consolidate.")
        return

    for f_name in tqdm(files, desc="Merging JSONs"):
        path = os.path.join(output_dir, f_name)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for mid, emb_list in data.items():
                    all_data.append({'movie_id': int(mid), 'embedding': emb_list})
        except Exception as e:
            continue # Skip corrupt files

    if not all_data:
        return

    df = pd.DataFrame(all_data)
    df = df.sort_values(by='movie_id').reset_index(drop=True)
    
    parquet_path = os.path.join(BASE_OUT_DIR, f"audio_embeddings_{model_type}.parquet")
    print(f"Saving Parquet to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    h5_path = os.path.join(BASE_OUT_DIR, f"audio_embeddings_{model_type}.h5")
    print(f"Saving HDF5 to {h5_path}...")
    
    matrix_embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
    ids = df['movie_id'].to_numpy(dtype=np.int32)

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('movie_ids', data=ids)
        hf.create_dataset('embeddings', data=matrix_embeddings)

    print("Consolidation Complete.")
    return 

# ==========================================
# 2. DATASET & DATALOADER (FIXED MEMORY LEAK)
# ==========================================
class VideoAudioDataset(Dataset):
    def __init__(self, movie_map, target_sample_rate=16000, max_seconds=30):
        self.movie_ids = list(movie_map.keys())
        self.paths = list(movie_map.values())
        self.target_sample_rate = target_sample_rate
        # Calcoliamo il numero massimo di campioni da caricare
        self.max_frames = target_sample_rate * max_seconds

    def __len__(self):
        return len(self.movie_ids)

    def _load_with_torchaudio(self, path):
        try:
            # 1. Ottieni info senza caricare il file
            info = torchaudio.info(path)
            orig_rate = info.sample_rate
            
            # Calcoliamo quanti frame leggere dal file originale per avere 30s
            frames_to_read = int(self.max_frames * (orig_rate / self.target_sample_rate))
            # Aggiungiamo un buffer di sicurezza del 10%
            frames_to_read = int(frames_to_read * 1.1)

            # 2. Carica SOLO i primi N frame
            waveform, sample_rate = torchaudio.load(path, num_frames=frames_to_read)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Taglio finale preciso
            if waveform.shape[-1] > self.max_frames:
                waveform = waveform[:, :self.max_frames]

            return waveform.squeeze(0)
        except:
            return None

    def _load_with_av(self, path):
        try:
            container = av.open(path)
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            if audio_stream is None:
                container.close()
                return None
            
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=self.target_sample_rate)
            
            npy_frames = []
            total_samples = 0
            
            # Decodifica solo finché non raggiungiamo il limite
            for frame in container.decode(audio_stream):
                frame.pts = None
                resampled = resampler.resample(frame)
                
                if resampled:
                    chunk = resampled[0].to_ndarray() # [1, Samples]
                    npy_frames.append(chunk)
                    total_samples += chunk.shape[1]
                
                if total_samples >= self.max_frames:
                    break
            
            container.close()

            if not npy_frames:
                return None
            
            waveform = np.concatenate(npy_frames, axis=1)
            
            # Taglio finale preciso
            if waveform.shape[1] > self.max_frames:
                waveform = waveform[:, :self.max_frames]

            return torch.from_numpy(waveform).squeeze(0)
        except Exception:
            return None

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        movie_id = self.movie_ids[idx]
        
        if not os.path.exists(video_path):
            return movie_id, None
            
        waveform = self._load_with_torchaudio(video_path)
        if waveform is None:
            waveform = self._load_with_av(video_path)
            
        return movie_id, waveform

def collate_fn_audio(batch):
    movie_ids = []
    waveforms = []
    for mid, wave in batch:
        if wave is not None:
            movie_ids.append(mid)
            waveforms.append(wave.numpy())
    return movie_ids, waveforms

# ==========================================
# 3. MODEL & INFERENCE
# ==========================================

def load_model_components(model_type):
    print(f"Loading {model_type}...")
    if model_type == 'vggish':
        os.environ['TORCH_HOME'] = os.path.join(LOCAL_ROOT, "torch_hub")
        model = torch.hub.load(
            f'{LOCAL_ROOT}/torch_hub/hub/harritaylor_torchvggish_master', 
            'vggish', source='local'
        )
        model.eval().to(DEVICE)
        return model, None
    elif model_type == 'whisper':
        path = os.path.join(LOCAL_ROOT, "whisper")
        model = WhisperModel.from_pretrained(path, local_files_only=True).to(DEVICE)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(path, local_files_only=True)
        model.eval()
        return model, feature_extractor
    elif model_type == 'ast':
        path = os.path.join(LOCAL_ROOT, "ast")
        model = ASTModel.from_pretrained(path, local_files_only=True).to(DEVICE)
        feature_extractor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
        model.eval()
        return model, feature_extractor

def run_inference_batch(model_type, model, processor, waveforms):
    if not waveforms:
        return []
    
    # Whisper e AST beneficiano di AMP, VGGish meno (spesso richiede float32 strict)
    use_amp = (model_type in ['whisper', 'ast'])
    
    # IMPORTANTE: torch.cuda.amp.autocast è vecchio, usiamo quello nuovo o compatibile
    # Se hai torch > 2.0 usa torch.amp.autocast('cuda', ...)
    # Per compatibilità, usiamo ancora cuda.amp ma gestito bene
    
    try:
        autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
    except AttributeError:
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    with autocast_ctx:
        with torch.no_grad():
            if model_type == 'whisper':
                inputs = processor(
                    waveforms, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=480000
                )
                input_features = inputs.input_features.to(DEVICE)
                outputs = model.encoder(input_features)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                return embeddings.float().cpu().numpy()

            elif model_type == 'ast':
                inputs = processor(
                    waveforms, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=1024
                )
                input_values = inputs.input_values.to(DEVICE)
                outputs = model(input_values)
                return outputs.pooler_output.float().cpu().numpy()

            elif model_type == 'vggish':
                results = []
                for wav in waveforms:
                    # VGGish vuole float32
                    # t_wav = torch.as_tensor(wav, device=DEVICE, dtype=torch.float32)
                    emb = model.forward(wav, fs=16000)
                    if emb.shape[0] > 0:
                        res = torch.mean(emb, dim=0)
                    else:
                        res = torch.zeros(128, device=DEVICE)
                    results.append(res.cpu().numpy())
                return np.array(results)
    return []

# ==========================================
# 4. MAIN
# ==========================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)

    # Load Map
    movie_trailer = dict()
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, 'r') as fin:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    movie_trailer[int(parts[0])] = parts[1]
    else:
        print("Error: Map file not found.")
        exit()

    for MODEL_TYPE in ['whisper', 'ast', 'vggish']:
        print(f"\n=== Starting {MODEL_TYPE} ===")
        torch.cuda.empty_cache()
        
        out_dir = os.path.join(BASE_OUT_DIR, f"audio_{MODEL_TYPE}")
        os.makedirs(out_dir, exist_ok=True)

        existing_files = set(os.listdir(out_dir))
        to_process = {}
        for k, v in movie_trailer.items():
            if f"{k}.json" not in existing_files:
                to_process[k] = v
        
        print(f"Videos to process: {len(to_process)} / {len(movie_trailer)}")

        if len(to_process) > 0:
            dataset = VideoAudioDataset(to_process, max_seconds=30)
            dataloader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=NUM_WORKERS, 
                collate_fn=collate_fn_audio,
                pin_memory=True,
                prefetch_factor=2
            )

            model, processor = load_model_components(MODEL_TYPE)
            pbar = tqdm(dataloader, desc=f"Encoding {MODEL_TYPE}", unit="batch")

            for batch_ids, batch_waveforms in pbar:
                if len(batch_ids) == 0:
                    continue

                embeddings = run_inference_batch(MODEL_TYPE, model, processor, batch_waveforms)
                
                for i, mid in enumerate(batch_ids):
                    emb = embeddings[i]
                    emb_list = np.round(emb, 6).tolist()
                    
                    dst_path = os.path.join(out_dir, f'{mid}.json')
                    with open(dst_path, 'w') as f:
                        json.dump({str(mid): emb_list}, f)

        # Consolida sempre alla fine
        consolidate_results(MODEL_TYPE, out_dir)

    print("\nProcessing Complete.")
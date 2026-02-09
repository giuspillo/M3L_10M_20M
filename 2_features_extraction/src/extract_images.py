import os
import json
import torch
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel
from torchvision import models, transforms
import torch.nn as nn

# 1. Configuration: Path to your images and output folders
# Replace 'path/to/images' with your actual image folder
IMAGE_FOLDER_PATH = '1_download_raw/download_posters/REPRO_poster_links.tsv' 

# 2. Wrapper Classes to Standardize "encode()"
# These classes make VGG and ViT behave exactly like SentenceTransformer (CLIP)

class VGGEncoder:
    def __init__(self):
        print("Loading VGG16 (Torchvision)...")
        # Load pre-trained VGG16
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Remove the last classification layer to get the 4096-dim embedding
        # VGG classifier structure: [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear(Output)]
        # We take everything up to the last Linear layer
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        self.model.eval() # Set to evaluation mode
        
        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image):
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.preprocess(image).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            output = self.model(input_tensor)
        return output[0].numpy() # Return flat vector

class ViTEncoder:
    def __init__(self):
        print("Loading ViT (HuggingFace)...")
        self.model_name = 'google/vit-base-patch16-224-in21k'
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)
        self.model.eval()

    def encode(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the pooler_output (CLS token) as the embedding
        return outputs.pooler_output[0].numpy()

# 3. Define the configuration
# For CLIP, we use SentenceTransformer directly. For others, we use our wrappers.
encoders_config = {
    "clip-image": "clip-ViT-B-32", # Handled by SentenceTransformer
    "vgg": "vgg-wrapper",    # Custom wrapper
    "vit": "vit-wrapper"     # Custom wrapper
}

def load_encoder(model_id):
    """Factory function to load the correct model type."""
    if model_id == "vgg-wrapper":
        return VGGEncoder()
    elif model_id == "vit-wrapper":
        return ViTEncoder()
    else:
        # Fallback to SentenceTransformer (perfect for CLIP)
        return SentenceTransformer(model_id)

def process_image_encodings(image_folder, output_folder, model_id):
    """
    Loads a model, encodes images from a folder, and saves them to JSON.
    """
    print(f"--- Processing: {output_folder} using {model_id} ---")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load Model
    try:
        model = load_encoder(model_id)
    except Exception as e:
        print(f"Failed to load model {model_id}: {e}")
        return

    # Get list of image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

    for filename in image_files:
        # Use filename as ID (minus extension)
        image_id = os.path.splitext(filename)[0]
        
        # Resume Check
        if os.path.isfile(f'{output_folder}/{image_id}.json'):
            continue
            
        image_path = os.path.join(image_folder, filename)
        MODEL_TYPE = output_folder 

        try:
            # Load Image
            img = Image.open(image_path)
            
            # Encode
            emb = model.encode(img)
            
            # Save
            if emb is not None:
                serializable_dict = {str(image_id): emb.tolist()}
                with open(f'{output_folder}/{image_id}.json', 'w') as f:
                    json.dump(serializable_dict, f, indent=4)
                print(f"Done {image_id}")
            
        except Exception as e:
            with open(f'log_{MODEL_TYPE}.txt', 'a') as fout:
                fout.write(f'Error {filename}: {str(e)}\n')
                print(f"Error processing {filename}: {e}")


    print(f"Finished processing {output_folder}.\n")

# 4. Execution Loop
if __name__ == "__main__":
    # Ensure your image folder exists before running
    if not os.path.exists(IMAGE_FOLDER_PATH):
        print(f"Error: The folder '{IMAGE_FOLDER_PATH}' does not exist. Please create it and add images.")
    else:
        for folder_name, model_id in encoders_config.items():
            process_image_encodings(IMAGE_FOLDER_PATH, folder_name, model_id)
            
        print("All tasks completed.")
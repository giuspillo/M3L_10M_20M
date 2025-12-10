import requests
import json
from tqdm import tqdm
import os
import time
import pandas as pd
import random
import shutil


def human_delay():
    base = random.uniform(8, 13)
    jitter = random.uniform(-0.5, 0.5)
    time.sleep(base + jitter)

def download_poster(item_id, url, destination_folder):

    if '.png' in url:
        extension = '.png'
    elif '.jpg' in url:
        extension = '.jpg'
    else:
        extension = ''
    
    filename = f'{destination_folder}/{item_id}{extension}'
    response = requests.get(url)

    with open(filename, "wb") as file:
        file.write(response.content)

 

# ================================
#     MAIN SCRIPT STARTS HERE
# ================================

df = pd.read_csv("REPRO_poster_links.tsv", sep="\t", header=None, names=["movieId", "poster_link"])

os.makedirs("poster", exist_ok=True)

for _, row in df.iterrows():
    movie_id = row["movieId"]
    url = row["poster_link"]

    # skip if file is already downloaded
    base = f"poster/{movie_id}"
    if (
        os.path.exists(base)
        or os.path.exists(base + ".jpg")
        or os.path.exists(base + ".png")
    ):
        continue

    human_delay()
    download_poster(movie_id, url, "poster")



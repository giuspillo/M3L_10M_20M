import yt_dlp
import time
import pandas as pd
import random
import shutil
import os


def human_delay():
    # mimic human pacing
    base = random.uniform(8, 13)
    jitter = random.uniform(-0.8, 0.8)
    time.sleep(base + jitter)


def download_youtube_video(item_id, url, destination_folder):
    filename = f'{destination_folder}/{item_id}'

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    ydl_opts = {
        'outtmpl': filename,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': True,
        'cookiefile': 'cookies.txt',
        'http_headers': headers,
        'nopart': True,
        'noprogress': True,
        'concurrent_fragment_downloads': 1,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# ================================
#     MAIN SCRIPT STARTS HERE
# ================================

df = pd.read_csv("REPRO_trailer_links.tsv", sep="\t", header=None, names=["movieId", "trailer_link"])

os.makedirs("trailer", exist_ok=True)

for _, row in df.iterrows():
    movie_id = row["movieId"]
    url = row["trailer"]

    # skip if file is already downloaded
    base = f"trailer/{movie_id}"
    if (
        os.path.exists(base)
        or os.path.exists(base + ".mp4")
        or os.path.exists(base + ".mkv")
        or os.path.exists(base + ".webm")
    ):
        continue

    human_delay()
    download_youtube_video(movie_id, url, "trailer")

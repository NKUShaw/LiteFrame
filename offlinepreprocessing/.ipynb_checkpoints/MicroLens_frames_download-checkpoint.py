import os
import cv2
import random
import shutil
import requests
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count


BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_videos/"
SAVE_RANDOM_DIR = "./Data/MicroLens-100k/MicroLens-100k_random_frames"
SAVE_AVG_DIR = "./Data/MicroLens-100k/MicroLens-100k_average_20_frames"
TMP_VIDEO_DIR = "./tmp_videos"

os.makedirs(SAVE_RANDOM_DIR, exist_ok=True)
os.makedirs(SAVE_AVG_DIR, exist_ok=True)
os.makedirs(TMP_VIDEO_DIR, exist_ok=True)


def download_video(video_id):
    url = f"{BASE_URL}{video_id}.mp4"
    save_path = os.path.join(TMP_VIDEO_DIR, f"{video_id}.mp4")

    if os.path.exists(save_path):
        return save_path  # 已下载则直接使用

    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return save_path
    except Exception as e:
        print(f"[ERROR] download {video_id} failed: {e}")
        return None

def extract_random_frames(cap, video_id):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = random.sample(range(frame_count), 5)

    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(os.path.join(SAVE_RANDOM_DIR, f"{video_id}-{i}.jpg"), frame)

def extract_average_frames(cap, video_id):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 1:
        return

    # 生成 20 个均匀分布的帧索引
    indices = np.linspace(0, frame_count - 1, 20).astype(int)

    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        # resize
        frame = cv2.resize(frame, (224, 224))

        # 保存
        cv2.imwrite(os.path.join(SAVE_AVG_DIR, f"{video_id}-{i}.jpg"), frame)
        

def process_video(video_id):
    # 1. 下载视频
    path = download_video(video_id)
    if path is None:
        return

    # 2. 打开视频
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_id}")
        return

    # 3. 抽帧
    try:
        extract_average_frames(cap, video_id)
    except Exception as e:
        print(f"[ERROR] extract frames for {video_id} failed: {e}")

    cap.release()

    # 4. 删除视频
    try:
        os.remove(path)
    except:
        pass

if __name__ == "__main__":
    # ID 从 1 到 19XXX（修改成你的最大值）
    MAX_ID = 19738
    video_ids = list(range(1, MAX_ID + 1))

    print("Start processing...")
    with Pool(processes=100) as pool:   
        list(tqdm(pool.imap(process_video, video_ids), total=len(video_ids)))

    print("All done!")

import os
import re
import torch
import argparse
import requests
import cv2
import numpy as np
import shutil
import time
import random
import threading
from queue import Queue
from PIL import Image
from torchvision.transforms import v2
from requests.auth import HTTPBasicAuth
import clip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ======================
# 全局配置
# ======================

GPU_ID = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0)) if os.environ.get('CUDA_VISIBLE_DEVICES') else 0

# 加载数据
# try:
#     df = torch.load("./baseline_vit_video_feature_dict.pt")
#     ALL_IDS = list(df.keys())
# except FileNotFoundError:
#     print("Error: Feature file not found. Please check path.")
#     ALL_IDS = []

VIDEO_URL_BASE = "https://fi.ee.tsinghua.edu.cn/datasets/short-video-dataset/raw_file/"
USERNAME = "videodata"
PASSWORD = "ShortVideo@10000"

TEMP_DIR = f"./temp_videos_gpu{GPU_ID}"
TITLE_DIR = "./temp_titles"
FILES_DINO_BASE = "./Data/ShortVideo-Dataset/dino_base"
FILES_DINO_RESAMPLE = "./Data/ShortVideo-Dataset/dino_resample"

# 配置参数
BATCH_SIZE = 8    
NUM_WORKERS_DL = 2  
NUM_WORKERS_DEC = 2  
QUEUE_SIZE = 4       

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FILES_DINO_BASE, exist_ok=True)
os.makedirs(FILES_DINO_RESAMPLE, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_all_ids_from_server():
    print(f"正在从服务器获取全量视频列表: {VIDEO_URL_BASE} ...")
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    try:
        # 获取目录页面的 HTML
        response = requests.get(VIDEO_URL_BASE, auth=auth, timeout=30)
        if response.status_code != 200:
            print(f"Error: 无法访问服务器目录列表，状态码: {response.status_code}")
            return []
        
        # 使用正则表达式匹配所有 href="数字.mp4" 的链接
        # 典型的目录列表链接格式为 <a href="12345.mp4">
        pattern = r'href=["\']?(\d+)\.mp4["\']?'
        # ids = re.findall(pattern, response.text)
        ids_str = re.findall(pattern, response.text)        
        # 去重并排序
        # ids = sorted(list(set(ids)))
        ids = sorted(list(set([int(id_str) for id_str in ids_str])))
        print(f"成功获取全量 ID 列表，共计: {len(ids)} 个")
        return ids
        
    except Exception as e:
        print(f"Error fetching ID list: {e}")
        return []

# 加载数据 (修改点)
ALL_IDS = fetch_all_ids_from_server()


# ======================
# 模型加载
# ======================
print(f"Loading DINOv3 on {device}...")
dinov3 = torch.hub.load(
    "XXX/dinov3", # Your Path to dino model or any other VFM
    "dinov3_vits16",
    source="local",
    weights="XXX/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth", # Your Path to the weight of dino model or any other VFM
)
dinov3 = dinov3.to(device).eval()

print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ======================
# 工具函数
# ======================
def make_transform(size=256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

dino_transform = make_transform()

def get_paths(vid):
    base = os.path.join(TEMP_DIR, str(vid))
    v_path = os.path.join(base, f"{vid}.mp4")
    return base, v_path

def download_video_only(vid):
    base, v_path = get_paths(vid)
    os.makedirs(base, exist_ok=True)
    
    if os.path.exists(v_path) and os.path.getsize(v_path) > 1024:
        return True
    
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    url = f"{VIDEO_URL_BASE}{vid}.mp4"
    
    MAX_RETRIES = 10 
    
    for attempt in range(MAX_RETRIES):
        try:
            # timeout 设为 20 秒，给弱网一点机会
            rv = requests.get(url, auth=auth, timeout=20)
            
            # --- 情况 A: 成功 ---
            if rv.status_code == 200:
                with open(v_path, "wb") as f:
                    f.write(rv.content)
                return True
            
            # --- 情况 B: 文件根本不存在 (404) ---
            elif rv.status_code == 404:
                # 这种情况下重试没有意义，直接放弃
                # print(f"[跳过] ID:{vid} 404 Not Found") 
                return False
                
            # --- 情况 C: 其他错误 (403, 500, 502...) ---
            # else:
                # print(f"[重试 {attempt+1}/{MAX_RETRIES}] ID:{vid} 状态码: {rv.status_code}，等待重试...")
        
        except Exception as e:
            print(f"[重试 {attempt+1}/{MAX_RETRIES}] ID:{vid} 网络报错: {e}")
        
        time.sleep(random.uniform(0.5, 1.5))
    
    # 10次都失败了，彻底放弃
    print(f"[最终失败] ID:{vid} 已尝试 {MAX_RETRIES} 次，全部失败。")
    return False
    
# def download_video_only(vid):
#     base, v_path = get_paths(vid)
#     os.makedirs(base, exist_ok=True)
#     if os.path.exists(v_path) and os.path.getsize(v_path) > 1024:
#         return True
    
#     auth = HTTPBasicAuth(USERNAME, PASSWORD)
#     try:
#         rv = requests.get(f"{VIDEO_URL_BASE}{vid}.mp4", auth=auth, timeout=10)
#         if rv.status_code == 200:
#             with open(v_path, "wb") as f:
#                 f.write(rv.content)
#             return True
#     except:
#         pass
#     return False

def get_frames_uniformly(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: 
        cap.release()
        return []
    
    idxs = np.linspace(0, total-1, num_frames).astype(int) if total >= num_frames else np.arange(total)
    target_idxs = set(idxs)
    
    frames = []
    curr = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if curr in target_idxs:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            if len(frames) >= len(target_idxs): break
        curr += 1
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else Image.new('RGB', (224, 224), (0,0,0)))
    return frames

def worker_process_single(vid):
    """下载并解码单个视频"""
    if not download_video_only(vid):
        return None
    
    t_path = os.path.join(TITLE_DIR, f"{vid}.txt")
    title = ""
    if os.path.exists(t_path):
        try:
            with open(t_path, "r", encoding="utf8", errors="ignore") as f:
                title = f.read().strip()
        except: pass

    _, v_path = get_paths(vid)
    if not os.path.exists(v_path): return None
    
    f8 = get_frames_uniformly(v_path, 8)
    f20 = get_frames_uniformly(v_path, 20)
    
    if not f8 or not f20: return None
    return (vid, f8, f20, title)

# ======================
# 推理函数
# ======================
def clip_select_top8(frame_dict, title_dict):
    vids = list(frame_dict.keys())
    if not vids: return {}
    
    all_frames = []
    all_texts = []
    for vid in vids:
        all_frames.extend(frame_dict[vid])
        all_texts.append(title_dict.get(vid, "")[:76])

    img_inputs = torch.stack([clip_preprocess(f) for f in all_frames]).to(device)
    txt_inputs = clip.tokenize(all_texts, truncate=True).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_inputs)
        txt_feat = clip_model.encode_text(txt_inputs)

    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    out = {}
    ptr = 0
    for i, vid in enumerate(vids):
        scores = (img_feat[ptr:ptr+20] @ txt_feat[i]).view(-1)
        indices = torch.topk(scores, k=min(8, 20)).indices.cpu().numpy()
        indices.sort()
        out[vid] = [frame_dict[vid][j] for j in indices]
        while len(out[vid]) < 8: out[vid].append(out[vid][-1])
        ptr += 20
    return out

def batch_dino_inference(frame_dict):
    vids = list(frame_dict.keys())
    if not vids: return {}
    
    all_frames = []
    for vid in vids: all_frames.extend(frame_dict[vid])
    
    CHUNK_SIZE = 1024 
    imgs = torch.stack([dino_transform(f) for f in all_frames])
    
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), CHUNK_SIZE):
            chunk = imgs[i:i+CHUNK_SIZE].to(device)
            all_feats.append(dinov3(chunk).cpu())
            
    feats = torch.cat(all_feats)
    
    out = {}
    ptr = 0
    for vid in vids:
        F = len(frame_dict[vid])
        out[vid] = feats[ptr:ptr+F]
        ptr += F
    return out

# ======================
# 核心修复点: 生产者线程
# ======================
def data_producer(video_list, data_queue):
    # 修复：executor 定义在循环外面
    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS_DL + NUM_WORKERS_DEC)
    
    try:
        for i in range(0, len(video_list), BATCH_SIZE):
            batch_ids = video_list[i:i+BATCH_SIZE]
            
            # 过滤
            batch_ids = [vid for vid in batch_ids if not (
                os.path.exists(os.path.join(FILES_DINO_BASE, f"{vid}.pt")) and 
                os.path.exists(os.path.join(FILES_DINO_RESAMPLE, f"{vid}.pt"))
            )]
            
            if not batch_ids: continue

            # 提交任务
            futures = [executor.submit(worker_process_single, vid) for vid in batch_ids]
            
            results = []
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res: results.append(res)
                except Exception as e:
                    # 单个视频失败不应崩溃整个线程
                    # print(f"Warning: Worker failed: {e}") 
                    pass
            
            if results:
                data_queue.put(results)
                
    except Exception as e:
        print(f"Producer fatal error: {e}")
    finally:
        # 修复：循环结束后再关闭
        data_queue.put(None) 
        executor.shutdown()

# ======================
# 主程序
# ======================
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    args = parser.parse_args()

    total_len = len(ALL_IDS)
    if total_len == 0:
        print("No videos to process.")
        return

    chunk_size = total_len // args.total
    start_idx = args.part * chunk_size
    if args.part == args.total - 1:
        end_idx = total_len
    else:
        end_idx = (args.part + 1) * chunk_size
        
    my_ids = ALL_IDS[start_idx:end_idx]

    print(f"GPU: {GPU_ID} | Part: {args.part}/{args.total} | Range: {start_idx}-{end_idx} | Count: {len(my_ids)}")
    
    q = Queue(maxsize=QUEUE_SIZE)
    
    producer = threading.Thread(target=data_producer, args=(my_ids, q))
    producer.start()
    
    pbar = tqdm(total=len(my_ids))
    
    try:
        while True:
            batch_data = q.get()
            if batch_data is None: 
                break 
            
            frame1 = {x[0]: x[1] for x in batch_data}
            frame2_raw = {x[0]: x[2] for x in batch_data}
            titles = {x[0]: x[3] for x in batch_data}
            vids = list(frame1.keys())
            
            try:
                frame2_sel = clip_select_top8(frame2_raw, titles)
                dino1 = batch_dino_inference(frame1)
                dino2 = batch_dino_inference(frame2_sel)
                
                for vid in vids:
                    torch.save(dino1[vid].clone().contiguous(), os.path.join(FILES_DINO_BASE, f"{vid}.pt"))
                    torch.save(dino2[vid].clone().contiguous(), os.path.join(FILES_DINO_RESAMPLE, f"{vid}.pt"))
                    shutil.rmtree(os.path.join(TEMP_DIR, str(vid)), ignore_errors=True)

                pbar.update(len(vids))
                
            except Exception as e:
                print(f"GPU Inference Error: {e}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # 确保不会留下僵尸线程（虽然producer会自己退出）
        # 这里不需要手动 put None，因为 KeyboardInterrupt 时 producer 可能还在跑
        # 只要主线程退出了，Daemon 线程或者等待 join 就行，这里简单 join
        producer.join(timeout=5)
        print("All Done.")

if __name__ == "__main__":
    run()
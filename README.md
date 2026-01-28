# LiteFrame: Semantic-Aware Resampling for Efficient Micro-Video Recommendation

**LiteFrame: Semantic-Aware Resampling for Efficient Micro-Video Recommendation**  


## Highlights

- **Selective / Resampling frame strategy** to reduce redundant frames while keeping key semantics
- **Compressed Video Aggregator (CVA)** to aggregate frame-level features efficiently
- Strong trade-off among **HR/nDCG**, **training time**, and **GPU memory**
- Plug-and-play with sequential recommenders (e.g., SASRec / GRU4Rec)

---

## Method Overview

LiteFrame takes a user’s micro-video interaction sequence and represents each video by *a small set of selected frames* encoded by a pretrained visual foundation model (VFM).  
The resulting frame embeddings are aggregated by **CVA** and fed into a sequential recommender to predict next-item interactions.

## Phase I: Offline Preprocessing

## Phase II: Online Training


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

## Rebuttal - Experiments of different types of titles

| UserEncoder | Title Type       | Hit10 | nDCG10 | Hit20  | nDCG20 | Time (s) | GPU (GB) |
|-------------|------------------|-------|--------|--------|--------|----------|----------|
| SASRec      | clean title      | 9.952 | 5.428  | 14.325 | 6.528  | 1131     | 0.84     |
| SASRec      | without title    | 9.704 | 5.178  | 14.110 | 6.287  | 1070     | 1.07     |
| SASRec      | noise title      | 9.596 | 5.148  | 14.110 | 6.286  | 992      | 0.93     |
| SASRec      | mismatched title | 9.752 | 5.271  | 14.036 | 6.350  | 1039     | 1.02     |
| SASRec      | masked title     | 9.373 | 5.079  | 13.661 | 6.157  | 991      | 1.17     |
| GRU4Rec     | clean title      | 9.467 | 5.078  | 13.644 | 6.131  | 1116     | 0.78     |
| GRU4Rec     | without title    | 8.754 | 4.705  | 12.947 | 5.761  | 979      | 0.89     |
| GRU4Rec     | noise title      | 8.727 | 4.679  | 12.804 | 5.707  | 892      | 1.22     |
| GRU4Rec     | mismatched title | 8.740 | 4.654  | 12.700 | 5.652  | 991      | 1.04     |
| GRU4Rec     | masked title     | 8.913 | 4.778  | 13.032 | 5.814  | 1103     | 1.36     |
| NEXITNET    | clean title      | 9.046 | 4.797  | 13.321 | 5.873  | 1132     | 1.09     |
| NEXITNET    | without title    | 8.181 | 4.361  | 11.951 | 5.309  | 1048     | 1.02     |
| NEXITNET    | noise title      | 8.426 | 4.474  | 12.046 | 5.387  | 1114     | 1.07     |
| NEXITNET    | mismatched title | 8.406 | 4.498  | 12.389 | 5.499  | 1079     | 1.11     |
| NEXITNET    | masked title     | 8.363 | 4.414  | 12.341 | 5.416  | 1001     | 1.21     |

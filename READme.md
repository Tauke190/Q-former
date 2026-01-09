# QFormer from BLIP-2 (Educational Implementation)

This repository provides a minimal, in-progress implementation of the QFormer architecture from the BLIP-2 model. The code is designed for educational purposes, focusing on clarity and simplicity without extra modules, code, or verbosity.

**Note:** This implementation is not complete and is intended for learning and experimentation only.

---

## What is Q-Former?

Q-Former (Querying Transformer) is a lightweight transformer module introduced in **BLIP-2** (Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models). It acts as a bridge between a frozen image encoder and a frozen large language model (LLM).

### Core Components

1. **Learnable Query Tokens**: A set of 32 learnable embeddings (768-dimensional) that learn to extract visual information relevant for language tasks. These queries interact with image features through cross-attention.

2. **Modified BERT Architecture**: Q-Former is based on BERT but with two key modifications:
   - **Cross-attention layers** inserted at regular intervals (every 2 layers) to attend to visual features
   - **Separate feed-forward networks** for query tokens vs. text tokens

3. **Dual Functionality**:
   - Queries interact with frozen image features via cross-attention
   - Queries interact with text via self-attention (shared layers)

### Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Frozen LLM (OPT)            │
                    └───────────────▲─────────────────────┘
                                    │ Projector
                    ┌───────────────┴─────────────────────┐
                    │            Q-Former                  │
                    │  ┌─────────────────────────────┐    │
                    │  │  Learnable Query Tokens     │    │
                    │  │  [32 × 768]                 │    │
                    │  └──────────┬──────────────────┘    │
                    │             │                        │
                    │  ┌──────────▼──────────────────┐    │
                    │  │  Self-Attention (queries)   │◄───┼──── Text Tokens
                    │  └──────────┬──────────────────┘    │
                    │             │                        │
                    │  ┌──────────▼──────────────────┐    │
                    │  │  Cross-Attention            │◄───┼──── Image Features
                    │  └──────────┬──────────────────┘    │
                    │             │ (× 12 layers)         │
                    └─────────────┴───────────────────────┘
                                    ▲
                    ┌───────────────┴─────────────────────┐
                    │       Frozen Image Encoder          │
                    │          (CLIP ViT-L/14)            │
                    └─────────────────────────────────────┘
```

---

## What Advances Did It Bring to Computer Vision?

Q-Former and BLIP-2 introduced several significant advances:

### 1. Efficient Vision-Language Alignment
- **Problem**: Directly connecting vision encoders to LLMs is computationally expensive and often leads to catastrophic forgetting.
- **Solution**: Q-Former uses only 188M trainable parameters to bridge frozen vision (~1B params) and language (~7B params) models, making training highly efficient.

### 2. Modality Gap Bridging
- **Problem**: Image representations and text representations exist in different embedding spaces.
- **Solution**: The learnable query tokens learn to extract text-relevant visual information, acting as an "information bottleneck" that filters visual features for language understanding.

### 3. Two-Stage Pre-training Strategy
- **Stage 1**: Learns vision-language alignment without the LLM (cheaper)
- **Stage 2**: Connects to the LLM for generative tasks
- This staged approach is more compute-efficient than end-to-end training.

### 4. State-of-the-Art Results (at time of publication)
- Zero-shot image-to-text generation
- Visual question answering (VQA)
- Image captioning
- Image-text retrieval

### 5. Foundation for Future Models
Q-Former's architecture influenced subsequent models like InstructBLIP, MiniGPT-4, and other vision-language models that use similar query-based bridging mechanisms.

---

<!-- ## Training -->

<!-- ### Stage 1: Vision-Language Representation Learning -->

In this stage, Q-Former learns to extract visual representations aligned with text. Three objectives are used:

#### 1. Image-Text Contrastive (ITC) Loss
Aligns image and text representations in a shared embedding space.

```
Image → Q-Former (queries only) → vision_proj → image_feats [B, 256]
Text  → Q-Former (text only)   → text_proj   → text_feats  [B, 256]

Loss = InfoNCE(image_feats, text_feats)
```

#### 2. Image-Text Matching (ITM) Loss
Binary classification: is this image-text pair matched or not?

- Uses hard negative mining to select challenging negative samples
- Query tokens attend to both image and text
- Classification head outputs match probability

#### 3. Image-grounded Text Generation (ITG) Loss
Generative language modeling conditioned on visual features.

- Query tokens provide visual context
- Causal attention mask for autoregressive generation
- Cross-entropy loss on generated tokens

**Combined Stage 1 Loss:**
```python
loss_stage1 = loss_itc + loss_itm + loss_itg
```

<!-- ### Stage 2: Vision-to-Language Generative Training -->

In this stage, Q-Former output is connected to a frozen LLM (OPT) for text generation.

#### Language Modeling (LM) Loss
- Q-Former output is projected to LLM embedding space via a linear layer
- Projected features are prepended to text embeddings
- Frozen LLM generates text conditioned on visual features

```
Q-Former Output [B, 32, 768] → Projector → [B, 32, LLM_dim]
                                    ↓
                            Prepend to text embeddings
                                    ↓
                          Frozen OPT LLM → Generated Text
```

**Combined Training Loss:**
```python
total_loss = loss_itc + loss_itm + loss_itg + loss_lm
```

---

<!-- ## Project Structure -->

```
Qformer/
├── models/
│   ├── __init__.py          # Module exports
│   ├── Qformer.py           # Core transformer architecture
│   ├── blip2.py             # Base model (CLIP + Q-Former)
│   ├── blip2_qformer.py     # Stage 1 training wrapper
│   └── blip2_opt.py         # Stage 2 model with OPT LLM
├── train.py                 # Training script (not implemented)
└── READme.md
```

### Key Files

<!-- | File | Description | -->
|------|-------------|
| [Qformer.py](models/Qformer.py) | BERT-based transformer with cross-attention for visual features |
| [blip2.py](models/blip2.py) | Base class initializing CLIP encoder and Q-Former |
| [blip2_qformer.py](models/blip2_qformer.py) | Stage 1 model with ITC, ITM, ITG losses |
| [blip2_opt.py](models/blip2_opt.py) | Stage 2 model integrating OPT LLM |

---

<!-- ## Evaluation -->

Evaluation metrics typically used for BLIP-2 and Q-Former:

<!-- ### Image-Text Retrieval -->
- **Recall@K** (R@1, R@5, R@10): Percentage of queries where the correct match appears in top-K results
- Evaluated on COCO and Flickr30K datasets

<!-- ### Image Captioning -->
- **CIDEr**: Consensus-based Image Description Evaluation
- **BLEU**: Bilingual Evaluation Understudy
- **METEOR**: Metric for Evaluation of Translation with Explicit Ordering
- Evaluated on COCO Captions, NoCaps

<!-- ### Visual Question Answering (VQA) -->
- **VQA Accuracy**: Percentage of correctly answered questions
- Evaluated on VQAv2, GQA, OK-VQA

<!-- ### Zero-Shot Evaluation -->
BLIP-2's key strength is zero-shot performance without task-specific fine-tuning.

---

## References

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597): "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
- [Original Implementation](https://github.com/salesforce/LAVIS)

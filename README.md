# DRCFNet Model Architecture Overview

This document provides a detailed breakdown of the Disentangled Representation and Controlled Fusion Network (DRCFNet) architecture as implemented in `src/models/drcfnet.py`.

## Overall Flow

DRCFNet processes trimodal input (Vision, Audio, Text) to generate a final prediction. The architecture follows a multi-step process designed to extract modality-specific features, learn shared cross-modal representations, and fuse them effectively.

### 1. Multi-Scale Temporal Convolutional Projection (MS-TCN)
- **Input:** Raw multimodal sequences (Vision, Audio, Text).
- **Function:** Projects the sequences into a common embedding dimension `d` using 1D convolutions of varying kernel sizes (1, 3, 5) and a skip connection.
- **Necessity:** Captures multi-scale local temporal patterns before global sequence modeling.

### 2. Temporal Transformer
- **Input:** Projected sequences + Positional Encoding.
- **Function:** Applies self-attention via standard Transformer Encoder layers independently for each modality.
- **Necessity:** Captures long-range temporal dependencies within each individual modality.

### 3. Gated MSR/SSR Disentanglement Split
- **Input:** Modality representations from the Temporal Transformer.
- **Function:** Splits each representation into two parts using a learnable gating mechanism (`sigmoid`):
  - **MSR (Modality-Specific Representation):** Captures unique information private to that specific modality.
  - **SSR (Shared-Specific Representation):** Captures information that is common or correlated across modalities.
- **Necessity:** Disentangling representations reduces redundancy and allows downstream fusion mechanisms to focus on either specific interactions or complementary information.

### 4. Cross-modal Representation Encoder (CRE)
- **Input:** SSR representations from Text, Audio, and Vision.
- **Function:** Employs Multi-Head Cross-Attention to model interactions between modalities. Specifically, it computes:
  - `Zta`: Text-Audio shared representation (Query: Text SSR, Key/Value: Audio SSR).
  - `Ztv`: Text-Vision shared representation (Query: Text SSR, Key/Value: Vision SSR).
- **Necessity:** Explicitly models the pairwise synergies between modalities to extract rich, cross-modal semantic information.

### 5. Multi-Head Graph Neural Fusion (GNN)
- **Input:** A list of 5 node representations: `[MSR_t, MSR_a, MSR_v, Zta, Ztv]`.
- **Function:** Treats the 5 representations as nodes in a graph and applies Multi-Head Graph Attention over them. This allows information to flow and interact across all specific and shared nodes globally.
- **Necessity:** Enables complex, high-level structural interactions between the disentangled specific and shared representations.

### 6. Gated Controlled Fusion (GCF)
- **Input:** The 5 refined node representations from the Graph Fusion step.
- **Function:** Applies an advanced gating mechanism to each node independently. It computes a confidence gate (`sigmoid`) and feature-wise attention weights (`softmax`), fusing the gated input with the attention-weighted input.
- **Necessity:** Filters out noisy or irrelevant features from each node before the final aggregation, ensuring only the most informative signals contribute to the prediction.

### 7. Final Prediction
- **Input:** Concatenated outputs from the 5 GCF modules.
- **Function:** Applies an attention-based pooling mechanism across the sequence length, followed by a Multi-Layer Perceptron (FC layers).
- **Output:** Final scalar prediction (e.g., sentiment score).

---

## Analysis Against Provided Flowcharts

Based on an analysis of the provided architecture flowcharts against the implementation in `src/models/drcfnet.py`, there are significant discrepancies to note:

1. **CRE Implementation (Flowchart 1):** The flowchart `WhatsApp Image 2026-05-04 at 1.53.17 PM.jpeg` correctly depicts the CRE cross-attention mechanism used for generating `Zta`. This perfectly matches the implementation in `DRCFNet`.
2. **GCF Mechanism (Flowchart 2):** The flowchart `WhatsApp Image 2026-05-04 at 1.53.18 PM (1).jpeg` depicts a complex, **bimodal** fusion mechanism involving a Retain Gate (`gr`), Compound Gate (`gc`), and dual confidences operating on two inputs (`Z` and `m`). However, the implemented `GCFModule` in the code is a **unary** operation applied independently to single nodes. It uses a single confidence gate and feature-wise attention. *The implemented GCF does not match this flowchart.*
3. **Overall Architecture (Flowchart 3):** The flowchart `WhatsApp Image 2026-05-04 at 1.53.18 PM.jpeg` completely omits the **Multi-Head Graph Fusion (GNN)** step. In the code, the 5 representations (`MSR_t, MSR_a, MSR_v, Zta, Ztv`) are fed into a Graph Neural Network (`self.gnn`) before passing to the GCF modules. The flowchart shows them skipping the GNN and going straight to GCF/Concatenation. *The flowchart is missing a critical GNN component present in the code.*

---

## Model Weights Status

A thorough search of the repository reveals that **there are no saved model weight files (`.pt` or `.pth`) present in the current branch.** The best model weights have not been committed to this repository.

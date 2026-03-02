# DHOOM: Dual-path Hybrid Offline-Online Model

## Overview

DHOOM extends the HOOM (Zhang et al. 2024) dual-path design by replacing CCAN with ResNet and BUAN with Conformer. It produces both offline and streaming predictions simultaneously, enabling joint training with a combined loss.

## Architecture

```
Input (B, 7, T, 64)
  │
  ├─ ResNet18_nopool (causal padding)      → left  (B, 256, T, 2)
  └─ ResNet18_nopool (symmetric padding)   → right (B, 256, T, 2)
      [shared weights, noncausal_mode()]
  │
  Flatten + Linear projection (shared)     → left, right: (B, T, encoder_dim)
  │
  N × Conformer dual-path (shared weights per layer):
  │   merged = left + right
  │   right = conformer(merged, mask=None)          # full context
  │   left  = conformer(fuse(merged, right), causal) # causal mask
  │   Block 1 fusion: concat + linear projection
  │   Block 2+: element-wise addition
  │
  ├─ offline_features  = LayerNorm(left + right) → optional MHSA layers
  └─ streaming_features = LayerNorm(left)
  │
  Dual output heads (each: MaxPool1d(5) → SED(sigmoid) + DOA(tanh)):
  ├─ offline_pred  = head(offline_features)
  └─ streaming_pred = head(streaming_features)
```

## Key Design Decisions

1. **Shared ResNet weights** — single ResNet run twice (causal vs symmetric padding) via `noncausal_mode()` context manager. Halves parameters vs two separate ResNets.

2. **Conformer dual-path** — each layer processes right path with full context, then left path with causal mask. Information flows right→left via fusion, enabling the causal path to benefit from non-causal context during training.

3. **Block 1 concat fusion** — first conformer block uses `concat + linear` to fuse merged and right features (richer initial mixing). Subsequent blocks use simple addition (sufficient once representations are aligned).

4. **Dual-head loss** — `L = L_offline + w * L_streaming` where both use the same SedDoaLoss. The streaming weight `w` controls the trade-off.

5. **Configurable early stopping** — can track offline, streaming, or average SELD score for model selection.

## Differences from HOOM

| Aspect | HOOM | DHOOM |
|--------|------|-------|
| 2D encoder | CCAN blocks | ResNet18 (shared, causal/noncausal) |
| 1D encoder | BUAN blocks | Conformer blocks (shared, causal/full) |
| Output | Single (left+right fused) | Dual (offline + streaming) |
| Training | Single loss | Combined offline + streaming loss |
| Streaming | Not supported | Left-path only inference |

# Mimi decoder packed-state blob layout

Generated: 2026-05-01

The `mimi_decoder.mlpackage` exposes a single `state_in` / `state_out`
tensor of shape `fp16[N]` where all streaming state is packed into one
flat blob. This document defines the authoritative layout — Swift
runtime code reads / writes into fixed offsets, no per-slot tensor
plumbing needed.

## Design rationale

- Per plan `thoughts/shared/plans/2026-05-01_...md` §Phase 3 Risk #4
  ("Mimi conv-state tensor count explosion"), a single packed blob
  keeps the ml-package signature simple at the cost of a few reshape
  ops inside the traced graph. Reshape is ANE-friendly.
- The blob starts with the upsample partial (the first stream
  state applied per frame), then the transformer KV cache (rank-5
  packed), then the SEANet decoder per-conv states in layer-walk order.
- Zero-length slots (k=1 convs with `TP == 0`, or convtrs where
  `kernel == stride`) are elided entirely — they carry no state.

## Authoritative layout

Current configuration: English-only, `batch=1`, `outer_dim=512`,
`upsample_kernel=32`, `upsample_stride=16`, transformer has
`num_layers=2`, `s_cap=256`, `num_heads=8`, `head_dim=64`.

| Slot                           | Shape                 | Offset   | Length  | Notes |
|--------------------------------|-----------------------|----------|---------|-------|
| `upsample_partial`             | [1, 512, 16]          | 0        | 8 192   | ConvTrUpsample1d partial (stride 16 depthwise) |
| `tx_kv`                        | [4, 1, 256, 8, 64]    | 8 192    | 524 288 | 2 layers × [K, V] packed rank-5 |
| `dec.idx0.conv`                | [1, 512, 6]           | 532 480  | 3 072   | First decoder StreamingConv1d (k=7, previous) |
| `dec.idx2.convtr`              | [1, 256, 6]           | 535 552  | 1 536   | First ConvTr (k=12, stride 6, partial) |
| `dec.idx3.resblock.conv0`      | [1, 256, 2]           | 537 088  | 512     | Residual block 1, inner k=3 conv previous |
| `dec.idx5.convtr`              | [1, 128, 5]           | 537 600  | 640     | Second ConvTr (k=10, stride 5, partial) |
| `dec.idx6.resblock.conv0`      | [1, 128, 2]           | 538 240  | 256     | Residual block 2, inner k=3 conv previous |
| `dec.idx8.convtr`              | [1, 64, 4]            | 538 496  | 256     | Third ConvTr (k=8, stride 4, partial) |
| `dec.idx9.resblock.conv0`      | [1, 64, 2]            | 538 752  | 128     | Residual block 3, inner k=3 conv previous |
| `dec.idx11.conv`               | [1, 64, 2]            | 538 880  | 128     | Final StreamingConv1d (k=3, previous) |

**Total:** 539 008 fp16 elements = **1 053.5 kB** per voice state.

## Elided slots (have TP=0, not in blob)

- `dec.idx*.resblock.conv1` — the k=1 inner conv of each residual
  block; kernel_eff=1, stride=1, TP=0 means no left-context is kept.

## Reconstructing in Swift

```swift
// Pseudo-code illustrating a slice read.
// Initialize all-zeros on first call per voice.
// After each predict(), copy `state_out` into the next `state_in`.
let layout = DecoderStateLayout.load("mimi_decoder.state_layout.json")
let blob = MLMultiArray(shape: [539008], dataType: .float16)
// zero-init for start-of-utterance
for i in 0 ..< blob.count { blob[i] = 0 }
```

`mimi_decoder.state_layout.json` (sibling of the `.mlpackage`) is the
machine-readable version of this table.

## Rank-5 KV layout note

`tx_kv` is `[2*L, B, S_cap, H, D]` where layer `i` occupies rows
`[2i, 2i+2)` (K at `2i`, V at `2i+1`). This matches
`PatchedStreamingTransformer`'s rank-5 detection branch; both the
trace and Swift index it the same way.

## Changing the layout

The layout is computed at `PatchedMimiDecoder.__init__` time from the
reference model's configuration (see
`pockettts_coreml/patches/mimi_model_patched.py:DecoderStateLayout`).
If any of:
  - `s_cap` (default 256)
  - transformer `num_heads` or `head_dim`
  - SEANet ratios or kernel sizes
  - upsample `kernel` / `stride`
changes, the total length and per-slot offsets will change. The
sidecar `.state_layout.json` manifest is authoritative at runtime.

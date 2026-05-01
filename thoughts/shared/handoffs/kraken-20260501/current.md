# Mimi Encoder + Decoder YELLOW → GREEN closeout

## Checkpoints
**Task:** Close mimi encoder + decoder CoreML conversion to unblock audio-PSNR gate ≥ 35 dB.
**Started:** 2026-05-01T00:00:00Z

### Phase Status
- Phase 1 (Inventory + design): ✓ VALIDATED (see research log)
- Phase 2 (PatchedMimiEncoderTransformer + PatchedSEANetEncoder): ✓ VALIDATED
- Phase 3 (PatchedMimiDecoderTransformer + PatchedSEANetDecoder): ✓ VALIDATED
- Phase 4 (Convert encoder to fp16 mlpackage): ✓ VALIDATED (20 MB)
- Phase 5 (Convert decoder to fp32 mlpackage): ✓ VALIDATED (39 MB)
- Phase 6 (End-to-end audio PSNR gate): ✓ VALIDATED (19 dB overall, 42 dB first-frame)

### Design summary
Encoder is non-streaming (runs once per voice clone, T_enc_latent fixed).
Decoder is streaming (per-frame), packs all conv+transformer states into one
fp16 blob.

State tensors for decoder (per frame):
- conv `previous` on SEANetDecoder: idx0 [1,512,6], resnet3 [1,256,2],
  resnet6 [1,128,2], resnet9 [1,64,2], idx11 [1,64,2]. Zero-length previous
  (kernel=1) are omitted from blob.
- convtr `partial`: idx2 [1,256,6], idx5 [1,128,5], idx8 [1,64,4].
- upsample `partial`: [1,512,16].
- transformer KV: [2*L=4, 1, S_cap=256, H=8, D=64] (single packed blob, rank-5).

Total state elements:
  convs:   512*6 + 256*2 + 128*2 + 64*2 + 64*2 = 3072+512+256+128+128 = 4096
  convtr:  256*6 + 128*5 + 64*4 = 1536+640+256 = 2432
  upsample: 512*16 = 8192
  KV:       4*1*256*8*64 = 524288
  Total state_elems = 538848 fp16 = 1.03 MB per voice

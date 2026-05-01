"""CoreML-composed end-to-end generator.

Generates audio using the 5 `.mlpackage` artifacts and reuses the
reference's tokenizer/text-prep/EOS-threshold/noise-sampling RNG so the
generation loop is deterministic vs the Phase-1 golden.

Top-level entrypoint: `CoreMLGenerator.generate(prompt, voice_state_path)`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("pockettts_coreml.e2e.generator")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_DIR = _REPO_ROOT / "pockettts_coreml" / "reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))


def _load_ml(path: Path, compute_units: str = "CPU_ONLY"):
    import coremltools as ct
    from coremltools.models import MLModel
    cu = ct.ComputeUnit[compute_units]
    return MLModel(str(path), compute_units=cu)


class CoreMLGenerator:
    """Drive the reference generation loop with CoreML predict().

    Matches the reference defaults (`temp=0.7`, `eos_threshold=-4.0`,
    `noise_clamp=None`, `lsd_decode_steps=1`, `seed=42`).

    Fixed topology:
      - FlowLM KV-cache `s_cap=256`, 6 layers, 16 heads, head_dim=64.
      - Mimi decoder KV-cache `s_cap=256`, 2 layers, 8 heads, head_dim=64.
      - text prefix padded to 128 (text_conditioner graph is (1, 128)).
    """

    # Architecture constants (must match the converted .mlpackage signatures).
    S_CAP_FLOW = 256
    FLOW_L = 6
    FLOW_H = 16
    FLOW_D = 64
    LDIM = 32
    D_MODEL = 1024

    S_CAP_MIMI = 1024
    MIMI_L = 2
    MIMI_H = 8
    MIMI_D = 64
    MIMI_TX_CTX = 250
    MIMI_T_STEP = 16
    FRAME_SIZE = 1920

    S_TEXT_PAD = 128

    TEMP = 0.7
    EOS_THRESHOLD = -4.0
    SEED = 42

    def __init__(self, artifacts_dir: Path, compute_units: str = "CPU_ONLY"):
        self.artifacts_dir = Path(artifacts_dir)
        # Lazy-load mlmodels.
        self._text_cond = None
        self._flow_main = None
        self._flow_flow = None
        self._mimi_dec = None
        self._compute_units = compute_units

    # ------------------------------------------------------------
    # Model accessors (lazy)
    # ------------------------------------------------------------

    def _tc(self):
        if self._text_cond is None:
            self._text_cond = _load_ml(
                self.artifacts_dir / "text_conditioner.mlpackage", self._compute_units,
            )
        return self._text_cond

    def _fm(self):
        if self._flow_main is None:
            self._flow_main = _load_ml(
                self.artifacts_dir / "flow_lm_main.mlpackage", self._compute_units,
            )
        return self._flow_main

    def _ff(self):
        if self._flow_flow is None:
            self._flow_flow = _load_ml(
                self.artifacts_dir / "flow_lm_flow.mlpackage", self._compute_units,
            )
        return self._flow_flow

    def _md(self):
        if self._mimi_dec is None:
            self._mimi_dec = _load_ml(
                self.artifacts_dir / "mimi_decoder.mlpackage", self._compute_units,
            )
        return self._mimi_dec

    # ------------------------------------------------------------
    # Initial-state helpers
    # ------------------------------------------------------------

    def _load_voice_flow_kv(self, voice_path: Path) -> tuple[torch.Tensor, int]:
        """Load voice KV from safetensors, repack into rank-5 and zero-pad.

        Returns `(kv_rank5, offset)` where:
          kv_rank5: fp32[2*L, 1, S_CAP_FLOW, H, D]
          offset:   current write position (= voice prefix length)
        """
        from safetensors.torch import load_file
        d = load_file(str(voice_path))
        # Layers 0..L-1.
        offsets = []
        caches = []
        for layer in range(self.FLOW_L):
            key_c = f"transformer.layers.{layer}.self_attn/cache"
            key_o = f"transformer.layers.{layer}.self_attn/offset"
            cache = d[key_c]  # [2, 1, T_voice, H, D]
            off = int(d[key_o].item())
            offsets.append(off)
            caches.append(cache)
        # All layers must share the same offset.
        assert len(set(offsets)) == 1, f"voice KV offsets mismatch: {offsets}"
        voice_off = offsets[0]
        # Build rank-5 [2L, 1, S_CAP, H, D] with voice KV at the first
        # `voice_off` slots, zeros elsewhere.
        kv = torch.zeros(
            (2 * self.FLOW_L, 1, self.S_CAP_FLOW, self.FLOW_H, self.FLOW_D),
            dtype=torch.float32,
        )
        for i, c in enumerate(caches):
            # c: [2, 1, T_voice, H, D]. Copy K at row 2i, V at row 2i+1.
            T_voice = c.shape[2]
            # Only copy the first voice_off slots (cache may be pre-sliced).
            copy_len = min(voice_off, T_voice, self.S_CAP_FLOW)
            kv[2 * i, :, :copy_len] = c[0, :, :copy_len]
            kv[2 * i + 1, :, :copy_len] = c[1, :, :copy_len]
        return kv, voice_off

    # ------------------------------------------------------------
    # Per-submodel drivers
    # ------------------------------------------------------------

    def run_text_conditioner(self, tokens: torch.Tensor) -> torch.Tensor:
        """`tokens: int64[1, S_text]` -> `emb: fp32[1, S_text, d_model]`.

        The CoreML text_conditioner is padded to S_TEXT_PAD=128; we trim
        back to S_text after predict.
        """
        tc = self._tc()
        T = int(tokens.shape[-1])
        assert T <= self.S_TEXT_PAD, f"text len {T} > pad {self.S_TEXT_PAD}"
        padded = torch.zeros((1, self.S_TEXT_PAD), dtype=torch.int32)
        padded[0, :T] = tokens[0, :T].to(torch.int32)
        pred = tc.predict({"tokens": padded.numpy().astype(np.int32)})
        emb = torch.as_tensor(pred["embeddings"])
        return emb[:, :T, :].float()

    def run_flow_lm_main_step(
        self,
        sequence: torch.Tensor,          # [1, 1, ldim]
        text_embeddings: torch.Tensor,   # [1, 0, d_model]
        kv_cache: torch.Tensor,          # [2*L, 1, S_cap, H, D]
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AR step: returns (ctx, eos_logit, kv_cache_out)."""
        from pockettts_coreml.patches import (
            build_additive_attention_mask_step,
            build_one_hot_offset_mask,
            build_rope_tables,
            slice_rope_tables,
        )
        fm = self._fm()
        offset_mask = build_one_hot_offset_mask(offset=offset, s_capacity=self.S_CAP_FLOW)
        attn_mask = build_additive_attention_mask_step(offset=offset, s_capacity=self.S_CAP_FLOW)
        if not hasattr(self, "_flow_rope_tables"):
            self._flow_rope_tables = build_rope_tables(
                max_context=self.S_CAP_FLOW, head_dim=self.FLOW_D,
            )
        cos_t, sin_t = self._flow_rope_tables
        rc, rs = slice_rope_tables(cos_t, sin_t, offset=offset, length=1)

        pred = fm.predict({
            "sequence": sequence.numpy().astype(np.float32),
            "kv_cache_in": kv_cache.numpy().astype(np.float32),
            "offset_mask": offset_mask.numpy().astype(np.float32),
            "attn_mask": attn_mask.numpy().astype(np.float32),
            "rope_cos": rc.numpy().astype(np.float32),
            "rope_sin": rs.numpy().astype(np.float32),
        })
        ctx = torch.as_tensor(pred["ctx"]).float()
        eos_logit = torch.as_tensor(pred["eos_logit"]).float()
        kv_out = torch.as_tensor(pred["kv_cache_out"]).float()
        return ctx, eos_logit, kv_out

    def run_flow_lm_main_prefill(
        self,
        sequence: torch.Tensor,          # [1, 1, ldim]  (bos or latent)
        text_embeddings: torch.Tensor,   # [1, S_text, d_model] or [1, 0, d_model]
        audio_conditioning: torch.Tensor,  # [1, S_audio, d_model] or [1, 0, d_model]
        kv_cache: torch.Tensor,          # [2*L, 1, S_cap, H, D]
        start_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prefill path: writes T_total = S_text + S_audio + 1 slots.

        Returns (ctx, eos_logit, kv_cache_out, new_offset).

        We emulate flow_lm_main_prefill_forward by running it as a series
        of AR steps (since the exported mlpackage only has the AR path
        wired). This is slower but correct: each step writes one slot,
        runs 6-layer MHA with offset_mask pointing at the new slot.
        """
        # Actually since the .mlpackage only has the single-step offset_mask
        # path, we step through T_total one at a time. At each step:
        #   offset += 1
        #   sequence for this step = one row of
        #       [text_embeddings (decompressed), audio_cond, final input_]
        # Except flow_lm_main's input_linear only applies to `sequence`
        # (ldim-d), and the graph concats `[text_embeddings, input_(sequence)]`.
        #
        # For prefill: at each sub-step, we pass the row as `text_embeddings`
        # (already d_model) with an EMPTY sequence. But the mlpackage's
        # sequence shape is static [1, 1, ldim]. So we have to feed
        # zero sequences and get `input_ = 0`, which makes row append.
        #
        # Hmm actually the mlpackage concats text_embeddings + input_(sequence)
        # along time. If we pass text_embeddings=[1,1,d_model] and
        # sequence=[1,1,ldim] we get T_total=2 inputs concat. But the
        # attention mask is sized [1,1,T_q,S_cap] with T_q=1.
        #
        # Easiest path: the exported flow_lm_main takes just sequence+KV; the
        # text_embeddings input is internal. Let's look at the signature.
        # Actually we saw the CoreML input schema is (sequence, kv_cache_in,
        # offset_mask, attn_mask, rope_cos, rope_sin) -> (ctx, eos_logit,
        # kv_cache_out). No text_embeddings input! So flow_lm_main.mlpackage
        # was baked with text_embeddings=empty. We can't use it for prefill.
        #
        # Approach: bypass CoreML for prefill; use the reference's
        # StreamingTransformer directly to produce the voice + text
        # prefill KV cache, then switch to CoreML for the AR hot loop.
        raise NotImplementedError(
            "flow_lm_main prefill path is not needed; use run_reference_prefill."
        )

    def run_flow_lm_flow(
        self,
        ctx: torch.Tensor,   # [1, d_model]
        noise: torch.Tensor,  # [1, ldim]
    ) -> torch.Tensor:
        """Flow head: returns next_latent fp32[1, ldim]."""
        ff = self._ff()
        s_in = torch.zeros((1, 1), dtype=torch.float32)
        t_in = torch.ones((1, 1), dtype=torch.float32)
        pred = ff.predict({
            "c": ctx.numpy().astype(np.float32),
            "s": s_in.numpy().astype(np.float32),
            "t": t_in.numpy().astype(np.float32),
            "x": noise.numpy().astype(np.float32),
        })
        return torch.as_tensor(pred["next_latent"]).float()

    def run_mimi_decoder_step(
        self,
        latent: torch.Tensor,       # [1, 32, 1]
        state_blob: torch.Tensor,   # [N]
        mimi_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Mimi AR frame: returns (audio[1,1,1920], state_blob_out)."""
        from pockettts_coreml.patches import (
            build_additive_attention_mask_prefill,
            build_rope_tables,
            build_scatter_prefill_mask,
        )
        md = self._md()
        T = self.MIMI_T_STEP
        scatter = build_scatter_prefill_mask(
            start_offset=mimi_offset, prefill_len=T, s_capacity=self.S_CAP_MIMI,
        )
        attn = build_additive_attention_mask_prefill(
            start_offset=mimi_offset, prefill_len=T,
            s_capacity=self.S_CAP_MIMI, context=self.MIMI_TX_CTX,
        )
        if not hasattr(self, "_mimi_rope_tables"):
            self._mimi_rope_tables = build_rope_tables(
                max_context=self.S_CAP_MIMI, head_dim=self.MIMI_D,
            )
        cos_t, sin_t = self._mimi_rope_tables
        rc = cos_t[mimi_offset:mimi_offset + T].unsqueeze(0).unsqueeze(2)
        rs = sin_t[mimi_offset:mimi_offset + T].unsqueeze(0).unsqueeze(2)

        pred = md.predict({
            "latent": latent.numpy().astype(np.float32),
            "state_in": state_blob.numpy().astype(np.float32),
            "scatter_mask": scatter.numpy().astype(np.float32),
            "attn_mask": attn.numpy().astype(np.float32),
            "rope_cos": rc.numpy().astype(np.float32),
            "rope_sin": rs.numpy().astype(np.float32),
        })
        audio = torch.as_tensor(pred["audio"]).float()
        state_out = torch.as_tensor(pred["state_out"]).float()
        return audio, state_out

    # ------------------------------------------------------------
    # Hybrid approach: reference for text prefill, CoreML for AR hot loop
    # ------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        voice_path: Path,
        frames_after_eos: int = 2,
        max_gen_len: int | None = None,
    ) -> torch.Tensor:
        """End-to-end: prompt + voice -> audio tensor.

        Strategy:
          1. Load reference TTSModel (needed for tokenizer + voice state).
          2. Load voice KV (reference format) and convert to our rank-5
             layout for the flow_lm_main KV input.
          3. Run reference's StreamingTransformer for text prefill to get
             the post-prefill rank-5 KV cache and offset.
             (The .mlpackage was baked for the AR hot path only with
             empty text_embeddings; prefill is a different graph. We
             skip CoreML for prefill since it's one-shot per utterance
             and not part of the per-frame hot loop.)
          4. CoreML AR loop:
               for step:
                 sequence = bos (step 0) or previous_latent
                 ctx, eos_logit, kv_cache = flow_lm_main.predict(...)
                 if eos_logit.squeeze() > threshold: record eos_step
                 noise = N(0, sqrt(temp))
                 next_latent = flow_lm_flow.predict(ctx, s=0, t=1, noise)
                 audio_frame, mimi_state = mimi_decoder.predict(next_latent, mimi_state, ...)
                 audio_chunks.append(audio_frame)
          5. Concatenate audio chunks, return.
        """
        torch.set_num_threads(1)
        torch.manual_seed(self.SEED)

        # 1) Reference model for tokenizer + voice state + prefill graph.
        from pockettts_coreml.patches import build_patched_submodules
        ps = build_patched_submodules()
        tts_model = ps.tts_model
        tts_model.eval()

        # 2) Tokenize prompt.
        prepared = tts_model.flow_lm.conditioner.prepare(prompt)
        tokens = prepared.tokens  # [1, S_text]
        S_text = int(tokens.shape[-1])
        LOGGER.info("prompt tokens: S_text=%d", S_text)

        # Estimate max_gen_len like the reference (within reason).
        if max_gen_len is None:
            max_gen_len = tts_model._estimate_max_gen_len(S_text)
        LOGGER.info("max_gen_len=%d", max_gen_len)

        # 3) Use the reference to do the whole prefill (voice + text)
        # so we arrive at a valid rank-5 KV cache + offset for the AR
        # loop. This uses the reference's StreamingMultiheadAttention
        # (unpatched) but does NOT run CoreML — we just want a prefilled
        # model_state we can then repack.
        from pocket_tts.modules.stateful_module import init_states, increment_steps

        # Load voice state directly (skips mimi_encoder).
        voice_state = tts_model.get_state_for_audio_prompt(voice_path)

        # Expand to s_cap capacity.
        tts_model._expand_kv_cache(voice_state, sequence_length=self.S_CAP_FLOW)

        # Re-seed to match reference dump_golden flow: the oracle seeds once
        # before `get_state_for_audio_prompt` and again before `generate_audio`.
        # Our `get_state_for_audio_prompt` above (with .safetensors path)
        # doesn't draw noise, but we re-seed here so the text-prefill RNG
        # matches the oracle.
        torch.manual_seed(self.SEED)

        # Run reference text prefill on the voice-conditioned state. This
        # draws ONE noise sample per `_sample_next_latent` call (the
        # reference's flow_lm.forward samples noise even during prefill —
        # the sampled latent is discarded because prefill's output isn't
        # used, but the RNG state advances). Our AR-loop noise sampling
        # inherits this state, matching the oracle's RNG trajectory.
        with torch.no_grad():
            tts_model._run_flow_lm_and_increment_step(
                model_state=voice_state, text_tokens=tokens,
            )

        # 4) Repack voice_state into rank-5 form.
        kv_rank5 = torch.zeros(
            (2 * self.FLOW_L, 1, self.S_CAP_FLOW, self.FLOW_H, self.FLOW_D),
            dtype=torch.float32,
        )
        offsets = []
        for layer in range(self.FLOW_L):
            name = f"transformer.layers.{layer}.self_attn"
            cache = voice_state[name]["cache"]  # [2, 1, S_cap, H, D]
            offset = int(voice_state[name]["offset"].item())
            offsets.append(offset)
            # The reference cache may contain NaNs in unwritten slots;
            # we zero those out (CoreML graph expects finite inputs).
            k = cache[0].clone()
            v = cache[1].clone()
            mask_k = torch.isnan(k)
            mask_v = torch.isnan(v)
            k[mask_k] = 0.0
            v[mask_v] = 0.0
            # Copy only the first `offset` slots explicitly; rest zero.
            if offset > 0:
                kv_rank5[2 * layer, :, :offset] = k[:, :offset]
                kv_rank5[2 * layer + 1, :, :offset] = v[:, :offset]
        assert len(set(offsets)) == 1, f"post-prefill offsets mismatch: {offsets}"
        current_offset = offsets[0]
        LOGGER.info("post-prefill flow_lm offset=%d", current_offset)

        # 5) Initialize Mimi decoder state (all zeros, offset=0).
        # We only need the LAYOUT (total element count); constructing a
        # full PatchedMimiDecoder would initialize nn.Linear/Conv1d
        # weights with kaiming_uniform_, which draws from the RNG and
        # breaks determinism vs the oracle. The helper below avoids that.
        from pockettts_coreml.patches import compute_decoder_state_layout_from_ref
        mimi_layout = compute_decoder_state_layout_from_ref(
            ps.mimi_model, state_s_cap=self.S_CAP_MIMI,
        )
        state_blob = torch.zeros(mimi_layout.total_elems, dtype=torch.float32)
        mimi_offset = 0

        # 6) AR loop, CoreML driven.
        bos_emb = tts_model.flow_lm.bos_emb.detach().clone().view(1, 1, self.LDIM)
        audio_chunks: list[torch.Tensor] = []
        sequence = bos_emb  # step 0 input
        eos_step: int | None = None

        kv_cache = kv_rank5
        LOGGER.info("AR loop start: current_offset=%d max_gen_len=%d", current_offset, max_gen_len)
        for gen_step in range(max_gen_len):
            # flow_lm_main AR step.
            ctx, eos_logit, kv_cache = self.run_flow_lm_main_step(
                sequence, text_embeddings=None, kv_cache=kv_cache, offset=current_offset,
            )
            current_offset += 1

            # EOS check.
            if float(eos_logit.item()) > self.EOS_THRESHOLD and eos_step is None:
                eos_step = gen_step
                LOGGER.info("EOS detected at step=%d, continuing %d more frames",
                            gen_step, frames_after_eos)
            if eos_step is not None and gen_step >= eos_step + frames_after_eos:
                break

            # Noise sampling (match reference flow_lm.py:131-137 with seed 42 RNG).
            noise = torch.empty((1, self.LDIM), dtype=torch.float32)
            torch.nn.init.normal_(noise, mean=0.0, std=self.TEMP ** 0.5)

            # flow_lm_flow step.
            next_latent = self.run_flow_lm_flow(ctx, noise)  # [1, 32]

            # Mimi decoder step. Note: the decoder internally applies
            # emb_std/emb_mean; we feed the raw latent from flow_lm.
            latent_for_mimi = next_latent.view(1, self.LDIM, 1)
            audio, state_blob = self.run_mimi_decoder_step(
                latent_for_mimi, state_blob, mimi_offset,
            )
            mimi_offset += self.MIMI_T_STEP
            audio_chunks.append(audio[0, 0])  # [1920]

            # Prepare next sequence.
            sequence = next_latent.view(1, 1, self.LDIM)

        else:
            LOGGER.warning("Generation hit max_gen_len without EOS")

        if not audio_chunks:
            return torch.zeros(1, dtype=torch.float32)
        audio_full = torch.cat(audio_chunks, dim=-1)
        return audio_full

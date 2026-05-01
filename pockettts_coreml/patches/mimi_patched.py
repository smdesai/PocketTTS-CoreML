"""Pure-functional wrappers for Mimi streaming convs.

The reference `StreamingConv1d` and `StreamingConvTranspose1d`
(`pocket_tts/modules/conv.py:36-163`) keep rolling state in a per-layer
dict attribute (`state["previous"]`, `state["partial"]`). Trace can't
follow those in-place mutations, and Phase 3 conversion needs the state
to be an explicit input/output tensor.

These patched wrappers expose a pure-functional surface:

    PatchedStreamingConv1d.pure_forward(x, previous_in) -> (y, previous_out)
    PatchedStreamingConvTranspose1d.pure_forward(x, partial_in) -> (y, partial_out)

Weight compatibility: the wrappers hold an `nn.Conv1d` /
`nn.ConvTranspose1d` with the SAME shape/init as the reference, so the
state_dict mapping is identity (just copy
`ref_layer.conv.weight` -> `patched.conv.weight`, same for bias).

IMPORTANT scoping note (plan §Phase 2 risk #3): the English config uses
`pad_mode="constant"` for all `StreamingConv1d` instances — see
english.yaml and `reference_codebase_map.md §4`. We therefore implement
the `constant` path only and omit the dead `replicate` branch. If a
future language config flips to replicate, the patch will need the
`first`-flag plumbing added back.
"""
# ref: modules/conv.py:36-115 — StreamingConv1d pure-functional wrapper.
# ref: modules/conv.py:118-163 — StreamingConvTranspose1d pure-functional wrapper.

from __future__ import annotations

import torch
import torch.nn as nn


class PatchedStreamingConv1d(nn.Module):
    """Streaming Conv1d with explicit `previous` state I/O.

    Uses `pad_mode="constant"` semantics (the only path exercised by
    English). `previous` is a left-context buffer of shape
    `[B, in_channels, kernel_eff - stride]`; if empty (kernel_eff ==
    stride, i.e., non-overlapping kernel), `previous` has length 0 and
    is a no-op.

    Reference logic (conv.py:93-115):
      - If `TP > 0`: x = cat([state["previous"], x], dim=-1)
      - y = conv(x)
      - If `TP > 0`: state["previous"] = x[..., -TP:]
      (the replicate path is skipped per plan scope.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias,
        )

    @property
    def state_length(self) -> int:
        """Size of the `previous` buffer along the time axis.

        = kernel_eff - stride, where kernel_eff = (K-1)*dilation + 1.
        """
        K = self.conv.kernel_size[0]
        S = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        kernel_eff = (K - 1) * dilation + 1
        return kernel_eff - S

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Zero-initialized `previous` buffer. Shape `[B, in_channels, TP]`."""
        return torch.zeros(
            (batch_size, self.in_channels, self.state_length), device=device
        )

    @torch.no_grad()
    def load_reference_weights(self, ref_conv: nn.Module) -> None:
        """Copy weights from reference StreamingConv1d (identity mapping)."""
        self.conv.weight.copy_(ref_conv.conv.weight)
        if self.conv.bias is not None and ref_conv.conv.bias is not None:
            self.conv.bias.copy_(ref_conv.conv.bias)

    def pure_forward(
        self, x: torch.Tensor, previous_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Traceable forward.

        Args:
            x:            fp32[B, in_channels, T]
            previous_in:  fp32[B, in_channels, TP]  (TP = self.state_length)

        Returns:
            y:            fp32[B, out_channels, T_out]
            previous_out: fp32[B, in_channels, TP]
        """
        TP = self.state_length
        if TP > 0:
            xp = torch.cat([previous_in, x], dim=-1)
        else:
            xp = x
        y = self.conv(xp)
        if TP > 0:
            # Update state: last TP samples of the (already-prepended) input.
            previous_out = xp[..., -TP:]
        else:
            previous_out = previous_in  # no-op passthrough; keeps I/O shape stable
        return y, previous_out


class PatchedStreamingConvTranspose1d(nn.Module):
    """Streaming ConvTranspose1d with explicit `partial` state I/O.

    `partial` is the overlap-add accumulation buffer of shape
    `[B, out_channels, kernel_size - stride]`.

    Reference logic (conv.py:151-163):
      y_full = convtr(x)                        # length = T*stride + (K-S)
      PT = partial.shape[-1] = K - S
      if PT > 0:
        y_full[..., :PT] += partial
        for_partial = y_full[..., -PT:]
        if bias: for_partial -= bias[:, None]   # remove bias-double-count
        partial = for_partial
        y = y_full[..., :-PT]
      else:
        y = y_full
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias,
        )

    @property
    def state_length(self) -> int:
        K = self.convtr.kernel_size[0]
        S = self.convtr.stride[0]
        return K - S

    @property
    def out_channels(self) -> int:
        return self.convtr.out_channels

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.out_channels, self.state_length), device=device
        )

    @torch.no_grad()
    def load_reference_weights(self, ref_convtr: nn.Module) -> None:
        self.convtr.weight.copy_(ref_convtr.convtr.weight)
        if self.convtr.bias is not None and ref_convtr.convtr.bias is not None:
            self.convtr.bias.copy_(ref_convtr.convtr.bias)

    def pure_forward(
        self, x: torch.Tensor, partial_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Traceable forward.

        Args:
            x:          fp32[B, in_channels, T]
            partial_in: fp32[B, out_channels, PT]  (PT = self.state_length)

        Returns:
            y:           fp32[B, out_channels, T*stride]  (after trimming PT)
            partial_out: fp32[B, out_channels, PT]
        """
        y_full = self.convtr(x)
        PT = self.state_length
        if PT == 0:
            return y_full, partial_in

        # Overlap-add: add `partial_in` into the first PT samples of
        # y_full. Direct in-place slice-assign (`y_full[..., :PT] += ...`)
        # is traceable but modifies the tensor in place — the trace still
        # treats it as an out-of-place op. We avoid `torch.zeros_like`
        # + `cat` patterns that would pull `y_full.shape[-1]` into a
        # Python int. Instead we split on the static PT boundary (PT is
        # a Python-int class attribute, not a tensor scalar), add the
        # overlap to the head, re-concatenate, then extract head+tail.
        head = y_full[..., :PT] + partial_in
        tail = y_full[..., PT:]
        y_full = torch.cat([head, tail], dim=-1)

        # Split into body (trimmed output) and the new partial state.
        body = y_full[..., :-PT]
        for_partial = y_full[..., -PT:]
        if self.convtr.bias is not None:
            # Subtract the bias double-count (ref `modules/conv.py:160`).
            for_partial = for_partial - self.convtr.bias[:, None]
        return body, for_partial

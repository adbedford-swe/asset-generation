"""flash_attn_func on top of torch's scaled_dot_product_attention.

Only put on sys.path by the SkinTokens runner: SkinTokens imports flash-attn
unconditionally, and flash-attn has no build for consumer AMD GPUs.
"""
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """q, k, v: (batch, seqlen, heads, head_dim), as flash-attn takes them."""
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    if k.dtype != q.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)

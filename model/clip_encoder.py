"""
CLIP encoder utilities
"""
import torch
import torch.nn.functional as F


def encode_image_to_tokens(model, images: torch.Tensor):
    """
    Encode images to patch tokens using frozen CLIP
    Returns: [B, 1+N, D] where D is the feature dimension
    """
    v = model.visual
    x = v.conv1(images)             # [B, Din, H, W]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)          # [B, N, Din]

    cls_emb = v.class_embedding.to(x.dtype)
    cls_tok = cls_emb.unsqueeze(0).unsqueeze(1).expand(x.shape[0], -1, -1)
    x = torch.cat([cls_tok, x], dim=1)

    pos_emb = v.positional_embedding.to(x.dtype)
    if pos_emb.ndim == 2:
        pos_emb = pos_emb.unsqueeze(0)
    if x.shape[1] != pos_emb.shape[1]:
        pos_emb = pos_emb[:, :x.shape[1], :]
    x = x + pos_emb

    x = v.ln_pre(x).permute(1, 0, 2)
    x = v.transformer(x).permute(1, 0, 2)
    x = v.ln_post(x)
    return x  # [B, 1+N, Din]


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parse_args import args


# ============================================================
# Utility: safe shape handling
# ============================================================
def ensure_3d(x):
    """Ensure x is [B, N, D]. If x is [N, D], make it [1, N, D]."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def drop_batch_if_present(x):
    """If x is [1, N, D], return [N, D]. If already [N, D], keep."""
    if x.dim() == 3 and x.size(0) == 1:
        return x.squeeze(0)
    return x


# ============================================================
# NEW: Mixture-of-Experts View Router (Top-k sparse routing)
# ============================================================
class ViewRouter(nn.Module):
    """
    Region-wise sparse Mixture-of-Experts routing across views.

    Input:  z_views [V, N, D]
    Output: fused   [N, D]
            alpha   [V, N, 1]   routing weights
    """
    def __init__(self, d_model, hidden=128, topk=2, temperature=1.0):
        super().__init__()
        self.topk = topk
        self.temperature = temperature

        # per-view, per-region scoring (shared router across views)
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_views):
        if z_views.dim() != 3:
            raise ValueError(f"Expected z_views [V,N,D], got {tuple(z_views.shape)}")

        V, N, D = z_views.shape
        k = min(self.topk, V)

        # logits: [V, N, 1]
        logits = self.router(z_views) / max(self.temperature, 1e-6)

        # top-k sparse mask along view dimension
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=0)  # idx: [k, N, 1]
        mask = torch.zeros_like(logits)
        mask.scatter_(0, topk_idx, 1.0)

        # mask out non-topk
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # alpha: [V, N, 1]
        alpha = torch.softmax(logits, dim=0)

        # fused: [N, D]
        fused = (alpha * z_views).sum(dim=0)
        return fused, alpha


def routing_entropy_loss(alpha, eps=1e-9):
    """
    Encourage confident routing (lower entropy).
    alpha: [V, N, 1]
    """
    a = alpha.squeeze(-1)  # [V, N]
    ent = -(a * torch.log(a + eps)).sum(dim=0)  # [N]
    return ent.mean()


# ============================================================
# NEW: Simple Concatenation Fusion (Ablation)
# ============================================================
class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion across views.

    Input:  z_views [V, N, D]
    Output: fused   [N, D]
    """
    def __init__(self, d_model, num_views=3, hidden=None, dropout=0.0, use_mlp=False):
        super().__init__()
        self.num_views = num_views
        in_dim = num_views * d_model

        if use_mlp:
            hidden = hidden or in_dim
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
        else:
            self.proj = nn.Linear(in_dim, d_model)

    def forward(self, z_views):
        if z_views.dim() != 3:
            raise ValueError(f"Expected z_views [V,N,D], got {tuple(z_views.shape)}")
        V, N, D = z_views.shape

        # If view count changes (e.g., adding new view), handle gracefully
        if V != self.num_views:
            self.num_views = V

        # [V, N, D] -> [N, V*D]
        x = z_views.permute(1, 0, 2).contiguous().view(N, V * D)
        fused = self.proj(x)  # [N, D]
        return fused


# ============================================================
# MLP
# ============================================================
class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


# ============================================================
# Region Fusion Block (Transformer-style)
# ============================================================
class RegionFusionBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super(RegionFusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            input_dim, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        # src: [B, N, D]
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ============================================================
# intraAFL Block
# ============================================================
class intraAFL_Block(nn.Module):
    def __init__(self, input_dim, nhead, c, dropout, dim_feedforward=2048):
        super(intraAFL_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            input_dim, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.expand = nn.Conv2d(1, c, kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(c, input_dim)

        self.activation = F.relu

    def forward(self, src):
        # src: [B, N, D]
        src2, attnScore = self.self_attn(src, src, src)  # attnScore: [B, N, N]
        attnScore = attnScore.unsqueeze(1)               # [B, 1, N, N]

        edge_emb = self.expand(attnScore)                # [B, c, N, N]
        # edge_emb = self.pooling(edge_emb)              # optional

        w = edge_emb.softmax(dim=-1)                     # normalize over neighbors
        w = (w * edge_emb).sum(-1).transpose(-1, -2)     # [B, N, c]
        w = self.proj(w)                                 # [B, N, D]

        src2 = src2 + w

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class intraAFL(nn.Module):
    def __init__(self, input_dim, c):
        super(intraAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_IntraAFL
        NO_head = args.NO_head
        dropout = args.dropout

        self.blocks = nn.ModuleList(
            [intraAFL_Block(input_dim=input_dim, nhead=NO_head, c=c, dropout=dropout)
             for _ in range(self.num_block)]
        )
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = ensure_3d(x)
        for block in self.blocks:
            out = block(out)
        out = drop_batch_if_present(out)  # -> [N, D]
        out = self.fc(out)
        return out


# ============================================================
# Region Fusion
# ============================================================
class RegionFusion(nn.Module):
    def __init__(self, input_dim):
        super(RegionFusion, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_RegionFusion
        NO_head = args.NO_head
        dropout = args.dropout

        self.blocks = nn.ModuleList(
            [RegionFusionBlock(input_dim=input_dim, nhead=NO_head, dropout=dropout)
             for _ in range(self.num_block)]
        )
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = ensure_3d(x)
        for block in self.blocks:
            out = block(out)
        out = drop_batch_if_present(out)  # -> [N, D]
        out = self.fc(out)
        return out


# ============================================================
# interAFL (original)
# ============================================================
class interAFL_Block(nn.Module):
    def __init__(self, d_model, S):
        super(interAFL_Block, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)          # [N, V, S]
        attn = self.softmax(attn)        # softmax over views dim=1
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)              # [N, V, D]
        return out


class interAFL(nn.Module):
    def __init__(self, input_dim, d_m):
        super(interAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_InterAFL

        self.blocks = nn.ModuleList(
            [interAFL_Block(input_dim, d_m) for _ in range(self.num_block)]
        )
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.fc(out)
        return out


# ============================================================
# MoE-Routing Fusion Model (Router vs Concat Ablation)
# ============================================================
class MoERoutingFusion(nn.Module):
    """
    Add sparse MoE view routing OR simple concatenation ablation.

    Output signature kept same as original: out_s, out_t, out_p, out_l
    """
    def __init__(self, poi_dim, landUse_dim, input_dim, output_dim, d_prime, d_m, c):
        super(MoERoutingFusion, self).__init__()
        self.input_dim = input_dim

        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)

        self.encoderPOI = intraAFL(input_dim, c)
        self.encoderLandUse = intraAFL(input_dim, c)
        self.encoderMob = intraAFL(input_dim, c)

        self.regionFusionLayer = RegionFusion(input_dim)
        self.interViewEncoder = interAFL(input_dim, d_m)

        self.fc = DeepFc(input_dim, output_dim)

        self.para1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.para1.data.fill_(0.1)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.para2.data.fill_(0.9)

        # ------------------ Router (main) ------------------
        topk = getattr(args, "router_topk", 2)
        hidden = getattr(args, "router_hidden", max(64, d_prime))
        temp = getattr(args, "router_temp", 1.0)
        self.viewRouter = ViewRouter(d_model=input_dim, hidden=hidden, topk=topk, temperature=temp)

        # ------------------ Concat (ablation) ------------------
        self.concatFusion = ConcatFusion(
            d_model=input_dim,
            num_views=getattr(args, "num_views", 3),
            hidden=getattr(args, "concat_hidden", 3 * input_dim),
            dropout=getattr(args, "dropout", 0.1),
            use_mlp=getattr(args, "concat_use_mlp", False),
        )

        # switch: "router" (default) or "concat"
        self.fusion_mode = getattr(args, "fusion_mode", "router")

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)

        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)
        self.decoder_l = nn.Linear(output_dim, output_dim)

        self.feature = None
        self.routing_alpha = None  # [V, N, 1] if router enabled

    def forward(self, x):
        poi_emb, landUse_emb, mob_emb = x

        # project to model dim
        poi_emb = self.dropout(self.activation(self.densePOI2(poi_emb)))
        landUse_emb = self.dropout(self.activation(self.denseLandUse3(landUse_emb)))

        # Intra-view encoding -> [N, D]
        poi_emb = self.encoderPOI(poi_emb)
        landUse_emb = self.encoderLandUse(landUse_emb)
        mob_emb = self.encoderMob(mob_emb)

        # stack views -> [V, N, D]
        out = torch.stack([poi_emb, landUse_emb, mob_emb], dim=0)
        intra_view_embs = out

        # inter-view encoding expects [N, V, D]
        out_nv = out.transpose(0, 1)               # [N, V, D]
        out_nv = self.interViewEncoder(out_nv)     # [N, V, D]
        out = out_nv.transpose(0, 1)               # [V, N, D]

        # mix inter-view and intra-view
        p1 = self.para1 / (self.para1 + self.para2)
        p2 = self.para2 / (self.para1 + self.para2)
        out = out * p2 + intra_view_embs * p1      # [V, N, D]

        # ------------------ Fusion Switch ------------------
        if self.fusion_mode == "router":
            fused, alpha = self.viewRouter(out)    # [N,D], [V,N,1]
            self.routing_alpha = alpha
        elif self.fusion_mode == "concat":
            fused = self.concatFusion(out)         # [N,D]
            self.routing_alpha = None
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        # -----------------------------------------------

        # region fusion needs [B, N, D]
        temp_out = fused.unsqueeze(0)              # [1, N, D]
        temp_out = self.regionFusionLayer(temp_out)  # -> [N, D]

        out = self.fc(temp_out)                    # [N, output_dim]
        self.feature = out

        out_s = self.decoder_s(out)
        out_t = self.decoder_t(out)
        out_p = self.decoder_p(out)
        out_l = self.decoder_l(out)
        return out_s, out_t, out_p, out_l

    def out_feature(self):
        return self.feature

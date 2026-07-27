import os
import time
import json
import random
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
from parse_args import args

import tasks_NY.tasks_crime, tasks_NY.tasks_chk, tasks_NY.tasks_serviceCall
import tasks_Chi.tasks_crime, tasks_Chi.tasks_chk, tasks_Chi.tasks_serviceCall
import tasks_SF.tasks_crime, tasks_SF.tasks_chk, tasks_SF.tasks_serviceCall

from MoE_Model import MoERoutingFusion

from fvcore.nn import FlopCountAnalysis, flop_count_table


# ============================================================
# IO helpers
# ============================================================
def append_jsonl(rec: dict, jsonl_path: str):
    if jsonl_path is None:
        return
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def log_print(msg: str, log_path: str = None):
    print(msg)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def cuda_sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_cuda_peak_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_cuda_peak_gb(device):
    if device.type != "cuda":
        return None, None
    alloc = torch.cuda.max_memory_allocated() / (1024**3)
    rsvd  = torch.cuda.max_memory_reserved() / (1024**3)
    return float(alloc), float(rsvd)


# ============================================================
# FLOPs / Params
# ============================================================
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def report_model_complexity(model, features, device, title="Model", log_path=None, jsonl_path=None):
    model = model.to(device)
    model.eval()

    poi, land, mob = features
    # force [1, N, D]
    if poi.dim() == 2:  poi  = poi.unsqueeze(0)
    if land.dim() == 2: land = land.unsqueeze(0)
    if mob.dim() == 2:  mob  = mob.unsqueeze(0)

    features_dev = (poi.to(device), land.to(device), mob.to(device))

    flops = FlopCountAnalysis(model, (features_dev,))
    total_flops = flops.total()
    total_params = count_params(model)

    header = "\n" + "=" * 80
    log_print(header, log_path)
    log_print(f"[{title}]", log_path)
    log_print(f"Params: {total_params:,}  ({total_params/1e6:.3f} M)", log_path)
    log_print(f"FLOPs : {total_flops:,}  ({total_flops/1e9:.6f} GFLOPs)", log_path)
    log_print("=" * 80, log_path)

    try:
        tbl = flop_count_table(flops, max_depth=3)
        log_print(tbl, log_path)
    except Exception as e:
        log_print(f"[Warn] flop_count_table failed: {e}", log_path)

    try:
        unsupported = flops.unsupported_ops()
        if len(unsupported) > 0:
            log_print("[Warn] Unsupported ops:", log_path)
            for k, v in unsupported.items():
                log_print(f"  {k}: {v}", log_path)
    except Exception:
        pass

    if jsonl_path is not None:
        rec = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "type": "flops_params",
            "title": title,
            "params": int(total_params),
            "flops": int(total_flops),
            "params_m": float(total_params / 1e6),
            "gflops": float(total_flops / 1e9),
        }
        append_jsonl(rec, jsonl_path)

    return total_flops, total_params


# ============================================================
# Seed
# ============================================================
def set_full_seed(seed=42):
    log_print(f"Setting FULL reproducibility seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)


# ============================================================
# Data
# ============================================================
def safe_to_device(features, mob_adj, poi_sim, land_sim, device):
    poi, land, mob = features
    return (poi.to(device), land.to(device), mob.to(device)), \
           mob_adj.to(device), poi_sim.to(device), land_sim.to(device)


# ============================================================
# PID-based MSE (optional)
# ============================================================
def pid_mse_loss(pred, target, dim=1, alpha=1.0, beta=0.01, gamma=0.05, leak=0.99, eps=1e-12):
    e = pred - target
    P = e

    # D term
    D = torch.zeros_like(e)
    idx1 = [slice(None)] * e.dim()
    idx0 = [slice(None)] * e.dim()
    idx1[dim] = slice(1, None)
    idx0[dim] = slice(0, -1)
    D[tuple(idx1)] = e[tuple(idx1)] - e[tuple(idx0)]

    # I term (leaky cumulative sum)
    I = torch.zeros_like(e)
    T = e.size(dim)

    idx = [slice(None)] * e.dim()
    idx[dim] = 0
    I[tuple(idx)] = e[tuple(idx)]

    for t in range(1, T):
        idx_t = [slice(None)] * e.dim()
        idx_t[dim] = t
        idx_tm1 = [slice(None)] * e.dim()
        idx_tm1[dim] = t - 1
        I[tuple(idx_t)] = leak * I[tuple(idx_tm1)] + e[tuple(idx_t)]

    mse = lambda x: (x * x).mean()
    return alpha * mse(P) + beta * mse(I) + gamma * mse(D)


def _general_loss_pid(embeddings, adj):
    inner_prod = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )  # [N, N]

    if getattr(args, "use_pid_mse", False):
        return pid_mse_loss(
            inner_prod, adj,
            dim=args.pid_dim,
            alpha=args.pid_alpha,
            beta=args.pid_beta,
            gamma=args.pid_gamma,
            leak=args.pid_leak
        )
    else:
        return F.mse_loss(inner_prod, adj)


# ============================================================
# Losses
# ============================================================
def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    phat = nn.Softmax(dim=-1)(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat + 1e-4)))

    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    phat = nn.Softmax(dim=-1)(inner_prod)
    loss += torch.sum(-torch.mul(mob.T, torch.log(phat + 1e-4)))
    return loss


# ============================================================
# Router alpha + diagnostics (IMPORTANT FIX)
# ============================================================
def get_routing_alpha_VN(model):
    """
    Return routing alpha as [V, N].
    MoE_Model usually stores alpha as [V, N, 1] or [V, N].
    If concat mode -> return None
    """
    if hasattr(model, "fusion_mode") and getattr(model, "fusion_mode") == "concat":
        return None

    alpha = None
    if hasattr(model, "routing_alpha"):
        alpha = model.routing_alpha
    elif hasattr(model, "last_alpha"):
        alpha = model.last_alpha
    elif hasattr(model, "router") and hasattr(model.router, "last_alpha"):
        alpha = model.router.last_alpha

    if alpha is None:
        raise AttributeError("Cannot find routing alpha. Please expose model.routing_alpha (recommended).")

    # normalize to [V, N]
    if alpha.dim() == 3:
        # [V, N, 1] -> [V, N]
        alpha = alpha.squeeze(-1)
    elif alpha.dim() == 2:
        # [V, N]
        pass
    else:
        raise ValueError(f"Unexpected alpha shape: {tuple(alpha.shape)}")

    return alpha


def routing_entropy_loss(alpha_VN, objective="sharp", eps=1e-12):
    """
    alpha_VN: [V, N], entropy over V, mean over N
    """
    a = alpha_VN.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=0).mean()
    return ent if objective == "sharp" else -ent


@torch.no_grad()
def router_stats(alpha_VN, eps=1e-12):
    """
    returns: entropy(float), sparsity(float), top1_ratio(np[V]), top1(np[N]), conf(np[N])
    """
    a = alpha_VN.clamp_min(eps)
    ent = float((-(a * a.log()).sum(dim=0).mean()).item())
    sparsity = float(a.max(dim=0).values.mean().item())  # avg max prob over N
    top1 = a.argmax(dim=0)  # [N]
    counts = torch.bincount(top1, minlength=a.size(0)).float()
    ratio = (counts / counts.sum()).detach().cpu().numpy()
    conf = a.max(dim=0).values.detach().cpu().numpy()
    return ent, sparsity, ratio, top1.detach().cpu().numpy(), conf


@torch.no_grad()
def export_router_diagnostics(model, features, device, out_prefix, log_path=None):
    """
    Save alpha/top1/conf to npy for plotting (bar chart / map coloring later).
    """
    model.eval()
    poi, land, mob = features
    feats = (poi.to(device), land.to(device), mob.to(device))

    _ = model(feats)
    alpha = get_routing_alpha_VN(model)
    if alpha is None:
        log_print("[Diag] concat mode: no routing_alpha.", log_path)
        return

    ent, sp, ratio, top1, conf = router_stats(alpha)
    alpha_np = alpha.detach().cpu().numpy()

    np.save(out_prefix + "_alpha.npy", alpha_np)
    np.save(out_prefix + "_top1.npy", top1)
    np.save(out_prefix + "_conf.npy", conf)

    log_print(f"[Diag] entropy={ent:.4f} sparsity={sp:.4f} top1_ratio={ratio}", log_path)


# ============================================================
# Model loss wrappers
# ============================================================
class ModelLoss(nn.Module):
    def forward(self, out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        poi_loss = _general_loss_pid(out_p, poi_sim)
        land_loss = _general_loss_pid(out_l, land_sim)
        return mob_loss, poi_loss, land_loss


class AutoWeightLoss(nn.Module):
    """Uncertainty weighting: sum_i exp(-s_i)*L_i + s_i"""
    def __init__(self, n_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses):
        total = 0.0
        for i, L in enumerate(losses):
            inv = torch.exp(-self.log_vars[i])
            total = total + inv * L + self.log_vars[i]
        return total


# ============================================================
# Downstream helper
# ============================================================
def run_downstream(city, task, embs):
    if task == "checkIn":
        if city == "NY":   return tasks_NY.tasks_chk.do_tasks(embs)
        if city == "Chi":  return tasks_Chi.tasks_chk.do_tasks(embs)
        if city == "SF":   return tasks_SF.tasks_chk.do_tasks(embs)

    if task == "crime":
        if city == "NY":   return tasks_NY.tasks_crime.do_tasks(embs)
        if city == "Chi":  return tasks_Chi.tasks_crime.do_tasks(embs)
        if city == "SF":   return tasks_SF.tasks_crime.do_tasks(embs)

    if task == "serviceCall":
        if city == "NY":   return tasks_NY.tasks_serviceCall.do_tasks(embs)
        if city == "Chi":  return tasks_Chi.tasks_serviceCall.do_tasks(embs)
        if city == "SF":   return tasks_SF.tasks_serviceCall.do_tasks(embs)

    raise ValueError(f"Unknown city/task: {city}/{task}")


# ============================================================
# Train (IMPORTANT FIX: save best_state, not only embedding)
# ============================================================
def train_model(features, mob_adj, poi_sim, land_sim,
                model, model_loss, city, task, device,
                log_path=None, jsonl_path=None,
                profile_tag="train", best_ckpt_path=None):

    epochs = args.epochs
    lr = args.learning_rate
    weight_decay = args.weight_decay

    # move data once
    features, mob_adj, poi_sim, land_sim = safe_to_device(features, mob_adj, poi_sim, land_sim, device)

    fusion_mode = getattr(model, "fusion_mode", getattr(args, "fusion_mode", "router"))
    use_awl = getattr(args, "auto_weight", False)

    awl = None
    if use_awl:
        n_losses = 4 if fusion_mode == "router" else 3
        awl = AutoWeightLoss(n_losses=n_losses).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(awl.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_r2 = -1e9
    best_state = None

    # profiling init
    try:
        N_regions = int(features[0].shape[0])
    except Exception:
        N_regions = None

    epoch_times = []
    reset_cuda_peak_stats(device)
    cuda_sync_if_needed(device)
    train_t0 = time.perf_counter()

    for epoch in range(epochs):
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()

        model.train()
        out_s, out_t, out_p, out_l = model(features)
        mob_loss, poi_loss, land_loss = model_loss(out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim)

        alpha_VN = get_routing_alpha_VN(model)  # None if concat
        if alpha_VN is None:
            loss_route = torch.tensor(0.0, device=device)
            if use_awl:
                loss = awl([mob_loss, poi_loss, land_loss])
            else:
                loss = mob_loss + poi_loss + land_loss
        else:
            loss_route = routing_entropy_loss(alpha_VN, objective=args.route_objective)
            if use_awl:
                loss = awl([mob_loss, poi_loss, land_loss, loss_route])
            else:
                loss = (mob_loss + poi_loss + land_loss) + args.lambda_moe * loss_route

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        if epoch % 30 == 0:
            embs = model.out_feature().detach().cpu().numpy()
            _, _, r2 = run_downstream(city, task, embs)

            msg = (f"Epoch {epoch} | total {loss.item():.6f} | "
                   f"mob {mob_loss.item():.6f} poi {poi_loss.item():.6f} land {land_loss.item():.6f} "
                   f"route {loss_route.item():.6f} | r2 {float(r2):.6f} | mode {fusion_mode}")

            ent = sp = None
            top1_ratio = None
            if alpha_VN is not None:
                ent, sp, top1_ratio, _, _ = router_stats(alpha_VN)
                msg += f" | router_ent {ent:.4f} router_sp {sp:.4f} top1_ratio {top1_ratio}"

            if use_awl:
                w = torch.exp(-awl.log_vars.detach()).cpu().numpy()
                msg += f" | awl_w {w}"

            log_print(msg, log_path)

            append_jsonl({
                "time": datetime.now().isoformat(timespec="seconds"),
                "type": "train_tick",
                "epoch": int(epoch),
                "city": city,
                "task": task,
                "fusion_mode": fusion_mode,
                "loss": float(loss.item()),
                "mob_loss": float(mob_loss.item()),
                "poi_loss": float(poi_loss.item()),
                "land_loss": float(land_loss.item()),
                "route_loss": float(loss_route.item()),
                "r2": float(r2),
                "router_entropy": float(ent) if ent is not None else None,
                "router_sparsity": float(sp) if sp is not None else None,
                "top1_ratio": top1_ratio.tolist() if top1_ratio is not None else None,
                "auto_weight": bool(use_awl),
            }, jsonl_path)

            if float(r2) > best_r2:
                best_r2 = float(r2)
                best_state = copy.deepcopy(model.state_dict())

    cuda_sync_if_needed(device)
    train_t1 = time.perf_counter()

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        log_print(f"[Restore] best checkpoint loaded (best_r2={best_r2:.6f})", log_path)

        if best_ckpt_path is not None:
            Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, best_ckpt_path)
            log_print(f"[Save] best checkpoint saved to: {best_ckpt_path}", log_path)

    # profiling summary
    total_time = train_t1 - train_t0
    avg_epoch = float(np.mean(epoch_times)) if len(epoch_times) > 0 else None
    p95_epoch = float(np.percentile(epoch_times, 95)) if len(epoch_times) > 0 else None
    throughput = (N_regions / avg_epoch) if (N_regions is not None and avg_epoch not in [None, 0.0]) else None
    peak_alloc_gb, peak_rsvd_gb = get_cuda_peak_gb(device)

    log_print("\n" + "-" * 80, log_path)
    log_print(f"[PROFILE:{profile_tag}] mode={fusion_mode} city={city} task={task}", log_path)
    if avg_epoch is not None:
        log_print(f"Avg time/epoch: {avg_epoch:.4f}s | P95: {p95_epoch:.4f}s | Total: {total_time/60:.2f} min", log_path)
    if throughput is not None:
        log_print(f"Throughput: {throughput:.2f} regions/s (N={N_regions})", log_path)
    if peak_alloc_gb is not None:
        log_print(f"Peak GPU mem: allocated {peak_alloc_gb:.3f} GB | reserved {peak_rsvd_gb:.3f} GB", log_path)
    log_print("-" * 80 + "\n", log_path)

    append_jsonl({
        "time": datetime.now().isoformat(timespec="seconds"),
        "type": "train_done",
        "city": city,
        "task": task,
        "fusion_mode": fusion_mode,
        "best_r2": float(best_r2),
        "epochs": int(epochs),
        "avg_epoch_s": avg_epoch,
        "p95_epoch_s": p95_epoch,
        "total_train_s": float(total_time),
        "throughput_regions_per_s": float(throughput) if throughput is not None else None,
        "peak_mem_alloc_GB": peak_alloc_gb,
        "peak_mem_reserved_GB": peak_rsvd_gb,
        "N_regions": N_regions,
        "best_ckpt_path": best_ckpt_path,
        "auto_weight": bool(use_awl),
    }, jsonl_path)

    # still save embedding for backward compatibility
    best_emb = model.out_feature().detach().cpu().numpy()
    np.save("best_emb.npy", best_emb)

    return model


# ============================================================
# Test (loads best_emb.npy) - keep for compatibility
# ============================================================
def test_model(city, task):
    best_emb = np.load("./best_emb.npy")
    log_print("Best region embeddings loaded.")

    mae, rmse, r2 = run_downstream(city, task, best_emb)
    log_print(f"[TEST] MAE {mae:.3f} | RMSE {rmse:.3f} | R2 {r2:.3f}")
    return mae, rmse, r2


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if not hasattr(args, "seed"):
        setattr(args, "seed", 42)
    set_full_seed(args.seed)

    log_dir = Path("concat-logs")
    log_dir.mkdir(exist_ok=True)

    # run tag (keep your original style)
    run_tag = f"{args.city}_{args.task}_seed{args.seed}_top{args.router_topk}_tem{args.router_temp}"
    log_path = str(log_dir / f"{run_tag}.log")
    jsonl_path = str(log_dir / f"{run_tag}.jsonl")
    best_ckpt_path = str(log_dir / f"{run_tag}_best_model.pt")

    # dump args
    log_print("=== ARGS ===", log_path)
    log_print(json.dumps(vars(args), indent=2, ensure_ascii=False), log_path)
    log_print("============", log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Device: {device}", log_path)

    # load data
    features, mob_adj, poi_sim, land_sim = utils.load_data()

    city = args.city
    task = args.task
    embedding_size = args.embedding_size
    d_prime = args.d_prime
    d_m = args.d_m
    c = args.c
    POI_dim = args.POI_dim
    landUse_dim = args.landUse_dim
    region_num = args.region_num

    # -------- FLOPs / Params report --------
    model_router = MoERoutingFusion(POI_dim, landUse_dim, region_num, embedding_size, d_prime, d_m, c).to(device)
    if hasattr(model_router, "fusion_mode"):
        model_router.fusion_mode = "router"
    report_model_complexity(model_router, features, device, title="PRISM (router)", log_path=log_path, jsonl_path=jsonl_path)

    model_concat = MoERoutingFusion(POI_dim, landUse_dim, region_num, embedding_size, d_prime, d_m, c).to(device)
    if hasattr(model_concat, "fusion_mode"):
        model_concat.fusion_mode = "concat"
    report_model_complexity(model_concat, features, device, title="w/o RAR (concat)", log_path=log_path, jsonl_path=jsonl_path)

    # -------- train model (respect args.fusion_mode) --------
    model = MoERoutingFusion(POI_dim, landUse_dim, region_num, embedding_size, d_prime, d_m, c).to(device)
    if hasattr(model, "fusion_mode"):
        model.fusion_mode = getattr(args, "fusion_mode", "router")

    model_loss = ModelLoss()

    log_print("Model Training-----------------", log_path)
    model = train_model(features, mob_adj, poi_sim, land_sim,
                        model, model_loss, city, task, device,
                        log_path=log_path, jsonl_path=jsonl_path,
                        profile_tag="full_train", best_ckpt_path=best_ckpt_path)

    # strongly recommended: reload best ckpt
    if Path(best_ckpt_path).exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.eval()
        log_print(f"[Load] best checkpoint loaded from: {best_ckpt_path}", log_path)

    # export router diagnostics files for plotting
    out_prefix = str(log_dir / run_tag)
    export_router_diagnostics(model, features, device, out_prefix=out_prefix, log_path=log_path)

    # downstream test (still uses best_emb.npy for compatibility)
    log_print("Downstream task test-----------", log_path)
    mae, rmse, r2 = test_model(city, task)
    append_jsonl({
        "time": datetime.now().isoformat(timespec="seconds"),
        "type": "downstream_full",
        "city": city, "task": task,
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
        "best_ckpt_path": best_ckpt_path
    }, jsonl_path)

    log_print("\n[Done]", log_path)

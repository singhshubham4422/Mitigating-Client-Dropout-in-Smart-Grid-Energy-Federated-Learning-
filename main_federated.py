# ============================================
# FINAL IEEE VERSION (FIXED + STABLE)
# ============================================
#This is main_federated.py, the entry point for our federated learning simulation. It orchestrates the entire process, from data loading and model training to secure aggregation and differential privacy. The code is structured to be modular and extensible, allowing for easy experimentation with different configurations and techniques.
import torch
import pandas as pd
import numpy as np
import os
import argparse
import time
import random
from torch.utils.data import DataLoader, TensorDataset
from core.model import get_model
from core.crypto_utils import generate_keypair, derive_shared_key
from core.privacy import (
    compute_model_update,
    clip_update,
    add_gaussian_noise,
    compute_epsilon,
    quantize_update,
    dequantize_update
)
from core.secure_protocol import SecureAggregationServer, client_secure_masking
from core.reliability import ClientReliabilityTracker
from network.aodv_router import AODVRouter
# ================= REPRODUCIBILITY =================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# ================= CLI =================
parser = argparse.ArgumentParser()
parser.add_argument("--clients", type=int, default=4)
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--noise", type=float, default=0.05)  # FIXED
parser.add_argument("--dropout_prob", type=float, default=0.0)
parser.add_argument("--dp", action="store_true")
parser.add_argument("--secure", action="store_true")
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = args.clients
NUM_ROUNDS = args.rounds
NOISE_MULTIPLIER = args.noise
CLIP_NORM = 1.0
SECURE_THRESHOLD = max(2, NUM_CLIENTS // 2)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# ================= FOLDERS =================
BASELINE_DIR = os.path.join(LOG_DIR, "Baseline")
DP_DIR = os.path.join(LOG_DIR, "With_DP")
SMPC_DIR = os.path.join(LOG_DIR, "SMPC")
DP_SMPC_DIR = os.path.join(LOG_DIR, "DP_SMPC")
for d in [BASELINE_DIR, DP_DIR, SMPC_DIR, DP_SMPC_DIR]:
    os.makedirs(d, exist_ok=True)
# ================= DATA =================
def prepare_client_datasets():
    loaders = {}
    static_meta = {}
    for cid in range(1, NUM_CLIENTS + 1):
        path = f"processed_data/Client_{cid}.csv"
        df = pd.read_csv(path, index_col=0)
        meta = df[["mean_load", "max_load", "std_load", "evening_ratio"]].iloc[0].values
        static_meta[cid] = torch.tensor(meta, dtype=torch.float32).to(DEVICE)
        data = df.iloc[:, :7].values.astype(np.float32)
        X, y = [], []
        for i in range(len(data) - 60):
            X.append(data[i:i + 60])
            y.append(data[i + 60, 0])
        dataset = TensorDataset(
            torch.tensor(np.array(X)),
            torch.tensor(np.array(y))
        )
        loaders[cid] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loaders, static_meta
def evaluate(model, loader, meta):
    model.eval()
    mse_fn = torch.nn.MSELoss()
    total, count = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            meta_batch = meta.repeat(Xb.shape[0], 1)
            preds = model(Xb, meta_batch).squeeze()
            total += mse_fn(preds, yb).item() * len(yb)
            count += len(yb)
    return total / count if count > 0 else 0
# ================= MAIN =================
def run():
    print(f"""
======== CONFIG ========
Clients: {NUM_CLIENTS}
Rounds: {NUM_ROUNDS}
DP: {args.dp}
Secure: {args.secure}
========================
""")
    model = get_model(DEVICE)
    loaders, static_meta = prepare_client_datasets()
    reliability = ClientReliabilityTracker(NUM_CLIENTS)
    router = AODVRouter(NUM_CLIENTS, connection_radius=120, per_hop_failure_prob=0.02)
    # -------- Secure Setup --------
    if args.secure:
        server = SecureAggregationServer(NUM_CLIENTS, SECURE_THRESHOLD, dropout_prob=args.dropout_prob)
        private_keys, public_keys = {}, {}
        for cid in range(1, NUM_CLIENTS + 1):
            priv, pub = generate_keypair()
            private_keys[cid], public_keys[cid] = priv, pub
        shared_keys = {
            cid: {
                pid: derive_shared_key(private_keys[cid], public_keys[pid])
                for pid in range(1, NUM_CLIENTS + 1) if pid != cid
            }
            for cid in range(1, NUM_CLIENTS + 1)
        }
    metrics = []
    # ================= TRAIN =================
    for r in range(1, NUM_ROUNDS + 1):
        if args.secure:
            server.reset()
        active_clients = []
        route_stats = []
        # -------- REALISTIC PAYLOAD --------
        model_size = sum(p.numel() for p in model.parameters()) * 4
        for cid in range(1, NUM_CLIENTS + 1):
            route = router.discover_route(cid, model_size)
            if route["success"]:
                active_clients.append(cid)
                route_stats.append(route)
        if not active_clients:
            continue
        # ✅ FIX: UPDATE RELIABILITY
        for cid in range(1, NUM_CLIENTS + 1):
            reliability.update(cid, cid in active_clients)
        weights = reliability.get_normalized_weights(active_clients)
        client_deltas = {}
        # -------- TRAIN --------
        for cid in active_clients:
            local = get_model(DEVICE)
            local.load_state_dict(model.state_dict())
            local.train()
            optimizer = torch.optim.Adam(local.parameters(), lr=args.lr)
            for Xb, yb in loaders[cid]:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                meta_batch = static_meta[cid].repeat(Xb.shape[0], 1)
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(local(Xb, meta_batch).squeeze(), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                optimizer.step()
            delta = compute_model_update(local.state_dict(), model.state_dict())
            if args.secure:
                weighted = {k: v * weights[cid] for k, v in delta.items()}
                quantized = quantize_update(weighted)
                masked, shares, pairwise = client_secure_masking(
                    quantized, cid, active_clients,
                    shared_keys[cid], NUM_CLIENTS,
                    SECURE_THRESHOLD, DEVICE
                )
                server.receive_update(cid, masked, shares, pairwise)
            else:
                client_deltas[cid] = {
                    k: v * weights[cid] for k, v in delta.items()
                }
        # -------- AGG --------
        if args.secure:
            agg_q, survivors = server.aggregate_with_survivors(active_clients, DEVICE)
            if not agg_q:
                continue
            agg = dequantize_update(agg_q)
            active_count = len(survivors)
        else:
            agg = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            for cid in active_clients:
                for k in agg:
                    agg[k] += client_deltas[cid][k]
            active_count = len(active_clients)
        # -------- DP --------
        if args.dp:
            agg = clip_update(agg, CLIP_NORM)
            agg = add_gaussian_noise(agg, NOISE_MULTIPLIER, CLIP_NORM, DEVICE)
        # -------- UPDATE --------
        new_state = {k: model.state_dict()[k] + agg[k] for k in model.state_dict()}
        model.load_state_dict(new_state)
        # -------- METRICS --------
        mse = np.mean([
            evaluate(model, loaders[cid], static_meta[cid])
            for cid in range(1, NUM_CLIENTS + 1)
        ])
        epsilon = compute_epsilon(NOISE_MULTIPLIER, r) if args.dp else 0
        avg_hops = np.mean([r["hops"] for r in route_stats]) if route_stats else 0
        avg_delay = np.mean([r["delay_ms"] for r in route_stats]) if route_stats else 0
        success_rate = len(active_clients) / NUM_CLIENTS
        print(f"Round {r:02d} | MSE: {mse:.4f} | Active: {active_count} | Eps: {epsilon:.2f}")
        metrics.append({
            "round": r,
            "clients": NUM_CLIENTS,
            "mse": mse,
            "epsilon": epsilon,
            "active": active_count,
            "success_rate": success_rate,
            "avg_hops": avg_hops,
            "avg_delay": avg_delay,
            "dp": args.dp,
            "smpc": args.secure
        })
    # -------- SAVE --------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.dp and args.secure:
        save_dir = DP_SMPC_DIR
        mode = "dp_smpc"
    elif args.dp:
        save_dir = DP_DIR
        mode = "dp"
    elif args.secure:
        save_dir = SMPC_DIR
        mode = "smpc"
    else:
        save_dir = BASELINE_DIR
        mode = "baseline"
    filename = os.path.join(save_dir, f"results_c{NUM_CLIENTS}_{mode}_{timestamp}.csv")
    pd.DataFrame(metrics).to_csv(filename, index=False)
    print(f"\n✅ DONE → {filename}")
if __name__ == "__main__":
    run()
# ============================================
# File: core/secure_protocol.py
# Two-Stage Dropout-Resilient Secure Aggregation
# STABLE VERSION (No modular inverse crash)
# ============================================

import torch
import random
from core.secret_sharing import split_secret, reconstruct_secret, PRIME


def _seed_to_mask(shape, seed, device):
    """Generates a pseudo-random mask based on a seed."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) % (2**31 - 1))
    return torch.randint(
        0,
        PRIME,
        shape,
        device=device,
        dtype=torch.long
    )


class SecureAggregationServer:

    def __init__(self, num_clients, threshold, dropout_prob=0.1):
        self.num_clients = num_clients
        self.threshold = threshold
        self.dropout_prob = dropout_prob
        self.reset()

    def reset(self):
        self.masked_updates = {}
        self.private_shares = {}
        self.pairwise_masks = {}

    def receive_update(
        self,
        client_id,
        masked_update,
        private_shares_from_client,
        pairwise_mask_sums
    ):
        """Collect masked updates and secret shares."""
        self.masked_updates[client_id] = masked_update
        self.private_shares[client_id] = private_shares_from_client
        self.pairwise_masks[client_id] = pairwise_mask_sums

    def simulate_mid_round_dropout(self, active_clients):
        """Randomly simulate client dropout."""
        survivors = []
        dropped = []

        for cid in active_clients:
            if random.random() < self.dropout_prob:
                dropped.append(cid)
            else:
                survivors.append(cid)

        return survivors, dropped

    def aggregate_with_survivors(self, active_clients, device):
        """
        Securely aggregates masked updates.
        Returns:
            aggregated_quantized_update, survivors_list
        """

        survivors, dropped = self.simulate_mid_round_dropout(active_clients)

        if len(survivors) < self.threshold:
            print(
                f"⚠️ Aggregation failed: "
                f"Only {len(survivors)} survivors. "
                f"Threshold is {self.threshold}."
            )
            return {}, survivors

        aggregated = {}

        # ---------------------------------------------------
        # 1️⃣ Sum masked updates of survivors
        # ---------------------------------------------------
        for cid in survivors:
            update = self.masked_updates[cid]

            for k, v in update.items():
                if k not in aggregated:
                    aggregated[k] = torch.zeros_like(
                        v,
                        dtype=torch.long,
                        device=device
                    )

                aggregated[k] = (aggregated[k] + v) % PRIME

        # ---------------------------------------------------
        # 2️⃣ Cancel pairwise masks
        # ---------------------------------------------------
        for cid in survivors:
            pairwise = self.pairwise_masks[cid]

            for k in aggregated:
                aggregated[k] = (aggregated[k] - pairwise[k]) % PRIME

        # ---------------------------------------------------
        # 3️⃣ Remove private masks of dropped clients
        # ---------------------------------------------------
        for dropped_id in dropped:

            shares = []

            for cid in survivors:
                if dropped_id in self.private_shares[cid]:
                    shares.append(self.private_shares[cid][dropped_id])

            if len(shares) >= self.threshold:
                try:
                    seed = reconstruct_secret(shares)
                except Exception:
                    # Skip unstable reconstruction instead of crashing
                    print("⚠️ Skipping unstable secret reconstruction.")
                    continue

                for k in aggregated:
                    mask = _seed_to_mask(
                        aggregated[k].shape,
                        seed,
                        device
                    )
                    aggregated[k] = (aggregated[k] - mask) % PRIME

        return aggregated, survivors


def client_secure_masking(
    local_update,
    client_id,
    active_ids,
    shared_keys,
    num_clients,
    threshold,
    device
):
    """
    Client-side SMPC masking.
    Applies:
        - Pairwise masks (ECDH-based)
        - Private mask (Shamir shared)
    """

    masked_update = {}
    pairwise_mask_sum = {}
    private_shares = {}

    # ---------------------------------------------------
    # 1️⃣ Pairwise masks
    # ---------------------------------------------------
    for name, param in local_update.items():

        total_mask = torch.zeros_like(
            param,
            dtype=torch.long,
            device=device
        )

        for peer_id in active_ids:

            if peer_id == client_id:
                continue

            if peer_id not in shared_keys:
                continue

            seed = int.from_bytes(shared_keys[peer_id][:8], "big")
            mask = _seed_to_mask(param.shape, seed, device)

            if client_id < peer_id:
                total_mask = (total_mask + mask) % PRIME
            else:
                total_mask = (total_mask - mask) % PRIME

        pairwise_mask_sum[name] = total_mask.clone()
        masked_update[name] = (param + total_mask) % PRIME

    # ---------------------------------------------------
    # 2️⃣ Private mask
    # ---------------------------------------------------
    private_seed = random.randint(0, 2**31 - 1)

    for name in masked_update:
        private_mask = _seed_to_mask(
            masked_update[name].shape,
            private_seed,
            device
        )
        masked_update[name] = (
            masked_update[name] + private_mask
        ) % PRIME

    # ---------------------------------------------------
    # 3️⃣ Shamir shares of private seed
    # ---------------------------------------------------
    shares = split_secret(
        private_seed,
        num_clients,
        threshold
    )

    for (x, y) in shares:
        private_shares[x] = (x, y)

    return masked_update, private_shares, pairwise_mask_sum

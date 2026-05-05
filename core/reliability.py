# ============================================
# File: core/reliability.py
# Reliability-weighted aggregation utilities
# ============================================

class ClientReliabilityTracker:
    """
    Tracks reliability score of each client.
    Provides normalized weights for aggregation.
    """

    def __init__(self, num_clients, decay=0.9, recovery=0.05):
        self.reliability_scores = {
            cid: 1.0 for cid in range(1, num_clients + 1)
        }

        self.decay = decay
        self.recovery = recovery
        self.min_score = 0.1

    def update(self, client_id, active: bool):
        if active:
            self.reliability_scores[client_id] = min(
                1.0,
                self.reliability_scores[client_id] + self.recovery
            )
        else:
            self.reliability_scores[client_id] = max(
                self.min_score,
                self.reliability_scores[client_id] * self.decay
            )

    def get(self, client_id):
        return self.reliability_scores.get(client_id, self.min_score)

    def get_normalized_weights(self, active_clients):
        """
        Returns reliability weights normalized to sum to 1.
        """
        weights = {
            cid: self.get(cid)
            for cid in active_clients
        }

        total = sum(weights.values())

        if total == 0:
            # fallback uniform
            return {cid: 1.0 / len(active_clients) for cid in active_clients}

        return {
            cid: weights[cid] / total
            for cid in active_clients
        }

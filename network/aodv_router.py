# ============================================
# File: network/aodv_router.py
# AODV with Payload & Energy Model
# ============================================

import random
import math
import networkx as nx


class AODVRouter:

    def __init__(
        self,
        num_clients,
        area_size=200,
        connection_radius=40,
        per_hop_delay_ms=3,
        per_hop_failure_prob=0.10,
        bandwidth_bytes_per_ms=50000,
        initial_energy=1000
    ):
        self.num_clients = num_clients
        self.server_node = 0
        self.area_size = area_size
        self.connection_radius = connection_radius
        self.per_hop_delay_ms = per_hop_delay_ms
        self.per_hop_failure_prob = per_hop_failure_prob
        self.bandwidth = bandwidth_bytes_per_ms

        self.node_energy = {
            i: initial_energy for i in range(num_clients + 1)
        }

        self._build_random_topology()

    def _build_random_topology(self):

        self.positions = {}

        for node in range(self.num_clients + 1):
            self.positions[node] = (
                random.uniform(0, self.area_size),
                random.uniform(0, self.area_size)
            )

        self.graph = nx.Graph()

        for node in self.positions:
            self.graph.add_node(node)

        for i in self.positions:
            for j in self.positions:
                if i >= j:
                    continue
                if self._distance(self.positions[i], self.positions[j]) <= self.connection_radius:
                    self.graph.add_edge(i, j)

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def discover_route(self, client_id, payload_bytes):

        if self.node_energy[client_id] <= 0:
            return {"success": False, "hops": 0, "delay_ms": 0}

        self._build_random_topology()

        try:
            path = nx.shortest_path(
                self.graph,
                source=client_id,
                target=self.server_node
            )

            hops = len(path) - 1

            for _ in range(hops):
                if random.random() < self.per_hop_failure_prob:
                    return {"success": False, "hops": 0, "delay_ms": 0}

            transmission_delay = payload_bytes / self.bandwidth

            total_delay = (hops * self.per_hop_delay_ms) + transmission_delay

            # Energy cost proportional to payload
            energy_cost = payload_bytes / 1e6
            self.node_energy[client_id] -= energy_cost

            return {
                "success": True,
                "hops": hops,
                "delay_ms": total_delay
            }

        except nx.NetworkXNoPath:
            return {"success": False, "hops": 0, "delay_ms": 0}

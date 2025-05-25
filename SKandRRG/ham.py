import math
import os
import time

import networkx as nx
import numba
import numpy as np
import torch


class Hamiltonian1D:
    def energy(self, s):
        assert s.shape[1] == self.n
        return -0.5 * torch.sum((s @ self.J) * s, dim=1)


class SKModel(Hamiltonian1D):
    def __init__(self, n=20, seed=1, device="cpu", *args, **kwargs):
        rng = np.random.default_rng(seed)
        self.J_np = rng.normal(size=(n, n)) / math.sqrt(n)
        self.J_np = np.triu(self.J_np, k=1)
        self.J_np = self.J_np + self.J_np.T
        self.J = torch.from_numpy(self.J_np).float().to(device)
        self.n = n
        self.seed = seed

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.n}, seed={self.seed}, mean(J)={self.J_np.mean():.4g}, std(J)={self.J_np.std():.4g})"


class RRGInstance(Hamiltonian1D):
    def __init__(self, n=20, d=3, seed=1, device="cpu", *args, **kwargs):
        rng = np.random.default_rng(seed)
        graph = nx.random_regular_graph(d=d, n=n, seed=seed)
        weights = rng.integers(2, size=len(graph.edges)) * 2 - 1
        for (u, v), w in zip(graph.edges(), weights):
            graph[u][v]["weight"] = w
        adj_matrix = nx.adjacency_matrix(graph)
        self.J_np = np.triu(adj_matrix.toarray(), k=1).astype(float)
        self.J_np = self.J_np + self.J_np.T
        self.J = torch.from_numpy(self.J_np).float().to(device)
        self.n = n
        self.d = d
        self.seed = seed

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.n}, d={self.d}, seed={self.seed}, mean(J)={self.J_np.mean():.4g}, std(J)={self.J_np.std():.4g})"


@numba.jit(nopython=True)
def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


@numba.jit(nopython=True)
def exact(beta, J):
    n = J.shape[0]
    assert n <= 30
    n_total = 2**n
    energy_arr = np.zeros(n_total)
    for idx in range(n_total):
        # Use bitwise operations to generate spin configurations
        s = ((idx >> np.arange(n)) & 1) * 2.0 - 1
        energy_arr[idx] = -0.5 * s.T @ J @ s
    lnZ = logsumexp(-beta * energy_arr)
    free_energy = -1.0 * lnZ / beta / n
    prob_arr = np.exp(-1.0 * beta * energy_arr - lnZ)
    energy = np.sum(prob_arr * energy_arr) / n
    entropy = beta * (energy - free_energy)

    return free_energy, energy, entropy


if __name__ == "__main__":
    n = 25
    seed = 4
    ham = SKModel(n=n, seed=seed)

    path = "./out/exact"
    if not os.path.exists(path):
        os.makedirs(path)

    for beta in np.linspace(0.1, 5.0, 50):
        t1 = time.time()
        free_energy, energy, entropy = exact(beta, ham.J_np)
        time_used = time.time() - t1
        print(f"bata={beta:.2f}, f={free_energy:.4f}, e={energy:.4f}, s={entropy:.4f}, time={time_used:.4f} s")

        with open(os.path.join(path, f"sk_n{n}_seed{seed}.txt"), "a", newline="\n") as f:
            f.write(f"{beta:.2f} {free_energy} {energy} {entropy} {time_used}\n")

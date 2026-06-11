#!/usr/bin/env python3
"""Build queueing DP oracle for small systems and export labeled distillation data.

For N=2,3 with truncated queue lengths (K=15-20), this script:
1. Builds the finite MDP
2. Solves it via discounted value iteration
3. Compares SED, QMD, JSWQ against the optimal policy
4. Exports a labeled dataset for oracle distillation training

Usage:
    python scripts/build_queueing_oracle.py [--output data/oracle_labels.pt]
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import torch


def build_mdp(N: int, K: int, lam: float, mu: torch.Tensor, gamma: float = 0.99):
    """Build finite MDP for N servers, max queue K per server.

    State space: {0..K}^N.  Actions: N (assign arrival to server i).
    Returns transition arrays and cost vector.
    """
    states = list(product(range(K + 1), repeat=N))
    S = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    P = torch.zeros(S, N, S)
    C = torch.zeros(S, dtype=torch.float32)
    mu_np = mu.cpu().numpy()
    for s_idx, s in enumerate(states):
        Q = torch.tensor(s, dtype=torch.float32)
        C[s_idx] = Q.sum().item()
        total_serv_rate = sum(mu_np[i] for i in range(N) if s[i] > 0)
        total_rate = lam + total_serv_rate
        for a in range(N):
            s_arrival = list(s)
            s_arrival[a] = min(s_arrival[a] + 1, K)
            s_arrival_idx = state_to_idx[tuple(s_arrival)]
            P[s_idx, a, s_arrival_idx] += lam / total_rate
            for i in range(N):
                if s[i] > 0:
                    s_service = list(s)
                    s_service[i] -= 1
                    s_service_idx = state_to_idx[tuple(s_service)]
                    P[s_idx, a, s_service_idx] += mu_np[i] / total_rate
    return P, C, states


def value_iteration(
    P: torch.Tensor, C: torch.Tensor, gamma: float = 0.99, tol: float = 1e-6, max_iter: int = 5000
):
    """Discounted value iteration."""
    S, N, _ = P.shape
    V = torch.zeros(S)
    Q_values = torch.zeros(S, N)
    for iteration in range(max_iter):
        V_prev = V.clone()
        for a in range(N):
            Q_values[:, a] = C + gamma * (P[:, a, :] @ V)
        V, _ = Q_values.min(dim=-1)
        if (V - V_prev).abs().max().item() < tol:
            break
    pi_oracle = Q_values.argmin(dim=-1)
    return V, Q_values, pi_oracle, iteration + 1


def heuristic_actions(states_t: torch.Tensor, mu: torch.Tensor):
    """Return SED, QMD, JSWQ actions for all states."""
    mu_b = mu.unsqueeze(0) if mu.dim() == 1 else mu
    sed = ((states_t + 1.0) / mu_b.clamp_min(torch.finfo(states_t.dtype).tiny)).argmin(dim=-1)
    qmd = ((2.0 * states_t + 1.0) / mu_b.clamp_min(torch.finfo(states_t.dtype).tiny)).argmin(dim=-1)
    jswq = (states_t / mu_b.pow(1.0).clamp_min(torch.finfo(states_t.dtype).tiny)).argmin(dim=-1)
    return sed, qmd, jswq


def compute_marginal_values(V: torch.Tensor, states, mu: torch.Tensor, gamma: float = 0.99):
    """Estimate delta V_i(Q) = V(Q+e_i) - V(Q).

    For each state and action, the marginal value of assigning to server i is:
      delta_i(Q) = Q[s, a=i] - V[s]  (from Bellman optimality)
    """
    S = V.shape[0]
    N = len(mu)
    delta = torch.zeros(S, N)
    for s_idx, s in enumerate(states):
        for i in range(N):
            s_next = list(s)
            s_next[i] = min(s_next[i] + 1, max(states)[i])
            s_next_idx = states.index(tuple(s_next))
            delta[s_idx, i] = V[s_next_idx].item() - V[s_idx].item()
    return delta


def analyze_system(N: int, K: int, lam: float, mu: torch.Tensor, gamma: float = 0.95):
    """Run full analysis for one (N, K, lam, mu) configuration."""
    if N > 3:
        raise ValueError(
            "Dense oracle construction only supports N<=3. "
            "Use a sparse / approximate method for larger systems."
        )
    t0 = time.time()
    P, C, states = build_mdp(N, K, lam, mu, gamma=gamma)
    V, Q_vals, pi_oracle, iters = value_iteration(P, C, gamma=gamma)
    elapsed = time.time() - t0
    S = len(states)
    states_t = torch.tensor(states, dtype=torch.float32)
    pi_sed, pi_qmd, pi_jswq = heuristic_actions(states_t, mu)
    agree_sed = (pi_oracle == pi_sed).float().mean().item()
    agree_qmd = (pi_oracle == pi_qmd).float().mean().item()
    agree_jswq = (pi_oracle == pi_jswq).float().mean().item()
    q_sed = Q_vals[range(S), pi_sed]
    q_qmd = Q_vals[range(S), pi_qmd]
    q_jswq = Q_vals[range(S), pi_jswq]
    q_oracle = Q_vals[range(S), pi_oracle]
    delta = compute_marginal_values(V, states, mu, gamma=gamma)
    dataset = {
        "N": N, "K": K, "lam": lam, "gamma": gamma,
        "mu": mu.cpu(), "states": states_t.cpu(),
        "V": V.cpu(), "Q_vals": Q_vals.cpu(),
        "pi_oracle": pi_oracle.cpu(), "delta_V": delta.cpu(),
        "pi_sed": pi_sed.cpu(), "pi_qmd": pi_qmd.cpu(), "pi_jswq": pi_jswq.cpu(),
    }
    results = {
        "N": N, "K": K, "lam": lam, "S": S, "iters": iters, "elapsed": elapsed,
        "agree_sed": agree_sed, "agree_qmd": agree_qmd, "agree_jswq": agree_jswq,
        "gap_sed": (q_sed - q_oracle).mean().item(),
        "gap_qmd": (q_qmd - q_oracle).mean().item(),
        "gap_jswq": (q_jswq - q_oracle).mean().item(),
        "max_gap_sed": (q_sed - q_oracle).max().item(),
        "max_gap_qmd": (q_qmd - q_oracle).max().item(),
        "max_gap_jswq": (q_jswq - q_oracle).max().item(),
    }
    return results, dataset


def print_table(results_list):
    print()
    print("=" * 130)
    print(f"{'N':>3s}  {'K':>4s}  {'lam':>6s}  {'States':>8s}  {'Iters':>6s}  "
          f"{'Time':>6s}  {'AgreeSED':>10s}  {'AgreeQMD':>10s}  {'AgreeJSWQ':>10s}  "
          f"{'GapSED':>8s}  {'MaxGapSED':>10s}")
    print("-" * 130)
    for r in results_list:
        print(f"{r['N']:>3d}  {r['K']:>4d}  {r['lam']:>6.2f}  {r['S']:>8d}  {r['iters']:>6d}  "
              f"{r['elapsed']:>6.1f}  {r['agree_sed']:>10.4f}  {r['agree_qmd']:>10.4f}  "
              f"{r['agree_jswq']:>10.4f}  {r['gap_sed']:>8.4f}  {r['max_gap_sed']:>10.4f}")
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Build queueing DP oracle labels")
    parser.add_argument("--output", type=str, default="data/oracle_labels.pt",
                        help="Output path for the labeled dataset")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configs = [
        (2, 15, 1.2, torch.tensor([1.0, 1.0])),
        (2, 15, 1.2, torch.tensor([1.0, 2.0])),
        (2, 15, 1.0, torch.tensor([1.0, 3.0])),
        (2, 20, 1.6, torch.tensor([1.0, 1.0])),
        (2, 20, 1.6, torch.tensor([1.0, 2.0])),
        (3, 10, 1.8, torch.tensor([1.0, 1.5, 0.8])),
    ]
    results_list = []
    datasets = []
    for N, K, lam, mu in configs:
        r, ds = analyze_system(N=N, K=K, lam=lam, mu=mu)
        results_list.append(r)
        datasets.append(ds)
    print_table(results_list)
    combined = {k: [ds[k] for ds in datasets] for k in datasets[0].keys()}
    combined["configs"] = configs
    torch.save(combined, output_path)
    print(f"\nLabeled dataset saved to {output_path}")
    print(f"  {len(datasets)} configurations, total states: "
          f"{sum(ds['states'].shape[0] for ds in datasets)}")


if __name__ == "__main__":
    main()

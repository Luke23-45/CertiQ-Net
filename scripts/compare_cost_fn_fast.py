"""
Fast comparison: SED vs QMD base geometry for CertiQ-Net.

No full training pipeline — instantiates two models with identical
random seeds, runs a lightweight REINFORCE loop directly, and compares.

Usage:
    python -m scripts.compare_cost_fn_fast
"""

from __future__ import annotations

import sys, torch, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from certiqnet.dispatcher.certiq.index_model import CertiQIndexModel
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.utils.ctmc import CTMCEnvironment

torch.manual_seed(42)
DEVICE = torch.device("cpu")


def make_model(cost_fn: str, seed: int = 42) -> CertiQIndexModel:
    torch.manual_seed(seed)
    return CertiQIndexModel(
        N=4, hidden_dim=128, tau=1.0, exploration_temperature=1.5,
        C=20.0, beta=1.0, cost_fn=cost_fn, token_layers=2, global_layers=2,
        dropout=0.0, certificate_mode="none", constraint_mode="exact",
    ).to(DEVICE)


@torch.no_grad()
def eval_avg_cost(model: CertiQIndexModel, env: CTMCEnvironment, horizon: int, gamma: float) -> float:
    """Greedy rollout from current env state, return average discounted return."""
    Q_init = env.Q.clone()
    env.reset(Q_init)
    returns = torch.zeros(env.B, device=DEVICE)
    running = torch.zeros(env.B, device=DEVICE)
    cost_trace = []
    for _ in range(horizon):
        out = model.forward_full(env.Q, env.mu, training_mode=False)
        action_idx = out.pi.argmax(dim=-1)
        action_pi = one_hot(action_idx, num_classes=env.N).float()
        step = env.step(action_pi)
        reward = -(step["cost"].float() * step["dt"].float())
        cost_trace.append(step["cost"].float().mean().item() * step["dt"].float().mean().item())
        running = reward + gamma * running
        returns += running
    env.reset(Q_init)
    return -returns.mean().item()


def main() -> None:
    N, B = 4, 64
    horizon, gamma = 64, 0.99
    train_steps, lr = 200, 3e-4
    lam = 2.0
    mu = torch.tensor([10.0, 8.0, 6.0, 4.0], device=DEVICE)
    mu_b = mu.unsqueeze(0).expand(B, -1)

    results_buf = io.StringIO()
    def log(msg="") -> None:
        print(msg)
        results_buf.write(msg + "\n")

    model_sed = make_model("sed")
    model_qmd = make_model("qmd")

    log("=" * 80)
    log("TRAINING: REINFORCE — SED vs QMD (identical seeds)")
    log(f"  horizon={horizon}, steps={train_steps}, B={B}, N={N}, lr={lr}, C=20, cert_mode=none")
    log("=" * 80)

    data: dict[str, dict] = {
        "sed": {"returns": [], "roll": [], "ent": []},
        "qmd": {"returns": [], "roll": [], "ent": []},
    }

    for cost_fn, model in (("sed", model_sed), ("qmd", model_qmd)):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        log(f"\n  === cost_fn={cost_fn} ===")

        for step in range(train_steps):
            # Fresh random initial state each step
            Q = torch.randint(0, 15, (B, N), device=DEVICE).float()
            env = CTMCEnvironment(N=N, lam=lam, mu=mu, B=B)
            env.reset(Q)

            # Initial forward pass
            init_out = model.forward_full(env.Q, mu_b, training_mode=True)

            # Rollout
            log_probs, rewards = [], []
            for _ in range(horizon):
                out = model.forward_full(env.Q, mu_b, training_mode=True)
                dist = Categorical(probs=out.pi)
                action_idx = dist.sample()
                action_pi = one_hot(action_idx, num_classes=N).float()
                step_data = env.step(action_pi)
                reward = -(step_data["cost"].float() * step_data["dt"].float())
                log_probs.append(dist.log_prob(action_idx))
                rewards.append(reward)

            rewards_t = torch.stack(rewards, dim=0)
            log_probs_t = torch.stack(log_probs, dim=0)
            returns = torch.zeros_like(rewards_t)
            running = torch.zeros(B, device=DEVICE)
            for t in reversed(range(horizon)):
                running = rewards_t[t] + gamma * running
                returns[t] = running

            # Loss
            q = torch.softmax(init_out.proposal_logits, dim=-1)
            L_ent = -(q * q.clamp_min(1e-8).log()).sum(dim=-1).mean()
            L_roll = -(log_probs_t.reshape(-1) * returns.reshape(-1).detach()).mean()
            total = 1.0 * L_roll - 0.02 * L_ent
            total.backward()
            opt.step()

            # Eval every 20 steps
            if step % 20 == 0 or step == train_steps - 1:
                avg_return = -returns.sum(dim=0).mean().item()
                avg_ent = -(out.pi * out.pi.clamp_min(1e-9).log()).sum(dim=-1).mean().item()
                data[cost_fn]["returns"].append(avg_return)
                data[cost_fn]["roll"].append(L_roll.item())
                data[cost_fn]["ent"].append(avg_ent)
                log(f"  step {step:>4}:  avg_return={avg_return:>9.2f}  "
                    f"L_roll={L_roll.item():>10.2f}  entropy={avg_ent:.4f}")

    # ── Final table ──
    log(f"\n{'='*80}")
    log("CONVERGENCE SIDE-BY-SIDE")
    log(f"{'='*80}")
    log(f"  {'Step':>6}  {'SED_ret':>9}  {'QMD_ret':>9}  {'SED_roll':>10}  {'QMD_roll':>10}  "
        f"{'SED_ent':>8}  {'QMD_ent':>8}")
    log(f"  {'-'*70}")
    for i in range(len(data["sed"]["returns"])):
        step = i * 20
        log(f"  {step:>6}  {data['sed']['returns'][i]:>9.2f}  {data['qmd']['returns'][i]:>9.2f}  "
            f"{data['sed']['roll'][i]:>10.2f}  {data['qmd']['roll'][i]:>10.2f}  "
            f"{data['sed']['ent'][i]:>8.4f}  {data['qmd']['ent'][i]:>8.4f}")

    # ── Compare learned residuals ──
    log(f"\n{'='*80}")
    log("LEARNED RESIDUALS SIDE-BY-SIDE (on random states)")
    log(f"{'='*80}")
    Q_test = torch.randint(0, 15, (B, N), device=DEVICE).float()
    with torch.no_grad():
        out_s = model_sed.forward_full(Q_test, mu_b, training_mode=False)
        out_q = model_qmd.forward_full(Q_test, mu_b, training_mode=False)
        base_s = model_sed._base_geometry(Q_test, mu_b)
        base_q = model_qmd._base_geometry(Q_test, mu_b)
        resid_s = out_s.index_values - base_s
        resid_q = out_q.index_values - base_q

    log(f"  {'Metric':<25} {'SED':>25} {'QMD':>25}")
    log(f"  {'-'*75}")
    log(f"  {'Base index (mean per queue)':<25} {str(base_s.mean(dim=0).tolist()):>25} {str(base_q.mean(dim=0).tolist()):>25}")
    log(f"  {'Residual (mean per queue)':<25} {str(resid_s.mean(dim=0).tolist()):>25} {str(resid_q.mean(dim=0).tolist()):>25}")
    log(f"  {'Residual |max| mean':<25} {resid_s.abs().mean().item():>25.4f} {resid_q.abs().mean().item():>25.4f}")
    log(f"  {'Residual std':<25} {resid_s.std().item():>25.4f} {resid_q.std().item():>25.4f}")

    # Ratio of learned correction to base
    ratio_s = (resid_s.abs() / base_s.clamp_min(1e-8)).mean().item()
    ratio_q = (resid_q.abs() / base_q.clamp_min(1e-8)).mean().item()
    log(f"  {'|residual|/|base| ratio':<25} {ratio_s:>25.4f} {ratio_q:>25.4f}")

    out_path = ROOT / "outputs" / "cost_fn_fast_compare.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(results_buf.getvalue(), encoding="utf-8")
    log(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

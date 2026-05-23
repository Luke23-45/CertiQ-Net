# Training Objectives

This file defines how CertiQ-Net should be trained without confusing training
success with mathematical certification.

## 1. Training Regimes

CertiQ-Net supports three training regimes.

### Regime 1: analytic warm start

Train the neural proposal to imitate the analytic base controller or a
rollout-improved target on sampled states:

\[
\mathcal L_{\mathrm{bc}}(\Theta)
=
\mathbb E_{Q\sim\mathcal D}
\left[
\mathrm{KL}
\left(
p^{\mathrm{target}}(\cdot|Q)
\Vert
\pi_\Theta(\cdot|Q)
\right)
\right].
\]

Targets may include:

- the analytic base controller,
- dynamic-programming policies for tiny systems,
- rollout-improved policies.

### Regime 2: differentiable simulation fine tuning

Use a smoothed discrete-event simulation objective:

\[
\widehat J_T(\Theta)
=
\frac1T
\int_0^T c(Q_\Theta(t),\xi)\,dt.
\]

Train by pathwise gradients when the simulator supports differentiable event
smoothing:

\[
\nabla_\Theta \widehat J_T(\Theta)
\approx
\frac{\partial}{\partial\Theta}
\left[
\frac1T\int_0^T c(Q_\Theta(t),\xi)\,dt
\right].
\]

This is an optimization tool, not a stability proof.

### Regime 3: score-function fallback

When differentiable simulation is not available, use policy-gradient estimators
such as REINFORCE as an optimizer fallback:

\[
\nabla_\Theta J(\Theta)
=
\mathbb E
\left[
\sum_k
\nabla_\Theta\log \pi_\Theta(A_k|Q_k)
\widehat A_k
\right].
\]

This is expected to have higher variance in long-horizon queueing systems.

## 2. Main Loss

The recommended training loss is

\[
\mathcal L(\Theta)
=
\widehat J_T(\Theta)
+
\lambda_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}(\Theta)
+
\lambda_{\mathrm{gate}}\mathcal L_{\mathrm{gate}}(\Theta)
+
\lambda_{\mathrm{drift}}\mathcal L_{\mathrm{drift}}(\Theta)
+
\lambda_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}(\Theta).
\]

Each term has a specific role.

## 3. Gate Penalty

The gate should not learn to ignore the base controller everywhere. Penalize large
residual use:

\[
\mathcal L_{\mathrm{gate}}
=
\mathbb E_Q[\eta_\Theta(Q,\xi)^2].
\]

For compact-support gates, also penalize leakage outside the certificate
region:

\[
\mathcal L_{\mathrm{tail}}
=
\mathbb E_Q
\left[
\eta_\Theta(Q,\xi)^2
\mathbf 1_{\{S(Q)>R_{\mathrm{cert}}\}}
\right].
\]

## 4. Drift Penalty

Let

\[
A_\Theta(Q)
=
\sum_i \pi_i^\Theta(Q,\xi)Q_i/\mu_i^\beta.
\]

Let \(B(Q)\) be a certified or conservative arrival envelope. A soft penalty is

\[
\mathcal L_{\mathrm{drift}}
=
\mathbb E_Q
\left[
\left(A_\Theta(Q)-B(Q)\right)_+^2
\right].
\]

If a projection layer is active, this term should be near zero and functions
mainly as a diagnostic.

## 5. Entropy Control

Entropy regularization can prevent premature collapse:

\[
\mathcal L_{\mathrm{ent}}
=
-
\mathbb E_Q
\left[
\sum_i \pi_i^\Theta(Q,\xi)\log\pi_i^\Theta(Q,\xi)
\right].
\]

Use this carefully. Too much entropy can fight the queueing objective by
over-routing to poor servers.

## 6. Curriculum

Recommended training order:

1. train CertiQ-Net-S by imitation of the analytic base controller,
2. enable residual gate inside a compact region,
3. fine tune with differentiable simulation on small systems,
4. compare architecture variants under the same system and training budget,
5. scale to larger \(N\),
6. only then enable learned context parameters \(\phi_\Theta(\xi)\).

## 7. Training Metrics

Track:

- average total queue length,
- tail quantiles of total queue length,
- server utilization imbalance,
- gate activation rate,
- drift-envelope violation rate,
- gradient variance,
- wall-clock training time,
- generalization to unseen \(\mu\) vectors and loads.

The gate activation rate is scientifically important. If the learned residual
never activates, CertiQ-Net has not improved the base controller. If it
activates in the tail without proof, the model is not certified.

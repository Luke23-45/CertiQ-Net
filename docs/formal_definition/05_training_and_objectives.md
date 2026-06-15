# Training And Objectives

## 1. Separation Of Roles

Training selects parameters. Certification defines admissible actions. The two
roles are distinct and must not be conflated.

Any training objective may improve rollout cost, imitation quality, entropy, or
proposal usefulness, but it does not itself certify the policy.

## 2. Objective Decomposition

The training loss may be written as

\[
\mathcal L(\Theta)
=
\omega_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+\omega_{\mathrm{ce}}\mathcal L_{\mathrm{ce}}
+\omega_{\mathrm{margin}}\mathcal L_{\mathrm{margin}}
+\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+\omega_{\mathrm{kl}}\mathcal L_{\mathrm{kl}},
\]

where the terms denote rollout cost, cross-entropy behavior-cloning loss,
heuristic ranking margin loss, entropy regularization, and the
proposal-certificate KL penalty.

The certificate penalty is no longer a function of the budget surplus;
it is instead the reverse KL divergence from the uncertified proposal
to the certified (projected) policy.

## 3. Proposal-Certificate KL Penalty

The KL penalty is defined at the proposal level:

\[
\mathcal L_{\mathrm{kl}}
=
\mathbb E\!\left[\mathrm{KL}\bigl(q_\Theta(\cdot\mid x)
\;\|\; \pi^\star(\cdot\mid x)\bigr)\right],
\]

where \(q_\Theta\) is the raw (uncertified) proposal distribution and
\(\pi^\star\) is the certified policy obtained by the KL projection
\(\pi^\star = \arg\min_{\pi\in\Delta_N} \mathrm{KL}(\pi\|q_\Theta)\)
subject to the budget constraint
\(\mathbb E_\pi[A] \le B\).

This formulation penalises the proposal only when it places probability
mass on actions that the certificate would cut off.  Because the
certificate sets \(\pi^\star(a)=0\) for infeasible actions, the ratio
\(q_\Theta(a)/\pi^\star(a)\) in the KL term diverges for any infeasible
mass — providing a strong gradient signal to pull the proposal back
into the feasible set.

Conversely, exploration that stays entirely within the feasible set
yields \(q_\Theta \approx \pi^\star\) and the KL penalty is near zero.
The reverse-KL direction is mode-seeking, ensuring that the proposal is
free to concentrate mass on any feasible action without penalty.

### 3.1 Why Not a Budget-Violation Penalty

A squared ReLU on budget surplus,
\(\mathbb E[(A_{q_\Theta}-B)_+^2]\), penalises the proposal only at the
level of the expected cost — it does not distinguish between different
infeasible allocations and provides no incentive to concentrate
probability on any particular feasible action.  The reverse-KL penalty
is stronger and more structured: it forces the proposal to match the
certified policy on a per-action basis, eliminating infeasible
probability mass entirely rather than merely reducing the expected
cost below budget.

### 3.2 Gradient Semantics

The certified policy \(\pi^\star\) is **detached** from the
computational graph in the KL term:

\[
\mathcal L_{\mathrm{kl}} = \mathbb E\,
\bigl[\mathrm{KL}(q_\Theta \;\|\; \mathrm{sg}[\pi^\star])\bigr],
\]

where \(\mathrm{sg}[\cdot]\) denotes the stop-gradient operator.
This ensures gradients flow only through the uncertified proposal
\(q_\Theta\), treating the certified boundary as a fixed constraint
envelope.  Gradients through \(\pi^\star\) would pull the projection
layer's Lagrange multiplier in the opposite direction, weakening the
constraint — exactly what the penalty is designed to prevent.

## 4. Proposal Learning

The learned proposal may be trained as a residual over the QMD geometry:

\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

Soft SED or QMD targets may be used for warm-start supervision. Such targets are
derived from the observed state and service rates rather than from an external
labeled dataset.

## 5. Evaluation Quantities

Evaluation quantities include performance and certificate measurements.

Performance quantities include average backlog, tail backlog quantiles, and
maximum observed backlog.

Certificate quantities include violation rate, minimum slack, average slack,
projection activation rate, fallback activation rate, and correction magnitude.

No empirical result is certified unless the certificate quantities are
reported.

## 6. Curriculum

A consistent curriculum is:

1. verify the geometry and budget definitions,
2. train the proposal against QMD targets,
3. activate exact certificate enforcement,
4. optimize rollout cost under the certified policy,
5. report certificate metrics together with performance metrics.

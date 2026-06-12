# CertiQ Dispatcher Architecture

## 1. Canonical Implementation Families

The repository implements two dispatcher families:

1. `CertiQIndexModel`
   - learned marginal-cost index,
   - SED/QMD-aligned backbone,
   - exact KL projection with finite-region SED fallback.
2. `CertiQDispatcher`
   - legacy reflected-pressure dispatcher,
   - analytic base geometry,
   - scalar usage cap with fallback modes.

The common architectural pattern is:
\[
\pi^\Theta(Q,\mu,\xi,p)
=
\mathcal C_Q\!\left(
\mathcal A_\Theta(Q,\mu,\xi,p),
\mathcal B(Q,\mu)
\right),
\]
where the pressure state \(p\) is optional and only used by the legacy
dispatcher path.

## 2. Index Model

The index model is trained against the quadratic drift geometry
\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

The learned head predicts a residual \(r_i^\Theta\) and an index value
\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

The raw proposal is
\[
q_\Theta=\operatorname{softmax}\!\left(-\hat I_\Theta/\tau\right).
\]

The certificate layer then applies the exact KL projection onto a delay-aligned budget:
\[
\pi^\Theta
=
\arg\min_{\pi\in\Delta_N} KL(\pi\|q_\Theta)
\quad\text{s.t.}\quad
\sum_i \pi_i d_i^{QMD}\le d_{\min}(Q,\mu)+C_d(Q).
\]

This is implemented by exponential tilting with a Lagrange multiplier. The
operator preserves the full simplex instead of mixing between two fixed
policies.

For the certified tail region, the learned index is disabled when the queue
state leaves a finite ball:
\[
\|Q\|_1>R \Rightarrow \pi^\Theta=\pi^{SED}.
\]

This tail rule keeps the learned model as a finite-region perturbation of a
classical delay-aware router.

## 3. Legacy Reflected-Pressure Dispatcher

The legacy dispatcher keeps a learned proposal on top of an analytic base
geometry:
\[
u_i^{\mathrm{cert}}
=
\gamma\log\mu_i
-
\alpha\frac{Q_i+c}{\mu_i^\beta}.
\]

The base policy is
\[
p^{\mathrm{cert}}=\operatorname{softmax}(u^{\mathrm{cert}}).
\]

The proposal module produces a correction \(r_i^\Theta\), then shifts logits by
an internal nonnegative pressure state \(p_i\):
\[
u_i^{\mathrm{prop}}
=
u_i^{\mathrm{cert}}+r_i^\Theta-\rho p_i.
\]

The raw proposal is
\[
p^{\mathrm{prop}}=\operatorname{softmax}(u^{\mathrm{prop}}).
\]

The certificate layer then chooses one of three modes:

1. `projection`
   - return a scalar usage cap that mixes `p^{cert}` and `p^{prop}`;
2. `fallback`
   - keep the certified base policy outside a tail region;
3. `uncertified`
   - return the raw usage for ablation only.

## 4. Proposal Module

The proposal module is permutation equivariant. It uses shared local features,
an invariant pooled summary, and an equivariant resource update.

In the legacy path, the local features include:

\[
[\log(1+Q_i),\log\mu_i,y_i,\mu_i/\Lambda,\log(1+p_i),\xi_i].
\]

The module returns:

1. proposal probabilities,
2. residual correction logits,
3. raw usage preference,
4. value estimate for training.

## 5. Certificate Boundary

The certificate boundary is part of the dispatcher itself. The final policy is
the distribution passed to the simulator, trainer, and evaluator.

For the index model, the boundary is an exact KL projection with a certified
SED tail fallback.
For the legacy dispatcher, the boundary is a scalar usage cap or a fallback
rule. In both cases, certification must be explicit at inference time.

## 6. Diagnostics

Every forward pass emits auditable diagnostics. The core quantities are:

1. certified arrival coordinate,
2. proposal arrival coordinate,
3. final arrival coordinate,
4. certificate slack,
5. usage raw and usage final,
6. fallback indicator when applicable,
7. projection multiplier when applicable,
8. projection activation flag,
9. correction magnitude,
10. policy entropy,
11. selected resource summary,
12. pressure statistics for the legacy path.

These diagnostics are not optional logging. They are part of the architecture
definition.

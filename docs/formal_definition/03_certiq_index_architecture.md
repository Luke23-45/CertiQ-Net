# CertiQ Dispatcher Architecture

## 1. Model Architecture

The repository implements one certified dispatch model:

- `CertiQIndexModel`
  - learned marginal-cost index,
  - SED/QMD-aligned backbone,
  - exact KL projection with a fixed QMD budget.

The architectural pattern is:

\[
\pi^\Theta(Q,\mu,\xi)
=
\mathcal C_Q\!\left(
\mathcal A_\Theta(Q,\mu,\xi),
\mathcal B(Q,\mu)
\right).
\]

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

The certificate layer then applies the exact KL projection onto a fixed QMD budget:

\[
\pi^\Theta
=
\arg\min_{\pi\in\Delta_N} KL(\pi\|q_\Theta)
\quad\text{s.t.}\quad
\sum_i \pi_i d_i^{QMD}\le d_{\min}(Q,\mu)+C.
\]

This is implemented by exponential tilting with a Lagrange multiplier.

## 3. Certificate Boundary

The certificate boundary is part of the dispatcher itself. The final policy is
the distribution passed to the simulator, trainer, and evaluator.

For the index model, the boundary is an exact KL projection with no additional
tail controller in the forward path. Certification must be explicit at inference
time.

## 4. Diagnostics

Every forward pass emits auditable diagnostics. The core quantities are:

1. certified arrival coordinate,
2. proposal arrival coordinate,
3. final arrival coordinate,
4. certificate slack,
5. usage raw and usage final,
6. projection multiplier when applicable,
7. projection activation flag,
8. correction magnitude,
9. policy entropy,
10. selected resource summary.

These diagnostics are not optional logging. They are part of the architecture
definition.

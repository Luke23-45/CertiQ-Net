# Certificate Layer

## 1. Certificate Quantity

For any policy \(\pi\) and any dispatch geometry \(d(Q,\mu)\), define the
arrival coordinate
\[
A_\pi(Q,\mu)=\sum_i \pi_i(Q,\mu,\xi)\,d_i(Q,\mu).
\]

The certificate compares the proposed arrival coordinate to a state-dependent
budget \(B(Q,\mu)\).

For the index model, the canonical geometry is the quadratic drift index
\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i},
\qquad
B(Q,\mu)=\min_i d_i^{QMD}(Q,\mu)+C.
\]

## 2. Exact KL Projection

The learned index model certifies by solving
\[
\pi^\star
=
\arg\min_{\pi\in\Delta_N} KL(\pi\|q_\Theta)
\quad\text{s.t.}\quad
A_\pi(Q,\mu)\le B(Q,\mu).
\]

When the constraint is active, the solution has the form
\[
\pi_i^\star(\nu)
=
\frac{q_i \exp(-\nu d_i)}{\sum_j q_j \exp(-\nu d_j)},
\qquad \nu\ge 0.
\]

The Lagrange multiplier \(\nu\) is found by bisection. If the proposal already
satisfies the budget, then \(\nu=0\) and the policy is unchanged.

The operator is exact at the level of the implemented numerical tolerance. It
is not a smooth approximation of the constraint set.

The budget \(C\) is a fixed certificate slack used by the current canonical
implementation. It is a constant scalar, not a state-dependent slack function.



## 3. Certificate Modes

The index model does not use a separate tail-controller fallback in its forward
path. The only certification mode is the exact KL projection.

## 4. Diagnostics Contract

Every certified forward pass must report:

1. \(A_{\mathrm{cert}}\),
2. \(A_{\mathrm{proposal}}\),
3. \(A_{\mathrm{final}}\),
4. \(m(Q)\),
5. \(B(Q)\),
6. \(B(Q)-A_{\mathrm{final}}\),
7. usage raw,
8. usage final,
9. usage cap,
10. fallback flag,
11. correction magnitude,
12. policy entropy,
13. selected resource,
14. projection multiplier,
15. projection activation flag,
16. projection slack.

These diagnostics are part of the formal contract, not optional experiment
metadata.

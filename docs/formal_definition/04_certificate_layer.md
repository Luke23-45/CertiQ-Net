# Certificate Layer

## 1. Arrival Coordinate

Let \(d(Q,\mu)=(d_1(Q,\mu),\ldots,d_N(Q,\mu))\) be a dispatch geometry. For any
policy \(\pi\in\Delta_N\), define the arrival coordinate

\[
A_\pi(Q,\mu)=\sum_{i=1}^N \pi_i(Q,\mu,\xi)\,d_i(Q,\mu).
\]

The certificate compares \(A_\pi(Q,\mu)\) against a state-dependent budget
\(B(Q,\mu)\).

For the canonical index model,

\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i},
\qquad
B(Q,\mu)=\min_{1\le i\le N} d_i^{QMD}(Q,\mu)+C,
\]

where \(C\ge 0\) is a fixed slack constant.

## 2. Certified Policy

Given a raw proposal \(q_\Theta(x)\in\Delta_N\), the certified policy is

\[
\pi^\star(x)
=
\arg\min_{\pi\in\Delta_N} \mathrm{KL}(\pi\|q_\Theta(x))
\quad\text{subject to}\quad
A_\pi(Q,\mu)\le B(Q,\mu).
\]

## 3. Closed-Form Projection

Assume \(q_{\Theta,i}(x)>0\) for every \(i\). If the feasible set is nonempty,
the projection problem has a unique minimizer. When the constraint is active,
the solution has the form

\[
\pi_i^\star(\nu)
=
\frac{q_{\Theta,i}(x)\exp(-\nu d_i(Q,\mu))}
{\sum_{j=1}^N q_{\Theta,j}(x)\exp(-\nu d_j(Q,\mu))},
\qquad \nu\ge 0,
\]

with \(\nu\) chosen so that

\[
\sum_{i=1}^N \pi_i^\star(\nu)\,d_i(Q,\mu)=B(Q,\mu).
\]

If the raw proposal already satisfies the budget, then \(\nu=0\) and
\(\pi^\star=q_\Theta\).

## 4. Solver Status And Fallback

Let the solver status be a discrete variable

\[
s(x)\in\{\textsf{success},\textsf{infeasible},\textsf{tol\_violation}\}.
\]

Let \(\varepsilon_{\mathrm{proj}}>0\) be the projection tolerance and let
\(\pi_{\mathrm{base}}(x)\) denote a certified fallback policy.

The runtime policy is

\[
\pi^{\mathrm{rt}}(x)
=
\begin{cases}
\pi^\star(x), & \text{if } s(x)=\textsf{success} \text{ and } A_{\pi^\star}(x)\le B(x)+\varepsilon_{\mathrm{proj}},\\
\pi_{\mathrm{base}}(x), & \text{otherwise}.
\end{cases}
\]

If the fallback policy is admissible for every state, then the runtime policy
is defined for every state and is admissible whenever either the certificate
layer succeeds or the fallback policy is used.

## 5. Diagnostics

Every certified forward pass must report:

1. \(A_{\mathrm{proposal}}\),
2. \(A_{\mathrm{certified}}\),
3. the selected resource,
4. \(B(Q,\mu)\),
5. \(B(Q,\mu)-A_{\mathrm{certified}}\),
6. the projection multiplier \(\nu\),
7. the projection activation flag,
8. the correction magnitude \(\|\pi^\star-q_\Theta\|\),
9. the policy entropy,
10. the solver status,
11. the fallback flag,
12. the raw proposal \(q_\Theta\),
13. the certified policy \(\pi^\star\).

The diagnostics record is part of the formal interface of the certificate
layer.

## 6. Proposal-Level Certificate Penalty

Let \(A_{q_\Theta}(x)\) denote the arrival coordinate of the raw proposal and
\(B(x)\) the corresponding budget. Define the proposal-level certificate
penalty by

\[
\mathcal L_{\mathrm{cert}}^{\mathrm{prop}}(\Theta)
=
\mathbb E\!\left[\bigl(A_{q_\Theta}(x)-B(x)\bigr)_+^2\right].
\]

This penalty can provide a nontrivial training signal for the proposal module
whenever the raw proposal violates the budget on a set of positive measure.

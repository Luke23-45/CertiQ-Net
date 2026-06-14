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

## 4. Fallback Contract

If the projection solver fails, does not converge, or returns a violation
larger than the numerical tolerance \(\varepsilon_{\mathrm{proj}}>0\), the
runtime must replace the result with a fallback policy \(\pi_{\mathrm{base}}\).

This condition is an implementation contract rather than a convex-analytic
statement.

## 5. Diagnostics

Every certified forward pass must report:

1. \(A_{\mathrm{proposal}}\),
2. \(A_{\mathrm{certified}}\),
3. \(B(Q,\mu)\),
4. \(B(Q,\mu)-A_{\mathrm{certified}}\),
5. the projection multiplier \(\nu\),
6. the projection activation flag,
7. the correction magnitude \(\|\pi^\star-q_\Theta\|\),
8. the policy entropy,
9. the solver status,
10. the fallback flag.

The diagnostics record is part of the formal interface of the certificate
layer.

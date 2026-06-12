# Certificate Layer

## 1. Certificate Quantity

For any policy \(\pi\) and any dispatch geometry \(d(Q,\mu)\), define the
arrival coordinate
\[
A_\pi(Q,\mu)=\sum_i \pi_i(Q,\mu,\xi)\,d_i(Q,\mu).
\]

The certificate compares the proposed arrival coordinate to a state-dependent
budget \(B(Q,\mu)\).

For the legacy dispatcher, the canonical geometry is
\[
y_i^\beta(Q)=Q_i/\mu_i^\beta,
\qquad
B(Q,\mu)=\min_i y_i^\beta(Q)+C.
\]

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

## 3. Scalar Usage Cap

The legacy dispatcher does not solve the full KL problem in its main forward
path. Instead it computes a scalar usage cap between a certified base policy
and a proposal policy.

Let
\[
A_{\mathrm{cert}}(Q,\mu)=A_{p^{\mathrm{cert}}}(Q,\mu),
\qquad
A_{\mathrm{prop}}(Q,\mu)=A_{p^{\mathrm{prop}}}(Q,\mu).
\]

If \(A_{\mathrm{prop}}\le A_{\mathrm{cert}}\), the proposal is no worse under
the certificate coordinate and may be used fully. Otherwise the safe usage cap
is
\[
\eta_{\mathrm{safe}}
=
\left[
\frac{B-A_{\mathrm{cert}}}{A_{\mathrm{prop}}-A_{\mathrm{cert}}}
\right]_0^1.
\]

The final usage is the minimum of the raw usage preference and the safe cap.

## 4. Fallback Mode

The fallback mode is a tail-safe mode for the legacy dispatcher.

Let
\[
S_\beta(Q)=\sum_i Q_i/\mu_i^\beta.
\]

For a chosen radius \(R_{\mathrm{cert}}\), the dispatcher uses the learned
proposal only when
\[
S_\beta(Q)\le R_{\mathrm{cert}}.
\]

Outside that region, the final policy equals the certified base policy.

This is conservative by construction. It makes the learned policy a finite
region perturbation of a stabilizing base policy.

## 5. Certificate Modes

The supported modes are:

1. `projection`
   - legacy scalar usage cap,
   - learned proposal is mixed with the base policy.
2. `fallback`
   - learned proposal is disabled outside a tail radius.
3. `uncertified`
   - raw proposal is returned for ablation only.
4. exact KL projection
   - used by `CertiQIndexModel`.

Only the explicitly chosen mode is guaranteed at inference time.

## 6. Diagnostics Contract

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

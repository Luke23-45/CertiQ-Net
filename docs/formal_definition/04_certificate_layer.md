# Certificate Layer

## 1. Certificate Quantity

For any policy \(\pi\), define:

\[
A_\pi(Q)=\sum_i \pi_i(Q,\mu,\xi)y_i(Q).
\]

The certified envelope is:

\[
B(Q)=m(Q)+C.
\]

The certificate condition is:

\[
A_{\pi^\Theta}(Q)\le B(Q).
\]

For the current z2 CTMC theorem, \(C=C_B\), the exact constant from the
analytic backbone proof.

## 2. Admissible Dispatch Set

At state \(Q\), define the admissible simplex:

\[
\mathcal P_{\mathrm{cert}}(Q)
=
\{\pi\in\Delta_N:A_\pi(Q)\le B(Q)\}.
\]

The certificate operator must return:

\[
\pi^\Theta(Q,\mu,\xi)\in\mathcal P_{\mathrm{cert}}(Q).
\]

This makes certification an action-space constraint, not an auxiliary loss.

## 3. Tail Fallback Operator

Define:

\[
S_\beta(Q)=\sum_i Q_i/\mu_i^\beta.
\]

For finite \(R_{\mathrm{cert}}\), the fallback operator sets:

\[
\eta=
\eta_{\mathrm{raw}}\mathbf 1_{\{S_\beta(Q)\le R_{\mathrm{cert}}\}}.
\]

Then:

\[
\pi^\Theta
=
(1-\eta)p^{\mathrm{cert}}+\eta p^\Theta_{\mathrm{prop}}.
\]

When \(S_\beta(Q)>R_{\mathrm{cert}}\), the policy equals the certified base
exactly.

This operator is conservative. It certifies by making the learned proposal a
finite-region perturbation of the base policy.

## 4. Drift-Envelope Projection Operator

Compute:

\[
A_{\mathrm{cert}}(Q)=A_{p^{\mathrm{cert}}}(Q),
\]

\[
A_{\mathrm{prop}}(Q)=A_{p^\Theta_{\mathrm{prop}}}(Q).
\]

If \(A_{\mathrm{prop}}(Q)\le A_{\mathrm{cert}}(Q)\), the proposal is no worse
than the base under the envelope coordinate and may be fully allowed:

\[
\eta_{\mathrm{safe}}=1.
\]

Otherwise:

\[
\eta_{\mathrm{safe}}
=
\left[
\frac{B(Q)-A_{\mathrm{cert}}(Q)}
{A_{\mathrm{prop}}(Q)-A_{\mathrm{cert}}(Q)}
\right]_0^1.
\]

The final usage is:

\[
\eta=\min(\eta_{\mathrm{raw}},\eta_{\mathrm{safe}}).
\]

The final policy:

\[
\pi^\Theta=(1-\eta)p^{\mathrm{cert}}+\eta p^\Theta_{\mathrm{prop}}
\]

satisfies:

\[
A_{\pi^\Theta}(Q)\le B(Q).
\]

This operator certifies by pointwise envelope preservation.

## 5. Diagnostics Contract

Every certified forward pass must emit:

1. \(A_{\mathrm{cert}}\),
2. \(A_{\mathrm{prop}}\),
3. \(A_{\pi^\Theta}\),
4. \(m(Q)\),
5. \(B(Q)\),
6. \(B(Q)-A_{\pi^\Theta}\),
7. raw proposal usage,
8. final proposal usage,
9. projection cap when applicable,
10. fallback indicator when applicable,
11. residual magnitude,
12. final policy entropy,
13. selected resource or action distribution summary.

These are architecture diagnostics. They are not optional experiment metadata.

## 6. Invalid Certificate Patterns

The following do not certify the architecture:

1. adding a drift penalty without exact enforcement,
2. using stale probabilities for projection,
3. projecting one policy and dispatching another,
4. using smooth fallback at inference without a separate proof,
5. claiming CTMC stability for adapters that violate the CTMC model,
6. allowing approximate projection without a numerical tolerance contract and
   proof that the resulting error preserves drift.


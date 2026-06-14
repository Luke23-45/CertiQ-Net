# Reflected Pressure Supplement

This file records a supplementary reflected-pressure extension of the CertiQ
proposal layer. It is not part of the canonical index model, but it is a
formalized variant of the proposal-control mechanism.

## 1. Pressure State

Let

\[
p\in\mathbb R_+^N
\]

denote a nonnegative resource-pressure state.

The purpose of \(p\) is to encode recent over-selection or routing pressure
separately from the physical backlog \(Q\).

## 2. Reflected Proposal

Let \(u_i^{\mathrm{cert}}(Q,\mu)\) denote a certified base score and
\(r_i^\Theta(Q,\mu,\xi)\) a learned residual correction. A pressure-aware
proposal may be defined by

\[
\mathcal A_\Theta^{\mathrm{press}}(Q,\mu,\xi,p)
=
\operatorname{softmax}\!\left(
u^{\mathrm{cert}}(Q,\mu)+r^\Theta(Q,\mu,\xi)-\rho p
\right),
\qquad \rho>0.
\]

The pressure term acts as a monotone penalty on resources with large pressure.

## 3. Reflected Update

Let \(\eta_p>0\) and \(\delta\in[0,1]\). A reflected update may be defined by

\[
p^+
=
\Pi_{\mathbb R_+^N}\!\left((1-\delta)p+\eta_p(\hat d-\bar d)\right),
\]

where \(\hat d\) denotes realized or expected dispatch mass and \(\bar d\) is a
target mass vector, for example \(\bar d_i=\mu_i/\sum_j \mu_j\).

## 4. Certificate Interaction

The certificate operator remains unchanged:

\[
\pi^\Theta
=
\mathcal C_Q\!\left(
\mathcal A_\Theta^{\mathrm{press}}(Q,\mu,\xi,p),
\mathcal B(Q,\mu)
\right).
\]

Pressure may influence the proposal, but it does not replace the certificate.
Certification still requires

\[
A_{\pi^\Theta}(Q,\mu)\le B(Q,\mu).
\]

## 5. Preservation Statement

If \(\mathcal C_Q\) returns an admissible policy for every proposal
distribution in its domain, then replacing the proposal by a reflected-pressure
proposal preserves certification.

This statement isolates pressure as a proposal-level control mechanism.

## 6. Supplementary Status

This extension is supplementary. The canonical index model defined in the
preceding chapters does not require a pressure state in its forward path.

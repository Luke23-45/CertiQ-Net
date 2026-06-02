# CertiQ Dispatcher Architecture

## 1. Canonical Equation

The z3 architecture is:

\[
\pi^\Theta(Q,\mu,\xi)
=
\mathcal C_Q\left(
\mathcal A_\Theta(Q,\mu,\xi),
\mathcal B(Q,\mu)
\right).
\]

This equation defines the architecture.

\(\mathcal B\) is the base certified geometry. \(\mathcal A_\Theta\) is the
learned assignment proposal. \(\mathcal C_Q\) is the state-dependent
certificate operator. The final policy \(\pi^\Theta\) is the only distribution
used for dispatch.

## 2. Resource Coordinates

The architecture begins by computing capacity-normalized workload:

\[
y_i(Q)=Q_i/\mu_i^\beta,
\qquad
m(Q)=\min_i y_i(Q).
\]

These coordinates are not optional metadata. They are the Lyapunov geometry
used by the certificate layer.

## 3. Base Certified Geometry

The base geometry produces logits:

\[
u_i^{\mathrm{cert}}
=
\gamma\log\mu_i
-
\alpha\frac{Q_i+c}{\mu_i^\beta}.
\]

The base policy is:

\[
p^{\mathrm{cert}}
=
\operatorname{softmax}(u^{\mathrm{cert}}).
\]

For the z2 CTMC theorem, this is the analytic backbone and satisfies:

\[
A_{p^{\mathrm{cert}}}(Q)\le m(Q)+C.
\]

In z3 language, the important object is not the old name "backbone". The
important object is a certified base distribution with an arrival-envelope
constant.

## 4. Learned Assignment Proposal

The proposal layer builds an equivariant score correction from resource-local
and global information.

Local features:

\[
s_i=[
\log(1+Q_i),
\log\mu_i,
y_i,
\mu_i/\Lambda,
\xi_i
].
\]

Shared local encoding:

\[
z_i^0=f_{\mathrm{loc}}(s_i).
\]

Invariant global context:

\[
g=f_{\mathrm{glob}}(\{z_i^0\}_{i=1}^N).
\]

Equivariant resource update:

\[
z_i^1=f_{\mathrm{res}}([z_i^0,g]).
\]

Proposal correction:

\[
r_i^\Theta=R_{\max}\tanh(h(z_i^1)).
\]

Proposal policy:

\[
p^\Theta_{\mathrm{prop}}
=
\operatorname{softmax}(u^{\mathrm{cert}}+r^\Theta).
\]

The proposal is anchored to the certified base logits. It is not a free
black-box dispatcher.

## 5. Raw Preference For Learning

The proposal layer may also produce a raw usage preference:

\[
\eta_{\mathrm{raw}}\in[0,1].
\]

\(\eta_{\mathrm{raw}}\) is not a certificate. It is only the learned desire to
use the proposal.

## 6. Candidate Mixture

The candidate mixture is:

\[
\pi_\eta
=
(1-\eta)p^{\mathrm{cert}}
+
\eta p^\Theta_{\mathrm{prop}}.
\]

The certificate operator chooses or modifies \(\eta\), or more generally
projects the candidate policy, so the final dispatch distribution is
admissible.

## 7. Certificate Operator

The certificate operator receives:

1. current state \(Q\),
2. base distribution \(p^{\mathrm{cert}}\),
3. proposal distribution \(p^\Theta_{\mathrm{prop}}\),
4. raw usage preference \(\eta_{\mathrm{raw}}\),
5. certificate constants and diagnostics.

It returns:

1. final policy \(\pi^\Theta\),
2. final gate or projection variables,
3. certificate diagnostics.

The final policy must be the distribution used by the simulator, trainer,
evaluator, and deployment path.

## 8. Architecture Presets

z3 may expose presets:

1. **CertiQ Dispatcher-F**: tail fallback certificate operator,
2. **CertiQ Dispatcher-P**: drift-envelope projection operator,
3. **CertiQ Dispatcher-X**: uncertified ablation.

These are presets of one architecture, not separate architectures.

The certified presets must share the same forward interface and diagnostics
contract.

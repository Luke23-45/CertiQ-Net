# CertiQ Dispatcher Architecture

## 1. Certified Index Model

The canonical learned dispatcher is the CertiQ index model. Its policy takes
the form

\[
\pi^\Theta(x)
=
\mathcal C_x\!\left(\mathcal A_\Theta(x),\mathcal B(Q,\mu)\right),
\qquad
x=(Q,\mu,\xi).
\]

The proposal map \(\mathcal A_\Theta\) produces a raw distribution over
resources, and the certificate map \(\mathcal C_x\) converts that proposal into
an admissible policy.

## 2. Learned Index Representation

The index model uses the quadratic-min-drift geometry

\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

Let \(r_i^\Theta(Q,\mu,\xi)\) denote a learned residual correction. The learned
index is

\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

The raw proposal is the softmax of the negative index:

\[
q_\Theta(x)=\operatorname{softmax}\!\left(-\hat I_\Theta(x)/\tau\right),
\qquad \tau>0.
\]

## 3. Shared-Feature Realization

One admissible realization of \(\mathcal A_\Theta\) is a shared-resource
encoder:

\[
h_i=\phi([Q_i,\mu_i,\xi_i]),\qquad
s=\rho(h_1,\ldots,h_N),\qquad
u_i=\psi([h_i,s]),\qquad
r_i^\Theta=g(u_i),
\]

with

\[
\hat I_i(Q,\mu,\xi)=z_i=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi),
\qquad
q_\Theta(x)=\operatorname{softmax}(-z/\tau).
\]

Here \(\phi\), \(\psi\), and \(g\) are shared across resources, while \(\rho\)
is permutation invariant. The scalar head \(g\) predicts the residual
correction, and \(z\) is the resulting learned index.

### 3.1 Structural Refinement Head

The architecture may include an additional head

\[
r_{\mathrm{struct}}^\Theta:\mathcal X\to\mathbb R^k,
\]

which predicts low-dimensional structural quantities such as slack, regime, or
calibration features. This head is used inside the proposal module and does not
replace the certificate layer.

## 4. Certificate Boundary

The certificate boundary is explicit and state dependent. The final policy is
the certified distribution returned by \(\mathcal C_x\), not the raw proposal.
This boundary is part of the dispatcher definition.

## 5. Diagnostics

Every forward pass must expose a diagnostics record containing at least:

1. the raw proposal \(q_\Theta\),
2. the certified policy \(\pi^\Theta\),
3. the selected resource,
4. the proposal arrival coordinate,
5. the certified arrival coordinate,
6. the certificate budget,
7. the certificate slack,
8. the projection multiplier,
9. the projection activation flag,
10. the correction magnitude,
11. the policy entropy,
12. the solver status,
13. the fallback flag.

The diagnostics record is part of the formal interface of the model.

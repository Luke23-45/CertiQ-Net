# CertiQ-Net Architecture

This file defines the proposed neural architecture. The architecture is not an
unconstrained neural controller. It is a certified-residual policy: the neural
part may improve decisions, but the analytic base controller remains available
as an enforceable fallback.

## 1. Design Requirements

CertiQ-Net must satisfy:

1. permutation equivariance over server labels,
2. smooth routing probabilities,
3. compatibility with differentiable simulation,
4. explicit tail fallback to a certifiable policy,
5. support for heterogeneous service rates,
6. no dependence on fixed \(N\) unless declared.

## 2. Analytic Base Controller

For parameters

\[
\phi=(\alpha,\beta,\gamma,c),
\qquad
\alpha>0,\ \beta>0,\ c\ge 0,\ \gamma\in\mathbb R,
\]

define

\[
u_i^{\mathrm{base}}(Q;\phi)
=
\gamma\log\mu_i
-
\alpha\frac{Q_i+c}{\mu_i^\beta},
\]

and

\[
p_i^{\mathrm{base}}(Q;\phi)
=
\operatorname{softmax}_i(u^{\mathrm{base}})
=
\frac{\exp(u_i^{\mathrm{base}})}
{\sum_j\exp(u_j^{\mathrm{base}})}.
\]

This is the analytic base controller. The parameters \(\phi\) may be fixed,
learned inside constrained ranges, or produced by a context network.

Constrained parameterization:

\[
\alpha=\alpha_{\min}+\operatorname{softplus}(a),
\qquad
\beta=\beta_{\min}+\operatorname{softplus}(b),
\qquad
c=\operatorname{softplus}(d).
\]

\(\gamma\) may remain unconstrained or be bounded by
\(\gamma=\gamma_{\max}\tanh(g)\) for numerical control.

## 3. Server Encoder

For each server, build a local feature vector

\[
z_i^0
=
\psi_{\mathrm{in}}(Q_i,\mu_i,\xi_i)
\in\mathbb R^d.
\]

The minimal version uses an MLP applied independently to every server:

\[
z_i^0
=
\mathrm{MLP}_{\mathrm{local}}
\left(
\log(1+Q_i),
\log\mu_i,
Q_i/\mu_i^\beta,
\mu_i/\Lambda
\right).
\]

This preserves permutation equivariance because the same map is used for each
server.

## 4. Context Aggregation

Define a permutation-invariant global context

\[
g
=
\rho
\left(
\sum_{i=1}^N a_i z_i^0
\right),
\]

where

\[
a_i
=
\frac{\exp(r(z_i^0))}{\sum_j\exp(r(z_j^0))}
\]

is an optional attention pooling weight. Mean or sum pooling is also allowed.

The context vector should encode:

- load shape,
- heterogeneity shape,
- bottleneck servers,
- normalized capacity distribution.

## 5. Equivariant Residual Network

Each server receives the shared context:

\[
z_i^1
=
\mathrm{MLP}_{\mathrm{res}}
\left([z_i^0,g]\right).
\]

The neural residual logit is

\[
r_i^\Theta(Q,\xi)=w^\top z_i^1+b.
\]

The unconstrained neural proposal is

\[
p_i^{\mathrm{nn}}(Q,\xi)
=
\operatorname{softmax}_i
\left(
u_i^{\mathrm{base}}(Q;\phi_\Theta(\xi))
+
r_i^\Theta(Q,\xi)
\right).
\]

Using the base-controller logit inside the neural proposal is intentional: it
makes the neural module learn corrections to a meaningful physical energy
rather than learning dispatch from scratch.

## 6. Certificate Gate

The final policy is a mixture:

\[
\pi_i^\Theta(Q,\xi)
=
(1-\eta_\Theta(Q,\xi))
p_i^{\mathrm{base}}(Q;\phi_\Theta(\xi))
+
\eta_\Theta(Q,\xi)
p_i^{\mathrm{nn}}(Q,\xi).
\]

The gate satisfies

\[
0\le \eta_\Theta(Q,\xi)\le \eta_{\max}\le 1.
\]

### Conservative tail gate

The recommended first implementation is

\[
\eta_\Theta(Q,\xi)
=
\eta_{\max}\sigma(h_\Theta(Q,\xi))
\cdot
\mathbf 1_{\{S(Q)\le R_{\mathrm{cert}}\}},
\]

with

\[
S(Q)=\sum_i Q_i/\mu_i^\beta.
\]

Equivalently, outside the certified radius,

\[
S(Q)>R_{\mathrm{cert}}
\quad\Longrightarrow\quad
\eta_\Theta(Q,\xi)=0.
\]

This forces the policy to equal the analytic base controller in the tail, which
is the most realistic path to a first stability theorem for the neural
architecture.

### Smooth tail gate

For differentiable simulation, replace the hard indicator by

\[
\chi_R(Q)
=
\sigma\left(\tau(R_{\mathrm{cert}}-S(Q))\right),
\]

and set

\[
\eta_\Theta(Q,\xi)
=
\eta_{\max}\sigma(h_\Theta(Q,\xi))\chi_R(Q).
\]

The hard-gated policy is easier to certify. The smooth-gated policy is easier
to train. The project should train with smooth gates and certify either the
hard limit or an explicit smooth-envelope bound.

## 7. Certificate Projection Layer

A more ambitious variant projects the gate onto a drift-safe interval.

For the weighted quadratic Lyapunov function

\[
V(Q)=\frac12\sum_i Q_i^2/\mu_i^\beta,
\]

the arrival part of the generator depends on

\[
A_\pi(Q)
=
\sum_i \pi_i(Q,\xi)\frac{Q_i}{\mu_i^\beta}.
\]

For the mixture policy,

\[
A_{\pi^\Theta}(Q)
=
(1-\eta)A_{\mathrm{base}}(Q)+\eta A_{\mathrm{nn}}(Q).
\]

If a certified upper envelope \(B(Q)\) is required, enforce

\[
A_{\pi^\Theta}(Q)\le B(Q).
\]

When \(A_{\mathrm{nn}}(Q)>A_{\mathrm{base}}(Q)\), this implies

\[
\eta
\le
\frac{B(Q)-A_{\mathrm{base}}(Q)}
{A_{\mathrm{nn}}(Q)-A_{\mathrm{base}}(Q)}.
\]

The projected gate is

\[
\eta_{\mathrm{proj}}
=
\min\left\{\eta_{\mathrm{raw}},\eta_{\mathrm{safe}}(Q)\right\}.
\]

This is the architecture's key certification mechanism. It converts a neural
proposal into a policy that obeys a drift constraint at the point of action.

## 8. Model Classes

### CertiQ-Net-S

Small certified model:

- fixed base-controller parameters,
- local encoder,
- residual logits,
- compact-support gate.

This is the first implementation target.

### CertiQ-Net-P

Projected model:

- fixed or learned base-controller parameters,
- residual logits,
- pointwise projection by the Lyapunov arrival envelope.

This is the first serious theorem target.

### CertiQ-Net-M

Meta model:

- context network predicts \(\phi_\Theta(\xi)\),
- residual network adapts across systems,
- certificate layer enforces tail or drift envelope.

This is the long-term architecture, not the first theorem target.

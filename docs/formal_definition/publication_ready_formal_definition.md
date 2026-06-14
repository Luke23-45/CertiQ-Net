# CertiQ Formal Definition

## 1. Purpose

This document gives a standalone formal specification of the CertiQ dispatch
family. The central object is a certified dispatch operator of the form

\[
\pi^\Theta(x)=\mathcal C_x\!\left(\mathcal A_\Theta(x),\mathcal B(x)\right),
\]

where \(\mathcal A_\Theta\) is the learned proposal map, \(\mathcal B\) is the
base geometry or budget map, and \(\mathcal C_x\) is the certificate operator
at state \(x\).

The specification below defines the state space, the geometry layer, the
proposal layer, the certificate layer, the training objective, and the
supporting references. It also records a supplementary reflected-pressure
extension.

## 2. Dispatch Primitive

CertiQ models online assignment of arriving work to heterogeneous resources.
The primitive is a map from state to a distribution over resources, together
with an explicit certificate that constrains the resulting distribution.

The canonical structure is

\[
\text{dispatch}=\text{proposal}\;\text{followed by}\;\text{certification}.
\]

The final policy is the certified policy returned by the certificate operator.

## 3. System Model

### 3.1 Resources And Context

Let \(N\ge 1\) denote the number of resources. Resource \(i\) has service
capacity \(\mu_i>0\), and

\[
\mu=(\mu_1,\ldots,\mu_N)\in\mathbb R_+^N.
\]

Each resource may also carry context

\[
\xi_i\in\mathbb R^{d_\xi},
\qquad
\xi=(\xi_1,\ldots,\xi_N).
\]

The context vector is an exogenous feature of the resource. It may encode
resource attributes such as location, compatibility, cost, or other domain
information.

### 3.2 Queue State And Arrivals

The system state is

\[
Q=(Q_1,\ldots,Q_N)\in\mathbb Z_+^N,
\]

where \(Q_i\) is the number of jobs currently assigned to resource \(i\).

In the primary continuous-time Markov chain model, arrivals form a Poisson
process with rate \(\lambda>0\). The total service capacity is

\[
\Lambda=\sum_{i=1}^N \mu_i,
\]

and the subcritical load condition is

\[
\lambda<\Lambda.
\]

### 3.3 Action Space

At each arrival epoch, the dispatcher outputs a distribution

\[
\pi(Q,\mu,\xi)\in\Delta_N,
\qquad
\Delta_N=\left\{p\in\mathbb R_+^N:\sum_{i=1}^N p_i=1\right\}.
\]

The arriving job is assigned to resource \(i\) with probability \(\pi_i\).

### 3.4 Service Dynamics

When \(Q_i>0\), resource \(i\) completes jobs at rate \(\mu_i\). Under policy
\(\pi\), the generator acting on bounded \(f:\mathbb Z_+^N\to\mathbb R\) is

\[
(\mathcal L_\pi f)(Q)
=
\lambda\sum_{i=1}^N \pi_i(Q,\mu,\xi)\bigl[f(Q+e_i)-f(Q)\bigr]
+\sum_{i=1}^N \mu_i\,\mathbf 1_{\{Q_i>0\}}\bigl[f(Q-e_i)-f(Q)\bigr].
\]

The policy controls only the routing term.

### 3.5 Performance Objective

The long-run average cost of policy \(\pi\) is

\[
J(\pi)
=
\limsup_{T\to\infty}
\frac1T
\mathbb E_\pi\!\left[\int_0^T c(Q(t),\mu,\xi)\,dt\right].
\]

The canonical cost is total backlog,

\[
c(Q)=\sum_{i=1}^N Q_i.
\]

Weighted costs may be used when the weights are positive and bounded.

### 3.6 Delay-Aligned Geometry

Two queueing geometries are used throughout the CertiQ family:

\[
d_i^{SED}(Q,\mu)=\frac{Q_i+1}{\mu_i},
\qquad
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

The first is the shortest-expected-delay geometry. The second is the
quadratic-min-drift geometry used by the index model.

### 3.7 Adapter Specification

A domain adapter maps an application into

\[
\mathcal D=(N,\lambda,\mu,\xi,Q,c).
\]

The adapter must specify:

1. the arrival process,
2. the resource set,
3. the meaning of \(\mu_i\),
4. the meaning of \(Q_i\),
5. the service law,
6. the meaning of \(\xi_i\),
7. the performance cost,
8. whether the CTMC assumptions hold exactly.

If the CTMC assumptions do not hold exactly, the model may still be used as an
empirical dispatcher, but the CTMC generator interpretation does not apply
without qualification.

## 4. Architecture Principles

Resource labels are not semantically meaningful. If \(\sigma\) is a
permutation of \(\{1,\dots,N\}\), the policy must satisfy

\[
\pi(\sigma x)=\sigma\pi(x).
\]

This requirement applies to the proposal map, the certificate map, and any
intermediate summary that is reused across resources.

The architecture is decomposed into three mathematical layers.

### 4.1 Geometry Layer

The geometry layer maps state and capacity into resourcewise indices or costs.
Its role is to define the admissibility and comparison scale used by the
certificate.

### 4.2 Proposal Layer

The proposal layer produces a probability distribution over resources from the
observed state. It may depend on queue lengths, service rates, and resource
context.

### 4.3 Certificate Layer

The certificate layer maps a proposal into an admissible distribution. It is
part of the policy definition and not a post hoc regularizer.

The final policy is the certified policy. A model is not certified merely
because its proposal is well trained or its loss is small.

## 5. Certified Index Architecture

The canonical learned dispatcher is the CertiQ index model. Its policy takes
the form

\[
\pi^\Theta(x)=\mathcal C_x\!\left(\mathcal A_\Theta(x),\mathcal B(Q,\mu)\right),
\qquad
x=(Q,\mu,\xi).
\]

The proposal map \(\mathcal A_\Theta\) produces a raw distribution over
resources, and the certificate map \(\mathcal C_x\) converts that proposal into
an admissible policy.

The index model uses the quadratic-min-drift geometry

\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i}.
\]

Let \(r_i^\Theta(Q,\mu,\xi)\) denote a learned residual correction. The learned
index is

\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

The raw proposal is the softmax of the negative learned index:

\[
q_\Theta(x)=\operatorname{softmax}\!\left(-\hat I_\Theta(x)/\tau\right),
\qquad \tau>0.
\]

### 5.1 Shared-Feature Realization

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

### 5.2 Structural Refinement Head

The architecture may include an additional head

\[
r_{\mathrm{struct}}^\Theta:\mathcal X\to\mathbb R^k,
\]

which predicts low-dimensional structural quantities such as slack, regime, or
calibration features. This head is used inside the proposal module and does not
replace the certificate layer.

### 5.3 Certificate Boundary

The certificate boundary is explicit and state dependent. The final policy is
the certified distribution returned by \(\mathcal C_x\), not the raw proposal.
This boundary is part of the dispatcher definition.

## 6. Certificate Layer

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

Given a raw proposal \(q_\Theta(x)\in\Delta_N\), the certified policy is

\[
\pi^\star(x)
=
\arg\min_{\pi\in\Delta_N} \mathrm{KL}(\pi\|q_\Theta(x))
\quad\text{subject to}\quad
A_\pi(Q,\mu)\le B(Q,\mu).
\]

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

### 6.1 Fallback Contract

If the projection solver fails, does not converge, or returns a violation
larger than the numerical tolerance \(\varepsilon_{\mathrm{proj}}>0\), the
runtime must replace the result with a fallback policy \(\pi_{\mathrm{base}}\).

This condition is an implementation contract rather than a convex-analytic
statement.

### 6.2 Diagnostics

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

## 7. Training And Objectives

Training selects parameters. Certification defines admissible actions. The two
roles are distinct and must not be conflated.

Any training objective may improve rollout cost, imitation quality, entropy, or
proposal usefulness, but it does not itself certify the policy.

The training loss may be written as

\[
\mathcal L(\Theta)
=
\omega_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+\omega_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}
+\omega_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+\omega_{\mathrm{cert}}\mathcal L_{\mathrm{cert}},
\]

where the terms denote rollout cost, behavior-cloning or distillation loss,
residual regression, entropy regularization, and certificate regularization.

The certificate penalty may be defined as

\[
\mathcal L_{\mathrm{cert}}
=
\mathbb E\!\left[\bigl(A_{q_\Theta}(Q,\mu)-B(Q,\mu)\bigr)_+^2\right].
\]

If the projection layer is active and numerically exact, this term is zero up
to tolerance.

The learned proposal may be trained as a residual over the QMD geometry:

\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

Soft SED or QMD targets may be used for warm-start supervision. Such targets
are derived from the observed state and service rates rather than from an
external labeled dataset.

Evaluation quantities include performance and certificate measurements.

Performance quantities include average backlog, tail backlog quantiles, and
maximum observed backlog.

Certificate quantities include violation rate, minimum slack, average slack,
projection activation rate, fallback activation rate, and correction magnitude.

No empirical result is certified unless the certificate quantities are
reported.

A consistent curriculum is:

1. verify the geometry and budget definitions,
2. train the proposal against QMD targets,
3. activate exact certificate enforcement,
4. optimize rollout cost under the certified policy,
5. report certificate metrics together with performance metrics.

## 8. Research References

### 8.1 Set Processing And Permutation Symmetry

Relevant sources:

- Ashish Vaswani et al., "Attention Is All You Need," 2017.
- Manzil Zaheer et al., "Deep Sets," NeurIPS 2017.
- Juho Lee et al., "Set Transformer," ICML 2019.

Supported use:

These works motivate shared local maps, invariant aggregation, and
equivariant set processing.

Unsupported use:

They do not prove queueing stability, certificate validity, or any CertiQ
dispatch guarantee.

### 8.2 Queueing Stability Language

Relevant sources:

- Sean P. Meyn and Richard L. Tweedie, *Markov Chains and Stochastic Stability*.
- Leandros Tassiulas and Anthony Ephremides, "Stability Properties of
  Constrained Queueing Systems and Scheduling Policies for Maximum Throughput
  in Multihop Radio Networks," 1992.

Supported use:

These works provide the standard language for drift, recurrence, and
queue-aware scheduling.

Unsupported use:

They do not by themselves prove the CertiQ index model or its KL projection.

### 8.3 Delay Geometry

Relevant sources:

- Classical shortest-expected-delay routing results for heterogeneous queues.
- Comparative heterogeneous-server routing literature using SED-style rules.

Supported use:

These sources justify SED and QMD as comparison geometries.

Unsupported use:

They do not establish optimality of the learned proposal.

### 8.4 Exact Projection

The KL projection in the CertiQ index model is a state-dependent convex
optimization operator. It supplies a clean certificate boundary, but it does
not itself imply positive recurrence or optimality.

### 8.5 Differentiable Convex Optimization Layers

Relevant sources:

- Brandon Amos and J. Zico Kolter, "OptNet: Differentiable Optimization as a
  Layer in Neural Networks," 2017.
- C. Katyal, "Differentiable Convex Optimization Layers in Neural Architectures:
  Foundations and Perspectives," 2024.

Supported use:

These works support embedding exact convex optimization operators, such as the
CertiQ KL projection, directly into the forward path.

Unsupported use:

They do not provide the queueing geometry, dispatch certificate, or subcritical
load statements required by CertiQ.

### 8.6 Project-Local Proof Package

Relevant sources:

- `docs/z2/formal_math/01_backbone_stability_and_constant.md`
- `docs/z2/formal_math/02_gate_inheritance_theorems.md`

Supported use:

These files support the analytic base envelope and its certified inheritance
statements.

Unsupported use:

They do not automatically extend to arbitrary learned residuals or adapters
that violate the stated CTMC model.

## 9. Supplementary Reflected-Pressure Extension

This section records a supplementary reflected-pressure extension of the CertiQ
proposal layer. It is not part of the canonical index model, but it is a
formalized variant of the proposal-control mechanism.

Let

\[
p\in\mathbb R_+^N
\]

denote a nonnegative resource-pressure state.

The purpose of \(p\) is to encode recent over-selection or routing pressure
separately from the physical backlog \(Q\).

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

Let \(\eta_p>0\) and \(\delta\in[0,1]\). A reflected update may be defined by

\[
p^+
=
\Pi_{\mathbb R_+^N}\!\left((1-\delta)p+\eta_p(\hat d-\bar d)\right),
\]

where \(\hat d\) denotes realized or expected dispatch mass and \(\bar d\) is a
target mass vector, for example \(\bar d_i=\mu_i/\sum_j \mu_j\).

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

If \(\mathcal C_Q\) returns an admissible policy for every proposal
distribution in its domain, then replacing the proposal by a reflected-pressure
proposal preserves certification.

This extension is supplementary. The canonical index model defined above does
not require a pressure state in its forward path.

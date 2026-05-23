# CertiQ-Net z2 End-to-End Formal Definition

Version: z2 strict architecture specification  
Project root: `CertiQ-Net`  
Primary object: Certified Differentiable Dispatcher (CDD)  
Primary certifiable model class: continuous-time stochastic assignment to
heterogeneous capacity-constrained resources

This document defines the standalone z2 research direction: CertiQ-Net is a
general certified differentiable dispatcher for systems in which discrete jobs,
tokens, requests, tasks, or packets must be assigned online to heterogeneous
resources with finite service capacity.

The document is deliberately formal. It separates definitions, theorem targets,
implementation obligations, and empirical claims. Anything not proved here is
marked as a proof obligation or empirical protocol, not as a completed result.
All notation and claims required for the z2 architecture are stated inside this
document; the formal definition is self-contained.

---

## 1. Scientific Contract

### 1.1 Core objective

CertiQ-Net must define a policy class that is simultaneously:

1. differentiable with respect to trainable parameters,
2. permutation equivariant with respect to resource labels,
3. compatible with heterogeneous service capacities,
4. able to include domain-specific context features,
5. guarded by an explicit stability certificate mechanism,
6. auditable at inference time through computable certificate quantities.

The architecture may learn performance-improving corrections, but the learned
part must not be allowed to silently destroy the stability mechanism.

### 1.2 Non-negotiable certification boundary

The primary theorem target is for the CTMC dispatch model defined in Section 2.
Other domains, such as mixture-of-experts routing and multi-robot task
allocation, inherit the theorem only when they are represented as the same
stochastic assignment model with declared arrivals, service completions, state,
and dispatch actions.

CertiQ-Net must not claim:

1. stability for arbitrary neural residuals,
2. stability merely because the policy is smooth,
3. stability merely because the base controller is present somewhere in the
   architecture,
4. domain-general certification without a domain adapter satisfying the formal
   model assumptions,
5. empirical state-of-the-art status before the experiment protocol supports it.

### 1.3 z2 headline

CertiQ-Net is a certified differentiable dispatcher: a permutation-equivariant
neural residual policy attached to a provably stable analytic backbone and
filtered by either a tail fallback gate or a Lyapunov drift-envelope projection.

---

## 2. Abstract Dispatch Model

### 2.1 Resources

There are \(N \ge 1\) resources. Resource \(i\) has service capacity
\(\mu_i > 0\). The capacity vector is

\[
\mu = (\mu_1,\ldots,\mu_N)\in \mathbb R_{>0}^N.
\]

The total capacity is

\[
\Lambda = \sum_{i=1}^N \mu_i.
\]

Each resource may also have a context vector

\[
\xi_i \in \mathbb R^{d_\xi}.
\]

Examples of \(\xi_i\) include expert metadata, hardware type, estimated energy
cost, robot position, channel quality, locality score, or job-resource
compatibility features. The formal stability theorem does not depend on the
meaning of \(\xi_i\); it depends only on the routing probabilities satisfying
the certificate constraints.

### 2.2 Jobs and arrivals

Jobs arrive according to a Poisson process of rate \(\lambda > 0\).

The formal subcritical load assumption is

\[
\lambda < \Lambda.
\]

This condition is necessary for stabilizing a work-conserving parallel-resource
system in the aggregate capacity sense. It is not sufficient for arbitrary
state-dependent dispatch policies.

### 2.3 State

The core state is the resource workload count

\[
Q(t)=(Q_1(t),\ldots,Q_N(t))\in \mathbb Z_+^N,
\]

where \(Q_i(t)\) is the number of active or queued jobs assigned to resource
\(i\).

For the core theorem, \(Q_i\) is an integer queue length. A domain adapter may
replace it by a discretized workload counter only if the service-completion
model remains a CTMC of the form in Section 2.6 or is separately certified.

### 2.4 Actions

At each arrival epoch, a dispatcher observes

\[
(Q,\mu,\xi)
\]

and outputs a probability vector

\[
\pi(Q,\mu,\xi)\in\Delta_N
=
\{p\in\mathbb R_+^N:\sum_{i=1}^N p_i=1\}.
\]

The arriving job is assigned to resource \(i\) with probability
\(\pi_i(Q,\mu,\xi)\).

### 2.5 Service completions

For the primary model, service times at resource \(i\) are exponential with
rate \(\mu_i\), independent across resources and jobs. If \(Q_i(t)>0\), a
completion at resource \(i\) occurs at rate \(\mu_i\), reducing \(Q_i\) by one.

This is the single-server-per-resource interpretation. If a domain uses
multi-server experts, batched execution, processor sharing, deterministic
service, or finite buffers, it must either be reduced to this model or given a
separate generator and certificate.

### 2.6 Controlled CTMC generator

Let \(e_i\) denote the \(i\)-th standard basis vector. For a test function
\(f:\mathbb Z_+^N\to\mathbb R\), the generator under policy \(\pi\) is

\[
(\mathcal L_\pi f)(Q)
=
\lambda \sum_{i=1}^N \pi_i(Q,\mu,\xi)
[f(Q+e_i)-f(Q)]
+
\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
[f(Q-e_i)-f(Q)].
\]

The policy affects only the arrival term. Service dynamics are fixed by
\(\mu\).

For any admissible policy and any state \(Q\), the total jump rate is

\[
\lambda+\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
\le \lambda+\Lambda.
\]

Hence the CTMC has a uniform finite rate bound and is non-explosive. If a
policy additionally satisfies \(\pi_i(Q,\mu,\xi)>0\) for all \(i\) and all
\(Q\), then the chain is irreducible on \(\mathbb Z_+^N\), because every
single-coordinate arrival move \(Q\to Q+e_i\) and every available service move
\(Q\to Q-e_i\) occurs with positive rate.

### 2.7 Objective

The default infinite-horizon objective is average total queue length:

\[
J(\pi)
=
\limsup_{T\to\infty}
\frac1T
\mathbb E_\pi
\left[
\int_0^T \sum_{i=1}^N Q_i(t)\,dt
\right].
\]

Allowed cost generalization:

\[
c(Q,\mu,\xi)=\sum_{i=1}^N w_i(\mu,\xi)Q_i,
\qquad
0 < w_{\min}\le w_i\le w_{\max}<\infty.
\]

The certificate is about stability, not optimality. A policy can be stable and
still have poor cost.

---

## 3. Symmetry Requirements

### 3.1 Permutations

Let \(\sigma\) be a permutation of \(\{1,\ldots,N\}\). Define

\[
(\sigma Q)_i = Q_{\sigma^{-1}(i)},\qquad
(\sigma \mu)_i = \mu_{\sigma^{-1}(i)},\qquad
(\sigma \xi)_i = \xi_{\sigma^{-1}(i)}.
\]

### 3.2 Equivariance definition

A policy \(\pi\) is permutation equivariant if

\[
\pi(\sigma Q,\sigma\mu,\sigma\xi)
=
\sigma \pi(Q,\mu,\xi)
\]

for every \(N\), every permutation \(\sigma\), every valid state \(Q\), every
capacity vector \(\mu\), and every context collection \(\xi\).

This property is required because resource labels are arbitrary. The model may
use resource features, but it must not depend on an externally fixed ordering.

---

## 4. Lyapunov Geometry

### 4.1 Weighted workload coordinate

Fix a parameter \(\beta>0\). Define the weighted queue coordinate

\[
y_i(Q)=\frac{Q_i}{\mu_i^\beta}.
\]

Define

\[
m(Q)=\min_{1\le i\le N} y_i(Q).
\]

### 4.2 Lyapunov function

The default Lyapunov candidate is the weighted quadratic

\[
V(Q)
=
\frac12
\sum_{i=1}^N
\frac{Q_i^2}{\mu_i^\beta}
=
\frac12
\sum_{i=1}^N
\mu_i^\beta y_i(Q)^2.
\]

This function is norm-like when \(N\) and \(\mu_i>0\) are fixed.

### 4.3 Exact generator drift for \(V\)

For the generator in Section 2.6,

\[
V(Q+e_i)-V(Q)
=
\frac{Q_i}{\mu_i^\beta}
+
\frac{1}{2\mu_i^\beta},
\]

and, for \(Q_i>0\),

\[
V(Q-e_i)-V(Q)
=
-
\frac{Q_i}{\mu_i^\beta}
+
\frac{1}{2\mu_i^\beta}.
\]

Therefore

\[
(\mathcal L_\pi V)(Q)
=
\lambda A_\pi(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
+
R_0(Q),
\]

where

\[
A_\pi(Q)
=
\sum_{i=1}^N \pi_i(Q,\mu,\xi)
\frac{Q_i}{\mu_i^\beta}
=
\sum_{i=1}^N \pi_i(Q,\mu,\xi)y_i(Q),
\]

and

\[
R_0(Q)
=
\frac{\lambda}{2}
\sum_{i=1}^N
\pi_i(Q,\mu,\xi)\mu_i^{-\beta}
+
\frac12
\sum_{i=1}^N
\mu_i^{1-\beta}\mathbf 1_{\{Q_i>0\}}.
\]

For fixed \(N,\lambda,\mu,\beta\), \(R_0(Q)\) is uniformly bounded:

\[
0\le R_0(Q)\le C_0
\]

with

\[
C_0
=
\frac{\lambda}{2}\max_i\mu_i^{-\beta}
+
\frac12\sum_{i=1}^N\mu_i^{1-\beta}.
\]

Thus the main certification problem is to control \(A_\pi(Q)\), the expected
weighted workload coordinate of the resource selected by an arrival.

---

## 5. Analytic Backbone

### 5.1 Backbone parameters

The analytic backbone uses parameters

\[
\phi=(\alpha,\beta,\gamma,c),
\qquad
\alpha>0,\quad \beta>0,\quad c\ge 0,\quad \gamma\in\mathbb R.
\]

For numerical and theorem discipline, implementation should use constrained
parameterizations:

\[
\alpha=\alpha_{\min}+\operatorname{softplus}(a),
\]

\[
\beta=\beta_{\min}+\operatorname{softplus}(b),
\]

\[
c=\operatorname{softplus}(d),
\]

with declared \(\alpha_{\min}>0\) and \(\beta_{\min}>0\). Optional numerical
bounding:

\[
\gamma=\gamma_{\max}\tanh(g).
\]

### 5.2 Backbone logits

The base logit for resource \(i\) is

\[
u_i^{\mathrm{base}}(Q,\mu;\phi)
=
\gamma\log\mu_i
-
\alpha
\frac{Q_i+c}{\mu_i^\beta}.
\]

Equivalently,

\[
u_i^{\mathrm{base}}
=
\gamma\log\mu_i
-
\alpha y_i(Q)
-
\alpha c\mu_i^{-\beta}.
\]

### 5.3 Backbone policy

The backbone routing distribution is

\[
p_i^{\mathrm{base}}(Q,\mu;\phi)
=
\frac{\exp(u_i^{\mathrm{base}})}
{\sum_{j=1}^N \exp(u_j^{\mathrm{base}})}.
\]

This policy is smooth in \(Q\) when \(Q\) is viewed through the real-valued
embedding used by differentiable simulation, smooth in \(\phi\), and
permutation equivariant in \((Q,\mu)\).

### 5.4 Backbone certificate obligation

The backbone must satisfy the following envelope for some finite constant
\(C_B\):

\[
A_{\mathrm{base}}(Q)
\le
m(Q)+C_B
\qquad
\text{for all }Q\in\mathbb Z_+^N.
\]

This document treats that statement as a z2 proof obligation. Until it is
proved, the envelope remains an obligation rather than a theorem.

Once the envelope holds,

\[
(\mathcal L_{p^{\mathrm{base}}}V)(Q)
\le
\lambda[m(Q)+C_B]
-
\sum_i\mu_i^{1-\beta}Q_i
+
C_0.
\]

Using \(Q_i=\mu_i^\beta[m(Q)+\delta_i(Q)]\) with
\(\delta_i(Q)\ge 0\),

\[
\sum_i\mu_i^{1-\beta}Q_i
=
\sum_i\mu_i[m(Q)+\delta_i(Q)]
=
\Lambda m(Q)+\sum_i\mu_i\delta_i(Q).
\]

Thus

\[
(\mathcal L_{p^{\mathrm{base}}}V)(Q)
\le
-(\Lambda-\lambda)m(Q)
-
\sum_i\mu_i\delta_i(Q)
+
\lambda C_B+C_0.
\]

For fixed positive \(\mu_i\), this implies a negative drift outside a finite
set whenever \(\lambda<\Lambda\), after converting the right-hand side to a
linear norm bound.

---

## 6. Neural Residual Module

### 6.1 Input features

For each resource \(i\), define local features

\[
s_i(Q,\mu,\xi)
=
\left[
\log(1+Q_i),
\log\mu_i,
y_i(Q),
\frac{\mu_i}{\Lambda},
\xi_i
\right].
\]

If \(\xi_i\) is absent, it is omitted. All scalar inputs should be normalized
in implementation, but normalization constants are not part of the theorem.

### 6.2 Local encoder

Apply a shared map to every resource:

\[
z_i^0 = f_{\mathrm{loc}}(s_i;\Theta_{\mathrm{loc}})\in\mathbb R^d.
\]

The same \(f_{\mathrm{loc}}\) is used for all \(i\). This is required for
equivariance.

### 6.3 Global invariant context

Construct a permutation-invariant summary

\[
g = f_{\mathrm{glob}}(\{z_1^0,\ldots,z_N^0\}).
\]

The default implementation is attention pooling:

\[
a_i
=
\frac{\exp(\rho(z_i^0))}
{\sum_{j=1}^N\exp(\rho(z_j^0))},
\qquad
\bar z=\sum_{i=1}^N a_i z_i^0,
\]

\[
g=\rho_{\mathrm{out}}(\bar z).
\]

Mean pooling is also valid:

\[
g=\rho_{\mathrm{out}}\left(\frac1N\sum_i z_i^0\right).
\]

The chosen pooling method must be stated in experiments.

### 6.4 Equivariant residual logits

Each resource receives the same global context:

\[
z_i^1=f_{\mathrm{res}}([z_i^0,g];\Theta_{\mathrm{res}}).
\]

The raw residual logit is

\[
\tilde r_i^\Theta(Q,\mu,\xi)=w^\top z_i^1+b.
\]

To make the residual certifiable by bounded-perturbation arguments, the
recommended output is bounded:

\[
r_i^\Theta(Q,\mu,\xi)
=
R_{\max}\tanh(\tilde r_i^\Theta(Q,\mu,\xi)),
\]

where \(R_{\max}<\infty\) is declared.

Bounded residual logits are required for any theorem route that argues a
bounded neural perturbation cannot dominate the analytic tail energy.

### 6.5 Neural proposal

The neural proposal is not an independent black-box dispatcher. It is a
backbone-plus-residual softmax:

\[
p_i^{\mathrm{nn}}(Q,\mu,\xi)
=
\operatorname{softmax}_i
\left(
u^{\mathrm{base}}(Q,\mu;\phi)
+
r^\Theta(Q,\mu,\xi)
\right).
\]

This keeps the neural policy anchored to the stable energy geometry.

### 6.6 Equivariance theorem for the residual

If:

1. \(f_{\mathrm{loc}}\) is shared across resources,
2. \(f_{\mathrm{glob}}\) is permutation invariant,
3. \(f_{\mathrm{res}}\) is shared across resources,
4. the final softmax is applied across resource logits,

then \(p^{\mathrm{nn}}\) is permutation equivariant.

Proof sketch: permuting resource labels permutes the multiset
\(\{z_i^0\}\), leaves \(g\) unchanged, permutes the list of residual logits,
and softmax commutes with permutation.

---

## 7. Certificate Gate and Final Policy

### 7.1 Mixture form

The final CertiQ-Net policy is

\[
\pi_i^\Theta(Q,\mu,\xi)
=
(1-\eta^\Theta(Q,\mu,\xi))
p_i^{\mathrm{base}}(Q,\mu;\phi)
+
\eta^\Theta(Q,\mu,\xi)
p_i^{\mathrm{nn}}(Q,\mu,\xi),
\]

where

\[
0\le \eta^\Theta(Q,\mu,\xi)\le 1.
\]

The gate may be scalar for the whole decision. A resource-wise gate is not part
of the default formal architecture because it complicates the arrival envelope
without a clear certification benefit.

### 7.2 Raw learned gate

The raw gate is

\[
\eta_{\mathrm{raw}}^\Theta(Q,\mu,\xi)
=
\eta_{\max}\sigma(h^\Theta(Q,\mu,\xi)),
\qquad
0<\eta_{\max}\le 1.
\]

The gate network \(h^\Theta\) must be permutation invariant. It may use the same
global context \(g\) as the residual:

\[
h^\Theta(Q,\mu,\xi)=w_g^\top g+b_g.
\]

### 7.3 Route A: hard tail fallback

Define the weighted tail size

\[
S_\beta(Q)=\sum_{i=1}^N \frac{Q_i}{\mu_i^\beta}.
\]

For a declared radius \(R_{\mathrm{cert}}<\infty\), the hard fallback gate is

\[
\eta^\Theta(Q,\mu,\xi)
=
\eta_{\mathrm{raw}}^\Theta(Q,\mu,\xi)
\mathbf 1_{\{S_\beta(Q)\le R_{\mathrm{cert}}\}}.
\]

Equivalently,

\[
S_\beta(Q)>R_{\mathrm{cert}}
\quad\Longrightarrow\quad
\pi^\Theta(Q,\mu,\xi)=p^{\mathrm{base}}(Q,\mu;\phi).
\]

This is the conservative first theorem route. The policy is exactly the
certified backbone in the tail.

### 7.4 Route A training surrogate: smooth tail gate

For differentiable training, use

\[
\chi_R(Q)=
\sigma(\tau[R_{\mathrm{cert}}-S_\beta(Q)]),
\qquad \tau>0,
\]

and

\[
\eta_{\mathrm{smooth}}^\Theta
=
\eta_{\mathrm{raw}}^\Theta\chi_R(Q).
\]

The smooth gate is a training surrogate unless a separate proof shows it
satisfies a drift envelope. Certification should be stated for the hard gate or
for an explicitly verified smooth envelope.

### 7.5 Route B: drift-envelope projection

Define

\[
A_{\mathrm{base}}(Q)
=
\sum_i p_i^{\mathrm{base}}(Q,\mu;\phi)y_i(Q),
\]

\[
A_{\mathrm{nn}}(Q)
=
\sum_i p_i^{\mathrm{nn}}(Q,\mu,\xi)y_i(Q).
\]

For the mixture,

\[
A_{\pi^\Theta}(Q)
=
(1-\eta)A_{\mathrm{base}}(Q)
+
\eta A_{\mathrm{nn}}(Q).
\]

Let the certified envelope be

\[
B(Q)=m(Q)+C_B.
\]

The projection requires

\[
A_{\pi^\Theta}(Q)\le B(Q)
\qquad
\text{for every dispatched state }Q.
\]

Define the safe gate upper bound:

\[
\eta_{\mathrm{safe}}(Q)=
\begin{cases}
1,
& A_{\mathrm{nn}}(Q)\le A_{\mathrm{base}}(Q),\\
\left[
\dfrac{B(Q)-A_{\mathrm{base}}(Q)}
{A_{\mathrm{nn}}(Q)-A_{\mathrm{base}}(Q)}
\right]_0^1,
& A_{\mathrm{nn}}(Q)>A_{\mathrm{base}}(Q),
\end{cases}
\]

where \([x]_0^1=\min\{1,\max\{0,x\}\}\).

The projected gate is

\[
\eta^\Theta_{\mathrm{proj}}(Q,\mu,\xi)
=
\min\{\eta_{\mathrm{raw}}^\Theta(Q,\mu,\xi),\eta_{\mathrm{safe}}(Q)\}.
\]

This construction is valid only if \(B(Q)-A_{\mathrm{base}}(Q)\) is nonnegative
for all \(Q\). That nonnegativity follows from the base envelope if \(C_B\) is
chosen correctly.

### 7.6 Recommended architecture block

The z2 architecture should use the following block order:

1. compute analytic backbone logits,
2. encode resource-local features,
3. compute invariant global context,
4. compute bounded residual logits,
5. compute neural proposal as backbone logits plus residual logits,
6. compute raw scalar gate,
7. apply either hard tail fallback or drift-envelope projection,
8. output mixture policy,
9. emit certificate diagnostics.

This block order is preferred over a fully neural dispatcher because every
route to certification depends on keeping the analytic energy visible at the
final action layer.

---

## 8. Certification Theorems

### 8.1 Base theorem target

Theorem target 1: analytic backbone stability.

Assume:

1. \(N<\infty\),
2. \(\mu_i>0\),
3. \(\lambda<\Lambda\),
4. \(\alpha>0,\beta>0,c\ge0,\gamma\in\mathbb R\),
5. the backbone envelope
   \(A_{\mathrm{base}}(Q)\le m(Q)+C_B\) holds for all \(Q\).

Then the CTMC controlled by \(p^{\mathrm{base}}\) is non-explosive and positive
Harris recurrent.

Proof obligation: complete the Foster-Lyapunov proof using \(V\), the generator
identity, the uniform rate bound from Section 2.6, and irreducibility under the
strictly positive softmax routing probabilities.

### 8.2 Tail fallback theorem target

Theorem target 2: CertiQ-Net with hard tail fallback is stable.

Assume:

1. all assumptions of Theorem target 1,
2. the final CertiQ-Net policy satisfies exact fallback:

\[
S_\beta(Q)>R_{\mathrm{cert}}
\Longrightarrow
\pi^\Theta(Q,\mu,\xi)=p^{\mathrm{base}}(Q,\mu;\phi),
\]

3. \(R_{\mathrm{cert}}<\infty\),
4. the neural residual and gate are finite on every finite state.

Then the CTMC controlled by \(\pi^\Theta\) is non-explosive and positive Harris
recurrent.

Reason: outside the finite fallback set, the drift is identical to the
backbone drift. Inside the finite set, the generator drift is bounded.

### 8.3 Drift projection theorem target

Theorem target 3: CertiQ-Net with exact projection is stable.

Assume:

1. \(N<\infty\),
2. \(\mu_i>0\),
3. \(\lambda<\Lambda\),
4. \(C_B<\infty\),
5. for all \(Q\),

\[
A_{\pi^\Theta}(Q)\le m(Q)+C_B.
\]

Then the CTMC controlled by \(\pi^\Theta\) satisfies the same Lyapunov drift
form as the backbone and is positive Harris recurrent.

This theorem does not require the neural residual to vanish in the tail. It
requires exact enforcement of the arrival envelope.

### 8.4 Bounded residual theorem target

Theorem target 4: bounded residual feasibility.

Assume:

\[
|r_i^\Theta(Q,\mu,\xi)|\le R_{\max}
\quad
\text{for all }i,Q,\mu,\xi.
\]

Prove or reject the following type of bound:

\[
A_{\mathrm{nn}}(Q)
\le
m(Q)+C_{\mathrm{nn}}(R_{\max},\alpha,\beta,\gamma,c,\mu,N).
\]

If true, this establishes that bounded residuals preserve a minimum-type
arrival envelope, perhaps with a larger constant. This is useful but not
required for the projection theorem, because projection can enforce the
envelope pointwise.

This theorem must not be claimed until the inequality is proved.

---

## 9. Certificate Diagnostics

Every implementation must expose the following quantities at inference and in
experiments:

1. \(A_{\mathrm{base}}(Q)\),
2. \(A_{\mathrm{nn}}(Q)\),
3. \(A_{\pi^\Theta}(Q)\),
4. \(m(Q)\),
5. \(B(Q)=m(Q)+C_B\),
6. drift slack \(B(Q)-A_{\pi^\Theta}(Q)\),
7. raw gate \(\eta_{\mathrm{raw}}\),
8. final gate \(\eta\),
9. projected gate cap \(\eta_{\mathrm{safe}}\), when projection is used,
10. tail fallback indicator, when fallback is used,
11. residual logit norm \(\max_i |r_i^\Theta|\),
12. final policy entropy,
13. selected resource and selected resource workload coordinate.

No performance result should be reported without certificate diagnostics.

---

## 10. Training Definition

### 10.1 Training objective

The training objective is an optimization tool, not a certificate.

For a finite horizon \(T\), define the empirical rollout cost

\[
\widehat J_T(\Theta)
=
\frac1T
\int_0^T
c(Q_\Theta(t),\mu,\xi)\,dt.
\]

The recommended loss is

\[
\mathcal L(\Theta)
=
\widehat J_T(\Theta)
+
\omega_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}
+
\omega_{\mathrm{gate}}\mathcal L_{\mathrm{gate}}
+
\omega_{\mathrm{drift}}\mathcal L_{\mathrm{drift}}
+
\omega_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+
\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}.
\]

The symbols \(\omega_{\mathrm{bc}},\omega_{\mathrm{gate}},
\omega_{\mathrm{drift}},\omega_{\mathrm{res}},\omega_{\mathrm{ent}}\) are
training hyperparameters and are unrelated to the arrival rate \(\lambda\).

### 10.2 Warm-start loss

\[
\mathcal L_{\mathrm{bc}}
=
\mathbb E_{Q\sim\mathcal D_{\mathrm{state}}}
\left[
\mathrm{KL}
\left(
p^{\mathrm{target}}(\cdot|Q)
\Vert
\pi^\Theta(\cdot|Q)
\right)
\right].
\]

Allowed targets:

1. analytic backbone,
2. rollout-improved backbone,
3. exact dynamic programming policy for tiny systems,
4. audited heuristic only if clearly labeled as heuristic.

### 10.3 Gate penalty

\[
\mathcal L_{\mathrm{gate}}
=
\mathbb E_Q[\eta^\Theta(Q,\mu,\xi)^2].
\]

For hard or smooth fallback:

\[
\mathcal L_{\mathrm{tail}}
=
\mathbb E_Q
\left[
(\eta^\Theta(Q,\mu,\xi))^2
\mathbf 1_{\{S_\beta(Q)>R_{\mathrm{cert}}\}}
\right].
\]

This penalty is a training aid. It is not a substitute for exact fallback.

### 10.4 Drift penalty

\[
\mathcal L_{\mathrm{drift}}
=
\mathbb E_Q
\left[
(A_{\pi^\Theta}(Q)-B(Q))_+^2
\right].
\]

If projection is active, this should be zero up to numerical tolerance. If it
is not zero, the implementation is not enforcing the certificate correctly.

### 10.5 Residual size penalty

\[
\mathcal L_{\mathrm{res}}
=
\mathbb E_Q
\left[
\frac1N\sum_i (r_i^\Theta(Q,\mu,\xi))^2
\right].
\]

This encourages the residual to act as a correction rather than as a full
replacement for the analytic energy.

### 10.6 Entropy term

\[
\mathcal L_{\mathrm{ent}}
=
-
\mathbb E_Q
\left[
\sum_i
\pi_i^\Theta(Q,\mu,\xi)
\log \pi_i^\Theta(Q,\mu,\xi)
\right].
\]

Entropy may help early training but can fight dispatch quality under high load.
Its sign and weight must be declared.

### 10.7 Training engines

Allowed training engines:

1. differentiable discrete-event simulation with smoothed event dynamics,
2. score-function policy gradients,
3. imitation learning from audited targets,
4. finite-state dynamic programming for tiny systems.

Differentiable simulation may improve gradient quality. It does not prove
stability.

### 10.8 Curriculum

Recommended curriculum:

1. fix \(\phi\) and train the residual to imitate the backbone,
2. enable the residual only inside a compact gate region,
3. train with rollout cost on small \(N\),
4. audit certificate diagnostics on sampled and grid state banks,
5. enable projection after the base envelope constant \(C_B\) is fixed,
6. test size transfer,
7. only then allow learned \(\phi\) or richer domain context.

---

## 11. Domain Adapters

### 11.1 Adapter contract

A domain adapter maps a real system into the abstract dispatch tuple

\[
\mathcal D
=
(N,\lambda,\mu,\xi,Q,c).
\]

The adapter must specify:

1. what counts as an arrival,
2. what counts as a resource,
3. how \(\mu_i\) is measured or estimated,
4. what \(Q_i\) counts,
5. what service-completion law is assumed,
6. what context \(\xi_i\) means,
7. what cost is optimized,
8. whether the CTMC certificate applies exactly or only as an approximation.

### 11.2 Queueing adapter

Resources: servers.  
Arrivals: jobs.  
\(\mu_i\): exponential service rate of server \(i\).  
\(Q_i\): number of jobs at server \(i\).  
Certificate status: exact, under the CTMC assumptions.

### 11.3 Mixture-of-experts adapter

Resources: experts.  
Arrivals: tokens or token groups.  
\(\mu_i\): estimated processing throughput of expert \(i\).  
\(Q_i\): tokens queued or active at expert \(i\).  
\(\xi_i\): expert metadata and token-compatibility features.  
Cost: queueing delay, drop penalty, or throughput-normalized latency.

Certificate status:

1. exact only if token arrivals and expert completions are modeled by the CTMC
   generator in Section 2.6,
2. approximate or empirical if batching, top-k routing, finite buffers,
   synchronous transformer layers, or non-exponential service are used without
   a separate proof.

The architecture may be used as a MoE routing mechanism, but the paper must
not claim the CTMC theorem applies to a transformer implementation unless the
adapter assumptions are satisfied or a new theorem is proved.

### 11.4 Multi-robot task allocation adapter

Resources: robots.  
Arrivals: tasks.  
\(\mu_i\): task completion rate for robot \(i\) under the declared task family.  
\(Q_i\): number of pending tasks assigned to robot \(i\).  
\(\xi_i\): position, battery, travel-time estimate, capability vector, or local
environment features.  
Cost: backlog, latency, travel cost, or weighted combination.

Certificate status:

1. exact if task arrivals are Poisson, service times are exponential with
   rates \(\mu_i\), and dispatch is irrevocable,
2. not exact if travel times create state-dependent service rates unless the
   state-dependent generator is separately certified.

### 11.5 Communication or channel adapter

Resources: channels or links.  
Arrivals: packets or requests.  
\(\mu_i\): service rate or transmission capacity.  
\(Q_i\): queued packets per channel.  
\(\xi_i\): noise, fading state, cost, or reliability features.

Certificate status depends on whether channel dynamics are fixed during the
dispatch decision or are modeled as part of an expanded Markov state.

---

## 12. Model Variants

### 12.1 CertiQ-Net-S

Small certified fallback model:

1. fixed backbone parameters,
2. local shared encoder,
3. invariant pooling,
4. bounded residual logits,
5. hard tail fallback,
6. no learned \(\phi\).

Purpose: first implementation and first clean theorem.

### 12.2 CertiQ-Net-P

Projected certified model:

1. fixed or bounded learned backbone parameters,
2. bounded residual logits,
3. exact pointwise drift projection,
4. certificate diagnostics required.

Purpose: strongest z2 architecture because the residual may remain active
outside the compact performance region while still obeying the Lyapunov
arrival envelope.

### 12.3 CertiQ-Net-M

Meta-adaptive dispatcher:

1. context network predicts \(\phi\),
2. residual adapts across domains,
3. projection or fallback remains mandatory,
4. adapters define domain-specific \(\xi\).

Purpose: long-term general dispatcher. This is not the first theorem target
unless the learned \(\phi\) range is explicitly constrained and included in the
proof.

### 12.4 CertiQ-Net-X

Experimental unconstrained ablation:

1. neural residual without certificate,
2. no fallback or projection,
3. used only to demonstrate failure modes or performance/certification
   tradeoffs.

Purpose: ablation. It is not deployable as a certified model.

---

## 13. Experiment Protocol

### 13.1 Required architecture comparisons

Every z2 experiment should include:

1. analytic backbone,
2. CertiQ-Net-S,
3. CertiQ-Net-P when \(C_B\) is available,
4. residual without certificate as an ablation,
5. learned-backbone-only ablation,
6. residual with smooth training gate and hard evaluation gate.

Classical dispatch rules may be reported as context, but they are not the main
novelty axis.

### 13.2 Queueing system families

Family A: fixed heterogeneous anchor.  
Family B: lognormal random service rates.  
Family C: clustered slow/medium/fast resources.  
Family D: size transfer from smaller \(N\) to larger \(N\).  
Family E: high-load stress tests with \(\rho=\lambda/\Lambda\) near one.

### 13.3 Cross-domain demonstrations

The multi-domain claim should be demonstrated by adapters, not by changing the
architecture:

1. queueing servers,
2. simulated MoE experts,
3. simulated robot task allocation,
4. optional communication-channel dispatch.

The same policy block should be used. Only \(\xi_i\), cost, and adapter-level
simulation details may change.

### 13.4 Required metrics

Performance:

1. average total queue length,
2. average cost,
3. latency proxy,
4. tail quantiles,
5. maximum observed backlog,
6. drop rate if finite buffers are used.

Certificate:

1. drift-envelope violation rate,
2. minimum drift slack,
3. average drift slack,
4. projection activation rate,
5. tail fallback activation rate,
6. gate activation rate,
7. residual logit magnitude,
8. empirical instability or runaway rate.

Training:

1. gradient variance,
2. training wall-clock,
3. sample count,
4. final cost under equal training budget,
5. transfer gap across \(N\), \(\mu\), and load.

### 13.5 State-bank audit

Before reporting trained results, evaluate the policy on an audited bank of
states:

1. random states from rollouts,
2. grid states for small \(N\),
3. boundary states where one resource is heavily loaded,
4. balanced high-load states,
5. states outside the fallback radius,
6. adversarial states selected to maximize \(A_{\pi^\Theta}(Q)-B(Q)\).

For projected models, the maximum violation must be zero up to declared
floating-point tolerance.

---

## 14. Implementation Invariants

The implementation must enforce:

1. all probabilities are normalized and nonnegative,
2. \(\alpha>0\), \(\beta>0\), \(c\ge0\),
3. \(\mu_i>0\) for all resources,
4. residual logits are bounded when using bounded-residual theorem routes,
5. hard fallback sets \(\eta=0\) exactly outside \(R_{\mathrm{cert}}\),
6. projection computes \(A_{\mathrm{base}}\), \(A_{\mathrm{nn}}\), and
   \(\eta_{\mathrm{safe}}\) from the same policy probabilities used for action,
7. certificate diagnostics are logged for every evaluation rollout,
8. permutation equivariance is unit-tested by randomly permuting resource
   order.

Minimal permutation test:

\[
\|\pi(\sigma Q,\sigma\mu,\sigma\xi)-\sigma\pi(Q,\mu,\xi)\|_\infty
\le \varepsilon_{\mathrm{num}}.
\]

Minimal projection test:

\[
A_{\pi^\Theta}(Q)-B(Q)\le \varepsilon_{\mathrm{num}}
\]

for every audited state.

---

## 15. Proof Checklist

Before manuscript claims are made, the following must be completed.

1. Prove the exact generator identity for \(V\).
2. Prove the base arrival envelope
   \(A_{\mathrm{base}}(Q)\le m(Q)+C_B\).
3. Convert the envelope into a negative Foster-Lyapunov drift under
   \(\lambda<\Lambda\).
4. State the CTMC irreducibility and non-explosion conditions.
5. Prove hard tail fallback inheritance.
6. Prove drift projection inheritance.
7. Decide whether bounded residuals independently imply an arrival envelope.
8. Verify every cited theorem from source papers before using it as literature
   support.
9. Separate theorem claims for the CTMC model from empirical claims for adapted
   domains.

---

## 16. Final Architecture Definition

Given \(Q\), \(\mu\), \(\xi\), and parameters \(\Theta\):

1. compute

\[
y_i=Q_i/\mu_i^\beta;
\]

2. compute backbone logits

\[
u_i^{\mathrm{base}}
=
\gamma\log\mu_i-\alpha(Q_i+c)/\mu_i^\beta;
\]

3. compute

\[
p^{\mathrm{base}}=\operatorname{softmax}(u^{\mathrm{base}});
\]

4. compute shared local features

\[
z_i^0=f_{\mathrm{loc}}([\log(1+Q_i),\log\mu_i,y_i,\mu_i/\Lambda,\xi_i]);
\]

5. compute invariant global context

\[
g=f_{\mathrm{glob}}(\{z_i^0\}_{i=1}^N);
\]

6. compute bounded residual logits

\[
r_i^\Theta=R_{\max}\tanh(w^\top f_{\mathrm{res}}([z_i^0,g])+b);
\]

7. compute neural proposal

\[
p^{\mathrm{nn}}
=
\operatorname{softmax}(u^{\mathrm{base}}+r^\Theta);
\]

8. compute raw gate

\[
\eta_{\mathrm{raw}}=\eta_{\max}\sigma(w_g^\top g+b_g);
\]

9. compute certified gate by one of:

Hard fallback:

\[
\eta=\eta_{\mathrm{raw}}\mathbf 1_{\{S_\beta(Q)\le R_{\mathrm{cert}}\}};
\]

Projection:

\[
\eta=\min\{\eta_{\mathrm{raw}},\eta_{\mathrm{safe}}(Q)\};
\]

10. output final policy

\[
\pi^\Theta
=
(1-\eta)p^{\mathrm{base}}+\eta p^{\mathrm{nn}}.
\]

This is the z2 CertiQ-Net architecture. The neural module is expressive, but it
is structurally subordinate to an auditable certificate mechanism. That is the
central research contribution.

---

## 17. Immediate Work Items

The next technical steps are:

1. freeze \(C_B\) by completing the base envelope proof,
2. implement CertiQ-Net-S first,
3. add exact diagnostic logging,
4. run permutation and projection unit tests,
5. train only after the untrained architecture passes certificate audits,
6. implement CertiQ-Net-P after the projection constant is fixed,
7. treat MoE and robotics as adapter demonstrations with explicit assumptions.

This order protects the project from the main failure mode: training an
impressive neural dispatcher before the certificate object is mathematically
well defined.

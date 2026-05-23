# Certification Route

This file defines how CertiQ-Net can become a certified neural architecture.
It deliberately separates architecture from theorem.

## 1. Base-Controller Certificate

For CertiQ-Net, the default Lyapunov candidate is the weighted quadratic

\[
V(Q)=\frac12\sum_i Q_i^2/\mu_i^\beta.
\]

For a policy \(\pi\), the exact generator is

\[
(\mathcal L_\pi V)(Q)
=
\lambda\sum_i \pi_i(Q)\frac{Q_i}{\mu_i^\beta}
-
\sum_i \mu_i^{1-\beta}Q_i
+
R_0(Q),
\]

where

\[
R_0(Q)
=
\frac{\lambda}{2}\sum_i\pi_i(Q)\mu_i^{-\beta}
+
\frac12\sum_i\mu_i^{1-\beta}\mathbf 1_{\{Q_i>0\}}.
\]

Since \(R_0(Q)\) is uniformly bounded, certification reduces to controlling

\[
A_\pi(Q)
:=
\sum_i \pi_i(Q)Q_i/\mu_i^\beta.
\]

The analytic base controller is designed so that this term admits a
minimum-type envelope:

\[
A_{\mathrm{base}}(Q)
\le
m(Q)+C_1,
\qquad
m(Q):=\min_i Q_i/\mu_i^\beta.
\]

Once this envelope is proved, it yields a negative linear drift under
\(\lambda<\Lambda\). This paper should state and prove that certificate
directly.

## 2. Route A: Tail Fallback Certificate

Assume:

1. the analytic base policy \(p^{\mathrm{base}}\) satisfies a Foster-Lyapunov
   inequality,
2. CertiQ-Net equals the base policy outside a finite or compact set:

\[
\pi_\Theta(Q)=p^{\mathrm{base}}(Q)
\quad\text{whenever}\quad |Q|_1>R.
\]

Then the generator drift outside that set is identical to the base-controller
drift. Inside the set, all generator differences are bounded because the set is
finite for the CTMC.

Therefore CertiQ-Net inherits the same Foster-Lyapunov tail drift.

This is the cleanest first theorem path.

## 3. Route B: Drift-Envelope Projection

Define a certified envelope

\[
B(Q)=m(Q)+C_B,
\]

where \(C_B\) is chosen so that the base-controller proof closes with

\[
A_\pi(Q)\le B(Q).
\]

CertiQ-Net is projected so that

\[
A_{\pi_\Theta}(Q)\le B(Q)
\quad\text{for all }Q.
\]

Then the same weighted-quadratic drift proof applies:

\[
(\mathcal L_{\pi_\Theta}V)(Q)
\le
\lambda B(Q)
-
\sum_i\mu_i^{1-\beta}Q_i
+
C_0.
\]

Using the decomposition

\[
Q_i=\mu_i^\beta m(Q)+\Delta_i(Q),
\qquad \Delta_i(Q)\ge 0,
\]

one obtains

\[
(\mathcal L_{\pi_\Theta}V)(Q)
\le
-(\Lambda-\lambda)m(Q)
-
\sum_i\mu_i^{1-\beta}\Delta_i(Q)
+
C,
\]

and hence

\[
(\mathcal L_{\pi_\Theta}V)(Q)
\le
-\epsilon |Q|_1+C.
\]

This route certifies a neural policy without requiring it to equal the base
controller in the tail, but only if the projection is exact and the envelope is
proved.

## 4. Route C: Small Residual Perturbation

A weaker route is to prove that the neural policy is uniformly close to the
base controller:

\[
\|\pi_\Theta(Q)-p^{\mathrm{base}}(Q)\|_1
\le
\delta(Q).
\]

Then

\[
|A_{\pi_\Theta}(Q)-A_{\mathrm{base}}(Q)|
\le
\max_i Q_i/\mu_i^\beta \cdot \delta(Q).
\]

This is not enough if \(\delta(Q)\) is a constant, because the multiplier grows
with the state. To close the proof, the residual must decay in the tail, for
example

\[
\delta(Q)
\le
\frac{C}{1+\max_i Q_i/\mu_i^\beta}.
\]

This route is mathematically possible but less attractive than tail fallback
or drift projection.

## 5. Certification Theorem Template

The target theorem should eventually look like this.

**Theorem template.**
Fix \(N\), \(\lambda<\Lambda\), \(\mu_i>0\), and base-controller parameters
\(\alpha>0,\beta>0,c\ge0,\gamma\in\mathbb R\). Suppose:

1. the analytic base controller satisfies the weighted-quadratic
   Foster-Lyapunov envelope, and
2. CertiQ-Net satisfies either exact tail fallback outside a finite set or the
   pointwise drift-envelope constraint
   \(A_{\pi_\Theta}(Q)\le m(Q)+C_B\).

Then the CertiQ-Net CTMC is non-explosive, irreducible, and positive Harris
recurrent.

## 6. What Must Be Verified In Code

Every implementation should expose:

- computed \(A_{\pi_\Theta}(Q)\),
- computed \(A_{\mathrm{base}}(Q)\),
- projected gate value,
- drift-envelope slack,
- tail fallback activation,
- empirical generator estimate for sampled state banks.

No experiment should report performance without also reporting whether the
certificate layer was active and whether drift constraints were violated.

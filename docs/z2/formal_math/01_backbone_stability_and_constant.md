# Analytic Backbone Stability And Exact Envelope Constant

This file proves the analytic-backbone part of the z2 CertiQ-Net program for
the primary CTMC dispatch model. The proof is direct. It does not use a fluid
limit and it does not rely on a separate deterministic reflected-ODE theorem.

---

## 1. Model And Backbone Policy

Fix:

1. a finite number of resources \(N \ge 1\),
2. service capacities \(\mu_i > 0\),
3. total capacity

\[
\Lambda := \sum_{i=1}^N \mu_i,
\]

4. arrival rate \(\lambda > 0\) with

\[
\lambda < \Lambda,
\]

5. backbone parameters

\[
\alpha > 0,\qquad \beta > 0,\qquad \gamma \in \mathbb R,\qquad c \ge 0.
\]

The state is \(Q=(Q_1,\ldots,Q_N)\in\mathbb Z_+^N\). Under the analytic
backbone, the routing probabilities are

\[
p_i^{\mathrm{base}}(Q,\mu;\phi)
=
\frac{\mu_i^\gamma \exp\!\left(-\alpha (Q_i+c)/\mu_i^\beta\right)}
{\sum_{j=1}^N \mu_j^\gamma \exp\!\left(-\alpha (Q_j+c)/\mu_j^\beta\right)}.
\]

The CTMC generator acting on a test function \(f:\mathbb Z_+^N\to\mathbb R\) is

\[
(\mathcal L_{\mathrm{base}} f)(Q)
=
\lambda \sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)
[f(Q+e_i)-f(Q)]
+
\sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
[f(Q-e_i)-f(Q)].
\]

Because \(p_i^{\mathrm{base}}(Q,\mu;\phi) > 0\) for every \(i\) and every
\(Q\), every arrival move \(Q\to Q+e_i\) has positive rate. Because
\(\mu_i>0\), every service move \(Q\to Q-e_i\) available at \(Q_i>0\) also has
positive rate. Hence the chain is irreducible on \(\mathbb Z_+^N\).

The total jump rate at state \(Q\) is

\[
\lambda + \sum_{i=1}^N \mu_i \mathbf 1_{\{Q_i>0\}}
\le \lambda + \Lambda.
\]

Therefore the chain has a uniform finite rate bound and is non-explosive.

---

## 2. Weighted Coordinates And Lyapunov Function

Define the weighted coordinates

\[
y_i(Q) := \frac{Q_i}{\mu_i^\beta},
\qquad
m(Q) := \min_{1\le i\le N} y_i(Q).
\]

Define the weighted quadratic Lyapunov function

\[
V(Q) := \frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i^\beta}.
\]

This function is norm-like on \(\mathbb Z_+^N\). In particular,

\[
V(Q) \ge \frac{1}{2\max_i \mu_i^\beta}\sum_{i=1}^N Q_i^2,
\]

so \(V(Q)\to\infty\) whenever \(|Q|_1\to\infty\).

---

## 3. Exact Generator Identity

For an arrival at coordinate \(i\),

\[
V(Q+e_i)-V(Q)
=
\frac{Q_i}{\mu_i^\beta}
+
\frac{1}{2\mu_i^\beta}.
\]

For a service completion at coordinate \(i\) with \(Q_i>0\),

\[
V(Q-e_i)-V(Q)
=
-\frac{Q_i}{\mu_i^\beta}
+
\frac{1}{2\mu_i^\beta}.
\]

Substituting into the generator gives

\[
(\mathcal L_{\mathrm{base}} V)(Q)
=
\lambda A_{\mathrm{base}}(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta} Q_i
+
R_0(Q),
\]

where

\[
A_{\mathrm{base}}(Q)
:=
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi) y_i(Q),
\]

and

\[
R_0(Q)
=
\frac{\lambda}{2}
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)\mu_i^{-\beta}
+
\frac12 \sum_{i=1}^N \mu_i^{1-\beta}\mathbf 1_{\{Q_i>0\}}.
\]

Since \(\sum_i p_i^{\mathrm{base}}(Q,\mu;\phi)=1\),

\[
0 \le R_0(Q) \le C_0,
\]

with

\[
C_0
:=
\frac{\lambda}{2}\max_{1\le i\le N}\mu_i^{-\beta}
+
\frac12\sum_{i=1}^N \mu_i^{1-\beta}.
\]

So the proof reduces to controlling \(A_{\mathrm{base}}(Q)\).

---

## 4. Exact Softmax Reformulation

Define the shifted offsets

\[
\kappa_i := \frac{c}{\mu_i^\beta} - \frac{\gamma}{\alpha}\log \mu_i.
\]

Then define

\[
s_i(Q) := y_i(Q) + \kappa_i
=
\frac{Q_i}{\mu_i^\beta}
+
\frac{c}{\mu_i^\beta}
-
\frac{\gamma}{\alpha}\log \mu_i.
\]

The backbone probabilities become

\[
p_i^{\mathrm{base}}(Q,\mu;\phi)
=
\frac{e^{-\alpha s_i(Q)}}{\sum_{j=1}^N e^{-\alpha s_j(Q)}}.
\]

Thus the analytic backbone is exactly a softmax over the shifted energies
\(s_i(Q)\).

---

## 5. Softmax Minimum Lemma

### Lemma 5.1

Let \(x=(x_1,\ldots,x_N)\in\mathbb R^N\), and define

\[
\pi_i(x) := \frac{e^{-\alpha x_i}}{\sum_{j=1}^N e^{-\alpha x_j}},
\qquad \alpha>0.
\]

Then

\[
\sum_{i=1}^N \pi_i(x)x_i
\le
\min_{1\le i\le N} x_i + \frac{\log N}{\alpha}.
\]

### Proof

The Gibbs variational identity gives

\[
\sum_{i=1}^N \pi_i(x)x_i
+
\frac{1}{\alpha}\sum_{i=1}^N \pi_i(x)\log \pi_i(x)
=
-\frac{1}{\alpha}\log\sum_{j=1}^N e^{-\alpha x_j}.
\]

Because

\[
\sum_{j=1}^N e^{-\alpha x_j}
\ge
e^{-\alpha \min_i x_i},
\]

the right-hand side satisfies

\[
-\frac{1}{\alpha}\log\sum_{j=1}^N e^{-\alpha x_j}
\le
\min_i x_i.
\]

Also the entropy term satisfies

\[
-\sum_{i=1}^N \pi_i(x)\log \pi_i(x) \le \log N,
\]

equivalently

\[
\sum_{i=1}^N \pi_i(x)\log \pi_i(x) \ge -\log N.
\]

Hence

\[
\sum_{i=1}^N \pi_i(x)x_i
\le
\min_i x_i + \frac{\log N}{\alpha}.
\]

`QED`

---

## 6. Exact Backbone Envelope And Exact Constant \(C_B\)

Apply Lemma 5.1 with \(x_i = s_i(Q)\). Then

\[
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)s_i(Q)
\le
\min_{1\le i\le N} s_i(Q) + \frac{\log N}{\alpha}.
\]

Now

\[
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)y_i(Q)
=
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)s_i(Q)
-
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)\kappa_i.
\]

Since

\[
\sum_{i=1}^N p_i^{\mathrm{base}}(Q,\mu;\phi)\kappa_i
\ge
\min_{1\le i\le N}\kappa_i,
\]

and

\[
\min_{1\le i\le N} s_i(Q)
=
\min_{1\le i\le N}\bigl(y_i(Q)+\kappa_i\bigr)
\le
m(Q)+\max_{1\le i\le N}\kappa_i,
\]

we obtain

\[
A_{\mathrm{base}}(Q)
\le
m(Q) + C_B,
\]

where the exact envelope constant is

\[
C_B
:=
\frac{\log N}{\alpha}
+
\max_{1\le i\le N}\kappa_i
-
\min_{1\le i\le N}\kappa_i.
\]

Equivalently,

\[
C_B
=
\frac{\log N}{\alpha}
+
\max_i\left(
\frac{c}{\mu_i^\beta}
-
\frac{\gamma}{\alpha}\log \mu_i
\right)
-
\min_i\left(
\frac{c}{\mu_i^\beta}
-
\frac{\gamma}{\alpha}\log \mu_i
\right).
\]

This is a finite constant because \(N<\infty\) and \(\mu_i>0\) are fixed.

That closes the backbone envelope required by Section 5.4 of the z2 end-to-end
formal definition.

---

## 7. Drift Closure

Substitute the envelope into the generator identity:

\[
(\mathcal L_{\mathrm{base}} V)(Q)
\le
\lambda[m(Q)+C_B]
-
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
+
C_0.
\]

Decompose

\[
Q_i = \mu_i^\beta m(Q) + \Delta_i(Q),
\qquad
\Delta_i(Q)\ge 0.
\]

Then

\[
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
=
\sum_{i=1}^N \mu_i^{1-\beta}\bigl(\mu_i^\beta m(Q)+\Delta_i(Q)\bigr)
=
\Lambda m(Q) + \sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q).
\]

Therefore

\[
(\mathcal L_{\mathrm{base}} V)(Q)
\le
-(\Lambda-\lambda)m(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q)
+
R_B,
\]

where

\[
R_B := \lambda C_B + C_0.
\]

Now

\[
|Q|_1
=
\sum_{i=1}^N Q_i
=
\left(\sum_{i=1}^N \mu_i^\beta\right)m(Q)
+
\sum_{i=1}^N \Delta_i(Q).
\]

Set

\[
\varepsilon_B
:=
\min\!\left\{
\frac{\Lambda-\lambda}{\sum_{i=1}^N \mu_i^\beta},
\min_{1\le i\le N}\mu_i^{1-\beta}
\right\}.
\]

Because \(\lambda<\Lambda\) and every \(\mu_i>0\), we have \(\varepsilon_B>0\).
Moreover,

\[
\varepsilon_B |Q|_1
\le
(\Lambda-\lambda)m(Q)
+
\sum_{i=1}^N \mu_i^{1-\beta}\Delta_i(Q).
\]

Hence

\[
(\mathcal L_{\mathrm{base}} V)(Q)
\le
-\varepsilon_B |Q|_1 + R_B.
\]

This is the required linear Foster-Lyapunov drift inequality.

---

## 8. Backbone Stability Theorem

### Theorem 8.1

For the CTMC dispatch model above, suppose:

1. \(N<\infty\),
2. \(\mu_i>0\) for all \(i\),
3. \(\lambda<\Lambda=\sum_i \mu_i\),
4. \(\alpha>0\), \(\beta>0\), \(\gamma\in\mathbb R\), \(c\ge 0\).

Then the analytic-backbone CTMC is irreducible, non-explosive, and positive
Harris recurrent.

### Proof

Irreducibility and non-explosion were established in Section 1. Section 7
proved that the norm-like function \(V\) satisfies

\[
(\mathcal L_{\mathrm{base}} V)(Q)
\le
-\varepsilon_B |Q|_1 + R_B.
\]

Therefore

\[
(\mathcal L_{\mathrm{base}} V)(Q)\le -1
\]

outside the finite set

\[
K_B := \left\{Q\in\mathbb Z_+^N:\ \varepsilon_B |Q|_1 \le R_B + 1\right\}.
\]

By the standard Foster-Lyapunov criterion for irreducible non-explosive
countable-state CTMCs, the chain is positive Harris recurrent.

`QED`

---

## 9. Outputs Required By The Proof

The proof determines explicit quantities that must be exposed by
implementation:

1. \(C_B\),
2. \(C_0\),
3. \(R_B=\lambda C_B+C_0\),
4. \(\varepsilon_B\),
5. \(A_{\mathrm{base}}(Q)\),
6. \(m(Q)\),
7. backbone drift slack

\[
\bigl[m(Q)+C_B\bigr] - A_{\mathrm{base}}(Q).
\]

These are theorem-facing diagnostics, not optional training metadata.

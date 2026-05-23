# Gate Inheritance Theorems

This file proves that the two certified z2 gate mechanisms inherit the
stability of the analytic backbone for the primary CTMC dispatch model.

All notation is inherited from
[01_backbone_stability_and_constant.md](./01_backbone_stability_and_constant.md).

---

## 1. Shared Preliminaries

Let

\[
y_i(Q):=\frac{Q_i}{\mu_i^\beta},
\qquad
m(Q):=\min_i y_i(Q),
\qquad
V(Q):=\frac12\sum_i \frac{Q_i^2}{\mu_i^\beta}.
\]

For any admissible policy \(\pi(Q,\mu,\xi)\in\Delta_N\), define the arrival
term

\[
A_\pi(Q)
:=
\sum_{i=1}^N \pi_i(Q,\mu,\xi) y_i(Q).
\]

The exact generator identity is

\[
(\mathcal L_\pi V)(Q)
=
\lambda A_\pi(Q)
-
\sum_{i=1}^N \mu_i^{1-\beta}Q_i
+
R_0^\pi(Q),
\]

where

\[
R_0^\pi(Q)
=
\frac{\lambda}{2}\sum_{i=1}^N \pi_i(Q,\mu,\xi)\mu_i^{-\beta}
+
\frac12\sum_{i=1}^N \mu_i^{1-\beta}\mathbf 1_{\{Q_i>0\}}.
\]

Since \(\sum_i \pi_i(Q,\mu,\xi)=1\),

\[
0\le R_0^\pi(Q)\le C_0,
\]

with the same constant

\[
C_0
=
\frac{\lambda}{2}\max_i \mu_i^{-\beta}
+
\frac12\sum_{i=1}^N \mu_i^{1-\beta}.
\]

Hence every gate proof reduces to controlling \(A_\pi(Q)\).

From file `01`, the analytic backbone satisfies

\[
A_{\mathrm{base}}(Q)\le m(Q)+C_B
\]

for every \(Q\), where

\[
C_B
=
\frac{\log N}{\alpha}
+
\max_i\left(\frac{c}{\mu_i^\beta}-\frac{\gamma}{\alpha}\log\mu_i\right)
-
\min_i\left(\frac{c}{\mu_i^\beta}-\frac{\gamma}{\alpha}\log\mu_i\right).
\]

Also define

\[
\varepsilon_B
:=
\min\!\left\{
\frac{\Lambda-\lambda}{\sum_i \mu_i^\beta},
\min_i \mu_i^{1-\beta}
\right\}
>0.
\]

---

## 2. Hard Tail Fallback

### Definition 2.1

Let \(R_{\mathrm{cert}}<\infty\), and define the weighted tail size

\[
S_\beta(Q):=\sum_{i=1}^N \frac{Q_i}{\mu_i^\beta}.
\]

The CertiQ-Net hard fallback policy is the z2 mixture policy

\[
\pi^\Theta
=
(1-\eta^\Theta)p^{\mathrm{base}}+\eta^\Theta p^{\mathrm{nn}},
\]

where \(p^{\mathrm{base}}\) and \(p^{\mathrm{nn}}\) are softmax probability
vectors with strictly positive coordinates, and such that

\[
S_\beta(Q)>R_{\mathrm{cert}}
\Longrightarrow
\pi^\Theta(Q,\mu,\xi)=p^{\mathrm{base}}(Q,\mu;\phi).
\]

### Lemma 2.2

The set

\[
F_R:=\{Q\in\mathbb Z_+^N:\ S_\beta(Q)\le R_{\mathrm{cert}}\}
\]

is finite.

### Proof

For every \(i\),

\[
0\le Q_i \le \mu_i^\beta R_{\mathrm{cert}}
\]

whenever \(Q\in F_R\). Since each \(Q_i\) is integer-valued and \(N<\infty\),
there are only finitely many such states.

`QED`

### Theorem 2.3

Assume:

1. the hypotheses of Theorem 8.1 in file `01`,
2. \(\pi^\Theta\) is a hard-fallback policy in the sense of Definition 2.1,
3. \(p^{\mathrm{nn}}(Q,\mu,\xi)\) is the backbone-plus-residual softmax neural
   proposal from the z2 architecture,
4. the neural residual and gate outputs are finite on every state.

Then the CTMC controlled by \(\pi^\Theta\) is irreducible, non-explosive, and
positive Harris recurrent.

### Proof

Because both \(p^{\mathrm{base}}\) and \(p^{\mathrm{nn}}\) have strictly
positive coordinates and \(\pi^\Theta\) is their convex mixture, every
coordinate of \(\pi^\Theta(Q,\mu,\xi)\) is strictly positive at every state.
Hence every arrival move \(Q\to Q+e_i\) has positive rate, and together with
the positive service rates this implies irreducibility on \(\mathbb Z_+^N\).

The total jump rate under \(\pi^\Theta\) is still bounded by \(\lambda+\Lambda\),
so the chain is non-explosive.

Outside the finite set \(F_R\), the hard-fallback policy equals the analytic
backbone exactly. Hence for \(Q\notin F_R\),

\[
(\mathcal L_{\pi^\Theta}V)(Q)
=
(\mathcal L_{\mathrm{base}}V)(Q)
\le
-\varepsilon_B |Q|_1 + R_B,
\]

where \(R_B=\lambda C_B+C_0\) is the backbone constant from file `01`.

Inside the finite set \(F_R\), the generator drift is finite because:

1. \(F_R\) is finite by Lemma 2.2,
2. \(V(Q\pm e_i)-V(Q)\) is finite at every finite state,
3. the jump rates are uniformly bounded.

Therefore there exists a finite constant

\[
M_R:=\max_{Q\in F_R} (\mathcal L_{\pi^\Theta}V)(Q).
\]

Let

\[
\widetilde R_R:=\max\{R_B,M_R\}.
\]

Then for every state \(Q\),

\[
(\mathcal L_{\pi^\Theta}V)(Q)\le -\varepsilon_B |Q|_1 + \widetilde R_R.
\]

Thus

\[
(\mathcal L_{\pi^\Theta}V)(Q)\le -1
\]

outside the finite set

\[
K_R:=\{Q\in\mathbb Z_+^N:\ \varepsilon_B |Q|_1 \le \widetilde R_R + 1\}.
\]

By the Foster-Lyapunov criterion for non-explosive irreducible countable-state
CTMCs, the chain is positive Harris recurrent.

`QED`

### Remark 2.4

This theorem certifies the exact hard gate only. The smooth sigmoid surrogate
used during training is not covered by this theorem unless it is separately
shown to satisfy a pointwise envelope.

---

## 3. Exact Drift-Envelope Projection

### Definition 3.1

Let \(p^{\mathrm{base}}\) denote the analytic backbone and
\(p^{\mathrm{nn}}\) denote the backbone-plus-residual neural proposal. Define

\[
A_{\mathrm{base}}(Q)
:=
\sum_i p_i^{\mathrm{base}}(Q,\mu;\phi)y_i(Q),
\]

\[
A_{\mathrm{nn}}(Q)
:=
\sum_i p_i^{\mathrm{nn}}(Q,\mu,\xi)y_i(Q).
\]

Let the certified envelope be

\[
B(Q):=m(Q)+C_B.
\]

For a scalar gate \(\eta\in[0,1]\), the mixture policy is

\[
\pi^\Theta
=
(1-\eta)p^{\mathrm{base}}+\eta p^{\mathrm{nn}}.
\]

Its arrival term is

\[
A_{\pi^\Theta}(Q)
=
(1-\eta)A_{\mathrm{base}}(Q)+\eta A_{\mathrm{nn}}(Q).
\]

Define the safe gate cap

\[
\eta_{\mathrm{safe}}(Q)=
\begin{cases}
1, & A_{\mathrm{nn}}(Q)\le A_{\mathrm{base}}(Q),\\[1ex]
\left[\dfrac{B(Q)-A_{\mathrm{base}}(Q)}
{A_{\mathrm{nn}}(Q)-A_{\mathrm{base}}(Q)}\right]_0^1,
& A_{\mathrm{nn}}(Q)>A_{\mathrm{base}}(Q),
\end{cases}
\]

where \([x]_0^1:=\min\{1,\max\{0,x\}\}\).

The exact projected gate is

\[
\eta_{\mathrm{proj}}^\Theta(Q,\mu,\xi)
:=
\min\{\eta_{\mathrm{raw}}^\Theta(Q,\mu,\xi),\eta_{\mathrm{safe}}(Q)\}.
\]

### Lemma 3.2

For every state \(Q\), the projected gate satisfies

\[
A_{\pi^\Theta}(Q)\le B(Q).
\]

### Proof

If \(A_{\mathrm{nn}}(Q)\le A_{\mathrm{base}}(Q)\), then
\(A_{\pi^\Theta}(Q)\le A_{\mathrm{base}}(Q)\le B(Q)\).

Now assume \(A_{\mathrm{nn}}(Q)>A_{\mathrm{base}}(Q)\). Since
\(\eta_{\mathrm{proj}}^\Theta\le \eta_{\mathrm{safe}}\), it is enough to prove
the claim for \(\eta=\eta_{\mathrm{safe}}\). If \(B(Q)\le A_{\mathrm{base}}(Q)\),
then the base envelope already forces equality \(B(Q)=A_{\mathrm{base}}(Q)\),
and the clipped value is \(\eta_{\mathrm{safe}}=0\), so the mixture reduces to
the base policy and the claim follows.

If \(A_{\mathrm{base}}(Q)<B(Q)\), then by direct substitution,

\[
(1-\eta_{\mathrm{safe}})A_{\mathrm{base}}(Q)
+
\eta_{\mathrm{safe}}A_{\mathrm{nn}}(Q)
=
B(Q)
\]

whenever the fraction lies in \([0,1]\), and clipping can only decrease the
mixture value relative to that boundary point. Hence
\(A_{\pi^\Theta}(Q)\le B(Q)\).

`QED`

### Theorem 3.3

Assume:

1. the hypotheses of Theorem 8.1 in file `01`,
2. the final CertiQ-Net policy uses the exact projected gate from Definition
   3.1,
3. all quantities used in the projection are computed from the same policy
   probabilities used for dispatch.

Then the CTMC controlled by \(\pi^\Theta\) is irreducible, non-explosive, and
positive Harris recurrent.

### Proof

Because both \(p^{\mathrm{base}}\) and \(p^{\mathrm{nn}}\) are softmax
probability vectors with strictly positive coordinates, the convex mixture
\(\pi^\Theta\) also has strictly positive coordinates. Therefore the chain is
irreducible on \(\mathbb Z_+^N\).

The total jump rate is again bounded by \(\lambda+\Lambda\), so the chain is
non-explosive.

By Lemma 3.2,

\[
A_{\pi^\Theta}(Q)\le B(Q)=m(Q)+C_B
\]

for every state \(Q\). Hence

\[
(\mathcal L_{\pi^\Theta}V)(Q)
\le
\lambda[m(Q)+C_B]
-
\sum_i \mu_i^{1-\beta}Q_i
+
C_0.
\]

The same decomposition as in file `01`,

\[
Q_i=\mu_i^\beta m(Q)+\Delta_i(Q),
\qquad \Delta_i(Q)\ge 0,
\]

then yields

\[
(\mathcal L_{\pi^\Theta}V)(Q)
\le
-(\Lambda-\lambda)m(Q)
-
\sum_i \mu_i^{1-\beta}\Delta_i(Q)
+
R_B
\le
-\varepsilon_B |Q|_1 + R_B.
\]

Thus

\[
(\mathcal L_{\pi^\Theta}V)(Q)\le -1
\]

outside the finite set

\[
K_P:=\{Q\in\mathbb Z_+^N:\ \varepsilon_B |Q|_1 \le R_B + 1\}.
\]

By the Foster-Lyapunov criterion for irreducible non-explosive countable-state
CTMCs, the chain is positive Harris recurrent.

`QED`

### Remark 3.4

This theorem covers exact projection only. If the projection is implemented
approximately, if stale probabilities are used, or if the gate is detached from
the actual action probabilities, the proof no longer applies as stated.

---

## 4. What The Gate Proofs Establish

The two theorems above complete the Phase 1 gate program required by the z2
scientific contract for the primary CTMC model:

1. the hard fallback route is a compact-set perturbation of the certified
   backbone,
2. the exact projection route is a pointwise envelope-preserving perturbation
   of the certified backbone.

These are theorem-level inheritance statements for the declared model, not
training heuristics.

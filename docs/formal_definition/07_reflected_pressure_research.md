# Reflected Pressure Research Note

This note compares the legacy CertiQ reflected-pressure dispatcher with the
Reflected MoE formal math package at:

```text
C:\Users\Hellx\Documents\Programming\python\Project\Neryva\moe_route\docs\formal_math
```

The comparison is architectural. It does not claim that MoE routing theorems
directly prove queueing-dispatch stability.

The reflected-pressure idea is used only by the legacy `CertiQDispatcher`
path: pressure enters the learned proposal, is updated between rollout steps,
and leaves the certificate boundary unchanged. The newer `CertiQIndexModel`
does not use this pressure state in its forward path.

## 1. What Neryva Formalizes

The Reflected MoE router is a load-aware expert-routing architecture. Its core
score is:

\[
s_{t,e}=a_{t,e}+b_e-\gamma q_e.
\]

Here:

1. \(a_{t,e}\) is learned token-expert affinity,
2. \(b_e\) is an expert bias,
3. \(q_e\ge 0\) is a nonnegative expert pressure state,
4. \(\gamma q_e\) penalizes experts that have accumulated excess load.

The pressure update is projected:

\[
q^+
=
\Pi_{\mathbb R_+^E}
\left(
(1-\delta)q+\eta(\hat m-\mu)
\right).
\]

\(\hat m\) is routed feedback mass and \(\mu\) is the target expert load. The
projection keeps pressure nonnegative. This makes load control an explicit
router state, not only an auxiliary loss.

The formal package separates three claim levels:

1. implemented finite-batch routing facts,
2. router-level pressure-control theory,
3. end-to-end model framing without claiming full nonconvex training
   convergence.

That separation is stronger than many informal architecture descriptions and
is directly useful for CertiQ.

## 2. What Transfers To CertiQ

The transferable idea is not sparse MoE routing. CertiQ is an online CTMC
dispatch architecture, not a Transformer MoE layer.

The transferable idea is the controller pattern:

\[
\text{affinity} + \text{bias} - \text{nonnegative pressure penalty}.
\]

For CertiQ, a reflected dispatch score takes the form:

\[
u_i^{\mathrm{ref}}
=
u_i^{\mathrm{cert}}
+
r_i^\Theta
-
\rho \psi_i(p_i),
\qquad
p_i\ge 0.
\]

Here:

1. \(u_i^{\mathrm{cert}}\) is the certified base geometry,
2. \(r_i^\Theta\) is the learned proposal correction,
3. \(p_i\) is a resource pressure state,
4. \(\psi_i\) is a monotone pressure penalty,
5. \(\rho>0\) controls how strongly pressure alters scores.

This makes the legacy learned proposal load-aware across time, instead of only
state-aware within one forward pass.

## 3. Why This Could Improve the Legacy Dispatcher

The legacy dispatcher uses:

\[
p_{\mathrm{prop}}
=
\operatorname{softmax}(u^{\mathrm{cert}}+r^\Theta)
\]

and then certifies the final mixture by projection or fallback. This is safe,
but the proposal has no explicit memory of repeated over-selection except
through the current queue vector \(Q\).

A reflected pressure state adds a second control channel:

1. \(Q_i\) represents physical backlog,
2. \(p_i\) represents recent routing pressure or certificate pressure,
3. projection still enforces the Lyapunov envelope.

This matters because a resource can be attractive under instantaneous weighted
queue geometry while still being overused by the learned proposal over recent
rollout windows. Pressure gives the legacy architecture a direct way to suppress
that overuse without relying on a large auxiliary balancing loss.

## 4. Pressure-Aware Proposal

The pressure-aware proposal operator is:

\[
\mathcal A_\Theta^{\mathrm{press}}(Q,\mu,\xi,p)
=
\operatorname{softmax}
\left(
u^{\mathrm{cert}}(Q,\mu)
+
r^\Theta(Q,\mu,\xi)
-
\rho p
\right).
\]

The pressure update uses realized or expected dispatch mass:

\[
p_i^+
=
\max
\left(
0,
(1-\delta)p_i
+
\eta_p(\hat d_i-\bar d_i)
\right).
\]

Possible definitions:

1. \(\hat d_i=\pi_i^\Theta\), the final certified dispatch mass at the current
   state,
2. \(\bar d_i=\mu_i/\sum_j\mu_j\), the capacity-proportional target,
3. \(\eta_p>0\), pressure step size,
4. \(\delta\ge 0\), pressure decay.

Using final certified dispatch mass instead of proposal mass controls the
actual dispatched policy in the legacy path.

## 5. Certificate Interaction

The certificate operator must remain after pressure adjustment:

\[
\pi^\Theta
=
\mathcal C_Q
\left(
\mathcal A_\Theta^{\mathrm{press}}(Q,\mu,\xi,p),
\mathcal B(Q,\mu)
\right).
\]

This ordering is important.

Pressure may improve load behavior, but pressure is not itself the CTMC
certificate. The legacy final policy must still satisfy:

\[
A_{\pi^\Theta}(Q)\le m(Q)+C.
\]

Therefore reflected pressure is a proposal-control mechanism. It is not a
replacement for envelope projection or fallback.

## 6. Theorem Obligations

Before the pressure controller is treated as theorem-bearing architecture,
these obligations must be separated:

1. **Finite-step invariants**:
   prove \(p_i\ge 0\), policy normalization, and certificate preservation after
   projection.
2. **Certificate inheritance**:
   show that adding pressure before \(\mathcal C_Q\) cannot break the existing
   projection or fallback theorem.
3. **Pressure dynamics**:
   analyze the idealized population update with fixed proposal parameters and
   stationary state distribution.
4. **No overclaim**:
   do not claim global convergence of the learned neural policy or optimality
   of the pressure controller.

The most defensible theorem is simple:

> If the certificate operator returns an admissible policy for every proposal
> distribution, then replacing the proposal with a reflected-pressure proposal
> preserves certification.

This theorem keeps pressure as an architecture component while leaving the
legacy CTMC certificate boundary unchanged.

## 7. What Does Not Transfer

The following Neryva components do not directly transfer to CertiQ:

1. token top-k expert selection,
2. MoE sparse capacity admission,
3. shared experts in the Transformer feed-forward layer,
4. z-loss on raw language-model router logits,
5. language-model transfer claims.

They may inspire experiments, but they are not CertiQ architecture facts.

## 8. Recommended Research Direction

The implemented pressure architecture is:

**CertiQ Dispatcher with Reflected Pressure Proposal**

\[
\boxed{
\pi^\Theta(Q,\mu,\xi,p)
=
\mathcal C_Q
\left(
\operatorname{softmax}
\left[
u^{\mathrm{cert}}
+
r^\Theta
-
\rho p
\right],
\mathcal B(Q,\mu)
\right)
}
\]

with projected pressure update:

\[
p^+
=
\Pi_{\mathbb R_+^N}
\left(
(1-\delta)p+\eta_p(\pi^\Theta-\mu/\Lambda)
\right).
\]

This gives every component a role:

1. certified base geometry gives a stable reference,
2. neural residual gives performance adaptation,
3. reflected pressure gives temporal load control,
4. certificate projection gives hard admissibility,
5. diagnostics expose both certificate slack and pressure behavior.

This is the implemented pressure-aware extension of the legacy CertiQ
dispatcher.

## 9. External Research Context

The Neryva formal package is consistent with the MoE literature direction:

1. Shazeer et al. introduced sparse-gated MoE routing.
2. GShard and Switch Transformer made sparse token-choice routing practical at
   large Transformer scale.
3. ST-MoE emphasized router stabilization, including z-loss.
4. Expert Choice routing changed assignment direction by letting experts choose
   token buckets.
5. Soft MoE replaced hard dispatch with soft token mixtures.
6. Auxiliary-loss-free balancing and DeepSeek-style bias adjustment move load
   control away from large auxiliary balancing gradients.

For CertiQ, the strongest inspiration is the last point: load control should be
part of the routing mechanism, not merely a loss term.

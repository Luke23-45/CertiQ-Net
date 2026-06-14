# Training And Objectives

## 1. Separation Of Roles

Training selects parameters. Certification defines admissible actions. The two
roles are distinct and must not be conflated.

Any training objective may improve rollout cost, imitation quality, entropy, or
proposal usefulness, but it does not itself certify the policy.

## 2. Objective Decomposition

The training loss may be written as

\[
\mathcal L(\Theta)
=
\omega_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+\omega_{\mathrm{ce}}\mathcal L_{\mathrm{ce}}
+\omega_{\mathrm{margin}}\mathcal L_{\mathrm{margin}}
+\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+\omega_{\mathrm{cert}}\mathcal L_{\mathrm{cert}}^{\mathrm{prop}},
\]

where the terms denote rollout cost, cross-entropy behavior-cloning loss,
heuristic ranking margin loss, entropy regularization, and the proposal-level
certificate penalty.

## 3. Certificate Penalty

The certificate penalty should be defined at the proposal level:

\[
\mathcal L_{\mathrm{cert}}^{\mathrm{prop}}
=
\mathbb E\!\left[\bigl(A_{q_\Theta}(Q,\mu)-B(Q,\mu)\bigr)_+^2\right].
\]

If the projection layer is active and numerically exact, the analogous
final-policy penalty is degenerate because the certified policy already
satisfies the budget by construction.

The proposal-level penalty is the useful training signal because it can remain
nonzero when the raw proposal violates the budget.

### 3.1 Final-Policy Degeneracy

Assume the certificate layer returns a feasible policy \(\pi^\star(x)\)
satisfying

\[
A_{\pi^\star}(x)\le B(x)
\]

for every state \(x\) in its domain. Then the final-policy penalty

\[
\mathcal L_{\mathrm{cert}}^{\mathrm{final}}(\Theta)
=
\mathbb E\!\left[\bigl(A_{\pi^\star}(x)-B(x)\bigr)_+^2\right]
\]

is identically zero whenever the certificate layer succeeds.

## 4. Proposal Learning

The learned proposal may be trained as a residual over the QMD geometry:

\[
\hat I_i(Q,\mu,\xi)=d_i^{QMD}(Q,\mu)+r_i^\Theta(Q,\mu,\xi).
\]

Soft SED or QMD targets may be used for warm-start supervision. Such targets are
derived from the observed state and service rates rather than from an external
labeled dataset.

## 5. Evaluation Quantities

Evaluation quantities include performance and certificate measurements.

Performance quantities include average backlog, tail backlog quantiles, and
maximum observed backlog.

Certificate quantities include violation rate, minimum slack, average slack,
projection activation rate, fallback activation rate, and correction magnitude.

No empirical result is certified unless the certificate quantities are
reported.

## 6. Curriculum

A consistent curriculum is:

1. verify the geometry and budget definitions,
2. train the proposal against QMD targets,
3. activate exact certificate enforcement,
4. optimize rollout cost under the certified policy,
5. report certificate metrics together with performance metrics.

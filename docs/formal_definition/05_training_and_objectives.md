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
+\omega_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}
+\omega_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+\omega_{\mathrm{cert}}\mathcal L_{\mathrm{cert}},
\]

where the terms denote rollout cost, behavior-cloning or distillation loss,
residual regression, entropy regularization, and certificate regularization.

## 3. Certificate Penalty

The certificate penalty may be defined as

\[
\mathcal L_{\mathrm{cert}}
=
\mathbb E\!\left[\bigl(A_{q_\Theta}(Q,\mu)-B(Q,\mu)\bigr)_+^2\right].
\]

If the projection layer is active and numerically exact, this term is zero up
to tolerance.

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

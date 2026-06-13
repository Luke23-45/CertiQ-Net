# Training And Objectives

## 1. Training Is Not Certification

Training selects useful parameters. Certification defines admissible actions.
Those roles must remain separate.

The training objective may improve cost, imitation quality, exploration,
entropy, and proposal usefulness. It must not be used as the basis for a
stability claim unless the final policy is also certified by construction.

## 2. Objective Hierarchy

The training objective should be organized by priority:

1. preserve exact certificate enforcement in the forward path,
2. reduce average queueing cost,
3. learn useful proposal corrections over the base geometry,
4. preserve resource permutation equivariance,
5. keep diagnostics auditable and non-silent.

This hierarchy prevents the learned proposal from becoming a hidden replacement
for the certificate layer.

## 3. Loss Components

The training loss may include
\[
\mathcal L
=
\omega_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+\omega_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}
+\omega_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+\omega_{\mathrm{usage}}\mathcal L_{\mathrm{usage}}
+\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+\omega_{\mathrm{cert}}\mathcal L_{\mathrm{cert}}.
\]

Typical interpretations:

1. \(\mathcal L_{\mathrm{roll}}\): rollout cost,
2. \(\mathcal L_{\mathrm{bc}}\): imitation or distillation loss,
3. \(\mathcal L_{\mathrm{res}}\): residual or value-index regression,
4. \(\mathcal L_{\mathrm{usage}}\): proposal usage regularization,
5. \(\mathcal L_{\mathrm{ent}}\): entropy regularization,
6. \(\mathcal L_{\mathrm{cert}}\): certificate penalty.

The certificate penalty is
\[
\mathcal L_{\mathrm{cert}}
=
\mathbb E[(A_{\pi^\Theta}(Q)-B(Q))_+^2].
\]

When exact projection is active, this term should be zero up to numerical
tolerance.

## 4. Proposal Learning

The learned proposal should be trained as a correction over the certified base
or as a marginal-cost index, not as an independent dispatcher.

The current implementation centers on quadratic-min-drift targets.
Soft or hard SED targets remain comparison baselines, and small-system oracle
targets may be used when available.

For the oracle targets, the relevant label is the action-value difference
\[
\Delta V_i(Q)=V(Q+e_i)-V(Q),
\]
with the target action
\[
i^\star_{oracle}=\arg\min_i \Delta V_i(Q).
\]

The legacy dispatcher also uses an internal pressure state during training and
evaluation. That pressure state is a nonnegative controller memory that shifts
proposal logits away from repeatedly preferred resources while leaving the
certificate boundary unchanged.

## 5. Evaluation

Evaluation must report both performance and certificate metrics.

Performance metrics:

1. average queue length,
2. average cost,
3. p95 backlog,
4. p99 backlog,
5. maximum observed backlog,
6. latency proxy where applicable.

For learned models, greedy argmax evaluation is the canonical deployment
comparison mode, and stochastic sampling remains a training-time mechanism.

Certificate metrics:

1. violation rate,
2. minimum certificate slack,
3. average certificate slack,
4. projection activation rate,
5. fallback activation rate,
6. proposal usage,
7. residual magnitude,
8. pressure statistics when applicable.

No result should be presented as certified without certificate metrics.

## 6. Ablations

Ablations should test the architecture thesis:

1. base policy only,
2. proposal without certificate,
3. certificate with weak proposal,
4. projection versus fallback,
5. equivariant proposal versus order-dependent proposal,
6. context-free versus context-aware proposal,
7. SED versus quadratic-min-drift geometry.

The uncertified ablation, when used for comparison, must be labeled as
uncertified.

## 7. Curriculum

Recommended curriculum:

1. verify base certificate quantities,
2. train the proposal to imitate QMD,
3. enable exact certificate enforcement during all policy rollouts,
4. optimize rollout cost,
5. audit state banks before reporting results,
6. only then expand to approximate adapters or richer domains.

For the index model, a practical curriculum is:

1. initialize from the quadratic drift index,
2. distill soft QMD behavior,
3. activate KL projection during all evaluations,
4. keep the fixed QMD budget active outside the warm-start phase,
5. reduce entropy only after the projection operator is stable.

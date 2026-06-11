# Training And Objectives

## 1. Training Is Not Certification

Training selects useful parameters. The certificate layer defines admissible
actions. These roles must remain separate.

The training objective may improve cost, imitation quality, exploration,
entropy, and proposal usefulness. It must not be used as the basis for a
stability claim unless the final policy is also certified by construction.

## 2. Objective Hierarchy

The z3 training objective should be organized by priority:

1. maintain exact certificate enforcement in the forward path,
2. reduce average queueing cost,
3. learn useful proposal corrections over the certified base geometry,
4. preserve resource permutation equivariance,
5. keep diagnostics auditable and non-silent.

This hierarchy prevents the learned proposal from becoming a hidden replacement
for the certified architecture.

## 3. Suggested Loss Components

The training loss may include:

\[
\mathcal L
=
\omega_{\mathrm{roll}}\mathcal L_{\mathrm{roll}}
+
\omega_{\mathrm{bc}}\mathcal L_{\mathrm{bc}}
+
\omega_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+
\omega_{\mathrm{usage}}\mathcal L_{\mathrm{usage}}
+
\omega_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
+
\omega_{\mathrm{cert}}\mathcal L_{\mathrm{cert}}.
\]

The certificate penalty:

\[
\mathcal L_{\mathrm{cert}}
=
\mathbb E[(A_{\pi^\Theta}(Q)-B(Q))_+^2]
\]

should be zero when exact projection is active. If it is not zero, the
implementation is wrong or the numerical tolerance is incorrectly declared.

## 4. Proposal Learning

The learned proposal should be trained as a correction over the certified base,
not as an independent dispatcher.

The current implementation uses a reflected pressure state during training and
evaluation. The pressure state is a nonnegative controller memory that shifts
proposal logits away from repeatedly preferred resources, while the
certificate operator still decides admissibility.

Acceptable proposal targets include:

1. rollout-improved certified policy,
2. dynamic-programming policy on tiny systems,
3. audited queueing heuristic,
4. cost-improving residual discovered by on-policy learning.

The target must be labeled. A heuristic target is not a theorem.

## 5. Evaluation

Evaluation must report both performance and certificate metrics.

Performance metrics:

1. average queue length,
2. average cost,
3. tail backlog quantiles,
4. maximum observed backlog,
5. latency proxy where applicable.

Certificate metrics:

1. violation rate,
2. minimum drift slack,
3. average drift slack,
4. projection activation rate,
5. fallback activation rate,
6. proposal usage,
7. residual magnitude.

No result should be presented as certified without certificate metrics.

## 6. Ablations

Ablations should test the architecture thesis:

1. base policy only,
2. proposal without certificate,
3. certificate with weak proposal,
4. projection versus fallback,
5. equivariant proposal versus order-dependent proposal,
6. context-free versus context-aware proposal.

The uncertified ablation, when used for comparison, must be labeled as
uncertified.

## 7. Curriculum

Recommended curriculum:

1. verify base certificate quantities,
2. train proposal to imitate base or audited heuristic,
3. enable exact certificate operator during all policy rollouts,
4. optimize rollout cost,
5. audit state banks before reporting results,
6. only then expand to approximate adapters or richer domains.

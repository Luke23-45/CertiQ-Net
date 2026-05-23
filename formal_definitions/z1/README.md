# CertiQ-Net Formal Definitions

This directory is the first formal architecture package for CertiQ-Net.

The purpose is not to declare a finished theorem before the work exists. The
purpose is to define a standalone architecture paper: a neural queueing
controller whose residual decisions are filtered by an explicit stability
certificate.

## Core Position

CertiQ-Net is a certifiably constrained neural routing architecture for
heterogeneous parallel-server queues. Its central design rule is:

> Learn only inside a stability envelope that contains a proved or audited
> stable analytic base controller.

The analytic base controller is defined inside this paper:

\[
p_i^{\mathrm{base}}(q)
=
\frac{\mu_i^\gamma \exp(-\alpha(q_i+c)/\mu_i^\beta)}
{\sum_j \mu_j^\gamma \exp(-\alpha(q_j+c)/\mu_j^\beta)}.
\]

The neural layer may improve performance, adapt across systems, and learn from
differentiable simulation, but it is not allowed to become an unconstrained
black-box dispatcher unless a separate stability certificate is supplied.

## Files

1. [00_claim_discipline.md](./00_claim_discipline.md)
   Defines what can and cannot be claimed at this stage.
2. [01_queueing_model.md](./01_queueing_model.md)
   Defines the stochastic queueing system, state, actions, objective, and
   stability target.
3. [02_certiq_net_architecture.md](./02_certiq_net_architecture.md)
   Gives the neural architecture: invariant encoders, analytic base controller,
   residual gate, certificate projection, and output policy.
4. [03_training_objectives.md](./03_training_objectives.md)
   Defines differentiable-simulation training losses, imitation warm starts,
   safety regularizers, and optimization protocol.
5. [04_certification_route.md](./04_certification_route.md)
   Defines the formal stability envelope and the proof obligations needed to
   promote CertiQ-Net from architecture to theorem.
6. [05_experiment_protocol.md](./05_experiment_protocol.md)
   Defines the architecture-ablation program and the evidence thresholds.
7. [06_research_references.md](./06_research_references.md)
   Records the external research anchors checked while drafting this package.

## Non-Negotiable Scientific Boundary

The current package may safely claim:

- CertiQ-Net defines a smooth analytic base controller.
- CertiQ-Net defines a permutation-equivariant neural residual.
- CertiQ-Net defines a certificate gate and a Lyapunov projection layer.
- A neural residual can be trained around the analytic base controller.
- Certification must be explicit; it does not follow automatically from neural
  smoothness.

The current package must not claim:

- CertiQ-Net is already state-of-the-art empirically.
- CertiQ-Net is globally stable for arbitrary neural parameters.
- Differentiable simulation alone proves stochastic stability.
- Classical dispatch-rule comparisons are the novelty of this paper.

## Architecture Summary

The policy is

\[
\pi_\Theta(q,\xi)
=
(1-\eta_\Theta(q,\xi))p^{\mathrm{base}}_{\phi_\Theta(\xi)}(q)
+
\eta_\Theta(q,\xi)p^{\mathrm{nn}}_\Theta(q,\xi),
\]

where:

- \(q\in\mathbb Z_+^N\) is the queue state,
- \(\xi\) contains system descriptors such as service rates and optional
  hardware/context features,
- \(p^{\mathrm{base}}\) is the analytic softmax base controller,
- \(p^{\mathrm{nn}}\) is a permutation-equivariant neural routing proposal,
- \(\eta_\Theta\in[0,\eta_{\max}]\) is a learned or projected residual gate.

Certification is attempted through one of two routes:

1. prove the mixed policy remains inside a Foster-Lyapunov drift envelope, or
2. set \(\eta=0\) outside a certified compact performance region, forcing the
   tail behavior to equal the stable base controller.

The second route is the conservative default because it is realistic.

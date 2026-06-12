# CertiQ Formal Definition

This directory documents the implemented CertiQ dispatch family and keeps the
formal description aligned with the code that is actually running.

The repository currently ships two implemented certified dispatcher paths:

1. `CertiQDispatcher`
   - legacy reflected-pressure architecture,
   - analytic base geometry,
   - scalar usage cap or fallback certification.
2. `CertiQIndexModel`
   - learned marginal-cost index architecture,
   - SED/QMD-aligned geometry,
   - exact KL projection onto a delay-aligned budget with finite-region SED fallback.

The formal chapters below describe the common queueing model, the dispatcher
family, the certificate layer, the training objectives, and the reflected
pressure extension.

## Reading Order

1. [01_architecture_principles.md](./01_architecture_principles.md)
2. [02_formal_dispatch_model.md](./02_formal_dispatch_model.md)
3. [03_certiq_dispatcher_architecture.md](./03_certiq_dispatcher_architecture.md)
4. [04_certificate_layer.md](./04_certificate_layer.md)
5. [05_training_and_objectives.md](./05_training_and_objectives.md)
6. [06_research_references.md](./06_research_references.md)
7. [07_reflected_pressure_research.md](./07_reflected_pressure_research.md)

## Canonical Thesis

The canonical implementation pattern is a certified dispatch operator:

\[
\pi^\Theta
=
\mathcal C_Q\!\left(
\mathcal A_\Theta(Q,\mu,\xi,p),
\mathcal B(Q,\mu)
\right),
\]

where \(\mathcal B\) is the base geometry, \(\mathcal A_\Theta\) is the
learned proposal, \(p\) is an optional reflected pressure state, and
\(\mathcal C_Q\) is the certificate operator.

For the newer index model, the final policy is the KL projection of a learned
marginal-cost proposal onto a delay-aligned budget, with a certified SED tail
fallback outside the learned region. For the legacy dispatcher, the final
policy is a scalar mixture of a certified base policy and a learned proposal,
capped by the certificate layer.

## Relationship To z2

The z2 material remains the proof package for the legacy analytic base model:

- [../../z2/formal_math/01_backbone_stability_and_constant.md](../../z2/formal_math/01_backbone_stability_and_constant.md)
- [../../z2/formal_math/02_gate_inheritance_theorems.md](../../z2/formal_math/02_gate_inheritance_theorems.md)

The new index model is implemented in code, but any global stability claim for
that model remains a theorem obligation unless separately proved.

## Non-Goals

This directory does not re-prove the queueing theorems.

It also does not claim that the index model inherits the z2 proof package by
default. Only the explicitly stated certificate boundaries are guaranteed.

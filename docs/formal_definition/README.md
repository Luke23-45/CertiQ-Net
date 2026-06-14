# CertiQ Formal Definition

This directory documents the implemented CertiQ dispatch architecture and keeps
the formal description aligned with the code that is actually running.

The repository ships one certified dispatch model:

- `CertiQIndexModel`
  - learned marginal-cost index architecture,
  - SED/QMD-aligned geometry,
  - exact KL projection onto a fixed QMD budget.

The formal chapters below describe the common queueing model, the dispatcher
architecture, the certificate layer, and the training objectives.

## Reading Order

1. [01_architecture_principles.md](./01_architecture_principles.md)
2. [02_formal_dispatch_model.md](./02_formal_dispatch_model.md)
3. [03_certiq_index_architecture.md](./03_certiq_index_architecture.md)
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
\mathcal A_\Theta(Q,\mu,\xi),
\mathcal B(Q,\mu)
\right),
\]

where \(\mathcal B\) is the base geometry, \(\mathcal A_\Theta\) is the
learned proposal, and \(\mathcal C_Q\) is the certificate operator.

The final policy is the KL projection of a learned marginal-cost proposal
onto a fixed QMD budget.

## Relationship To z2

The z2 material remains the proof package for the legacy analytic base model:

- [../../z2/formal_math/01_backbone_stability_and_constant.md](../../z2/formal_math/01_backbone_stability_and_constant.md)
- [../../z2/formal_math/02_gate_inheritance_theorems.md](../../z2/formal_math/02_gate_inheritance_theorems.md)

The index model is implemented in code, but any global stability claim for
that model remains a theorem obligation unless separately proved.

## Non-Goals

This directory does not re-prove the queueing theorems.

It also does not claim that the index model inherits the z2 proof package by
default. Only the explicitly stated certificate boundaries are guaranteed.

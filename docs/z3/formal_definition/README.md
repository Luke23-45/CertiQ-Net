# CertiQ Dispatcher Formal Definition

This directory defines the z3 CertiQ Dispatcher architecture.

The purpose of z3 is not to patch the old model names. It is to state the
architecture from first principles before any implementation rewrite happens.
The core object is a certified online dispatcher for assigning arriving work to
heterogeneous capacity-constrained resources.

## Reading Order

1. [01_architecture_principles.md](./01_architecture_principles.md)
2. [02_formal_dispatch_model.md](./02_formal_dispatch_model.md)
3. [03_certiq_dispatcher_architecture.md](./03_certiq_dispatcher_architecture.md)
4. [04_certificate_layer.md](./04_certificate_layer.md)
5. [05_training_and_objectives.md](./05_training_and_objectives.md)
6. [06_research_references.md](./06_research_references.md)

## z3 Thesis

CertiQ Dispatcher is a certified differentiable architecture for online
assignment:

\[
\pi^\Theta(Q,\mu,\xi)
=
\mathcal C_Q\left(
\mathcal A_\Theta(Q,\mu,\xi),
\mathcal B(Q,\mu)
\right).
\]

Here:

1. \(\mathcal B\) is the certified base dispatch geometry,
2. \(\mathcal A_\Theta\) is the learned assignment proposal,
3. \(\mathcal C_Q\) is the state-dependent certificate operator,
4. \(\pi^\Theta\) is the final dispatch distribution.

The architecture is not defined by preview implementation variants. It is the
composition of base geometry, learned proposal, and certificate operator.

## Relationship To z2

The z2 material remains the current proof package for the CTMC model:

- [../../z2/formal_math/01_backbone_stability_and_constant.md](../../z2/formal_math/01_backbone_stability_and_constant.md)
- [../../z2/formal_math/02_gate_inheritance_theorems.md](../../z2/formal_math/02_gate_inheritance_theorems.md)

z3 reuses proven z2 facts where they apply, but it changes the architecture
language. In z3, certificate mechanisms are first-class operators, not gates
attached after a neural policy has already been designed.

## Non-Goals

This directory does not implement the rewrite.

It also does not prove new stability theorems beyond the z2 proof package. New
claims introduced by z3 must remain theorem obligations until separately
proved.

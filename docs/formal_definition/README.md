# CertiQ Formal Definition

This directory gives the canonical formal specification of the CertiQ
dispatch family.

The central object is the certified dispatch operator

\[
\pi^\Theta(x)
=
\mathcal C_x\!\left(
\mathcal A_\Theta(x),
\mathcal B(x)
\right),
\]

where \(\mathcal A_\Theta\) is the learned proposal map, \(\mathcal B\) is the
base geometry, and \(\mathcal C_x\) is the certificate operator at state
\(x\).

## Reading Order

1. [01_architecture_principles.md](./01_architecture_principles.md)
2. [02_formal_dispatch_model.md](./02_formal_dispatch_model.md)
3. [03_certiq_index_architecture.md](./03_certiq_index_architecture.md)
4. [04_certificate_layer.md](./04_certificate_layer.md)
5. [05_training_and_objectives.md](./05_training_and_objectives.md)
6. [06_research_references.md](./06_research_references.md)
7. [07_reflected_pressure_research.md](./07_reflected_pressure_research.md)

Chapter 7 is supplementary and not part of the canonical index specification.

## Canonical Model

The canonical CertiQ index model uses:

\[
d_i^{QMD}(Q,\mu)=\frac{2Q_i+1}{\mu_i},
\qquad
\pi^\Theta=\mathcal C_x\!\left(\mathcal A_\Theta(x),\mathcal B(Q,\mu)\right),
\]

with \(\mathcal C_x\) defined as an exact KL projection onto a state-dependent
certificate budget.

## Scope

This directory defines the architecture, state model, symmetry requirement,
certificate boundary, and training objective. It does not itself prove long-run
stability or optimality.

## Relationship To z2

The z2 material remains the proof package for the project-local analytic base model:

- [../../z2/formal_math/01_backbone_stability_and_constant.md](../../z2/formal_math/01_backbone_stability_and_constant.md)
- [../../z2/formal_math/02_gate_inheritance_theorems.md](../../z2/formal_math/02_gate_inheritance_theorems.md)

Any stability claim for the learned index model requires a separate theorem
statement unless it is explicitly inherited by a stated certificate result.

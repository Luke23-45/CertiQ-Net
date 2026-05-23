# Claim Discipline

This file defines the paper boundary for CertiQ-Net.

CertiQ-Net is a new architecture paper. It is not a rerun of classical routing
comparisons and it is not positioned as a continuation paper whose novelty is
another comparison table. The contribution is the architecture: a neural routing
policy whose expressive component is constrained by an explicit certificate
gate.

## 1. Claim Ladder

### Tier A: architecture definitions

These claims are definitions introduced by this paper.

- CertiQ-Net combines an analytic softmax base controller, an equivariant
  neural residual, and a certificate gate.
- The neural residual is permutation equivariant over server labels.
- The certificate gate can force the neural policy to revert to the analytic
  base controller outside a declared safe region.
- The projection layer can enforce a Lyapunov drift envelope at inference time.

### Tier B: theorem targets

These claims are mathematical targets for this paper.

- Tail fallback gives a clean Foster-Lyapunov certification route: if the
  analytic base controller has negative drift outside a compact set and the
  neural residual is disabled outside that set, the full architecture inherits
  the same tail drift.
- Drift projection gives a stronger route: if the projection enforces the
  weighted-quadratic arrival envelope pointwise, the neural policy can remain
  active while preserving the Lyapunov inequality.
- These theorem targets require explicit proof; they do not follow merely from
  differentiability or neural smoothness.

### Tier C: empirical questions

The experiments should answer architecture questions, not replay a
dispatch-rule comparison landscape.

- Does the neural residual improve over the analytic base controller under the
  same certificate envelope?
- How much performance is lost when the certificate gate is tightened?
- Does the projection layer prevent drift violations by construction?
- Does the equivariant architecture transfer across system sizes and
  heterogeneous service-rate profiles?
- Do pathwise gradients from differentiable simulation train the gated residual
  more efficiently than score-function estimators?

## 2. Removed From This Paper

The following are not the novelty axis of CertiQ-Net:

- broad comparison against classical dispatch rules,
- re-establishing that the analytic base controller is competitive,
- treating classical policies as the central empirical story,
- presenting the architecture as a minor add-on to another project.

Those topics may be cited as background or placed in a short related-work
paragraph, but they should not define the experimental agenda.

## 3. Forbidden Claims

The paper must not claim:

- "The neural policy is stable because it is smooth."
- "The architecture is certified because the base controller is certified."
- "The residual is harmless for all states."
- "Differentiable simulation proves stability."
- "CertiQ-Net is state of the art" before the architecture-specific protocol
  supports that statement.
- "The paper's contribution is another dispatch-rule comparison."

## 4. Correct Headline

The defensible headline is:

> CertiQ-Net is a stability-constrained neural architecture for queueing
> control: an equivariant residual policy whose actions are filtered by a
> certificate gate or Lyapunov projection layer.

This framing makes CertiQ-Net a standalone research paper.

## 5. Research Contract

Every theorem, experiment, and implementation should answer one of these
questions:

1. Does the certificate gate preserve the Lyapunov tail condition?
2. Does the residual improve the analytic base controller inside the safe
   region?
3. Does the projection layer enforce the drift envelope pointwise?
4. Does the equivariant model generalize across \(N\), service rates, and load?
5. Does differentiable simulation make the residual trainable at lower
   gradient variance?

If a result does not address one of these questions, it belongs outside the
main CertiQ-Net paper.

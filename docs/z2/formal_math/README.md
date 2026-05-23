# CertiQ-Net z2 Formal Math Package

This directory is the Phase 1 mathematical package for the z2 CertiQ-Net
program. Its purpose is to complete the proof-facing prerequisites required by
Section 15 and Section 17 of the z2 end-to-end formal definition before neural
training claims are made.

The package is intentionally narrow:

1. prove the analytic-backbone arrival envelope,
2. compute the exact envelope constant \(C_B\),
3. close the weighted-quadratic Foster-Lyapunov drift argument for the
   backbone CTMC,
4. prove inheritance of the same stability route for the hard tail fallback
   gate,
5. prove inheritance of the same stability route for the exact drift-envelope
   projection gate,
6. record a verified literature review for the external sources that may be
   cited around these proofs.

This package is self-contained as a CertiQ-Net z2 artifact. It may reuse proof
ideas that also appear in the GibbsQ z2 formal-math directory, but every claim
needed for CertiQ-Net is restated here in CertiQ-Net notation and with CertiQ-
Net theorem boundaries.

## Files

1. [01_backbone_stability_and_constant.md](./01_backbone_stability_and_constant.md)
   Exact backbone arrival envelope, exact constant \(C_B\), generator drift
   decomposition, and positive Harris recurrence theorem for the analytic
   backbone.
2. [02_gate_inheritance_theorems.md](./02_gate_inheritance_theorems.md)
   Formal inheritance theorems for the hard tail fallback gate and the exact
   drift-envelope projection gate.
3. [03_verified_literature_review.md](./03_verified_literature_review.md)
   Proofread literature review with only verified external references and a
   strict statement of what each source does and does not support.

## Scientific Position

The results in files `01` and `02` are the mathematical core required to
support the z2 certification boundary for the CTMC dispatch model. They justify
the following CertiQ-Net claims and only the following claims:

1. the analytic backbone satisfies an explicit minimum-type arrival envelope,
2. the constant \(C_B\) is computable exactly from the declared backbone
   parameters and resource capacities,
3. the analytic backbone CTMC is non-explosive, irreducible, and positive
   Harris recurrent whenever \(\lambda < \Lambda\),
4. the hard tail fallback gate inherits backbone stability,
5. the exact drift-envelope projection gate inherits backbone stability.

This package does not prove:

1. stability for arbitrary neural residuals,
2. stability for approximate projection,
3. stability for the smooth surrogate gate used only for training,
4. CTMC certification for domain adapters that violate the Section 2 model
   assumptions in the z2 end-to-end definition.

## Reading Order

Read:

1. [01_backbone_stability_and_constant.md](./01_backbone_stability_and_constant.md)
2. [02_gate_inheritance_theorems.md](./02_gate_inheritance_theorems.md)
3. [03_verified_literature_review.md](./03_verified_literature_review.md)

That order mirrors the dependency structure of the proofs.

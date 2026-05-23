# CertiQ-Net z2 Formal Definitions

This directory contains the second research direction for CertiQ-Net.

The z2 direction reframes CertiQ-Net as a Certified Differentiable Dispatcher:
a general architecture for assigning discrete work to heterogeneous,
capacity-constrained resources while preserving an explicit stability
certificate.

## Files

1. [main_strict_directions.md](./main_strict_directions.md)
   Research-direction memo for the z2 architecture.
2. [end_to_end_formal_definition.md](./end_to_end_formal_definition.md)
   Formal architecture definition, theorem boundaries, certificate mechanisms,
   training protocol, domain adapters, implementation invariants, and proof
   checklist.
3. [formal_math/README.md](./formal_math/README.md)
   Phase 1 proof package: backbone envelope and exact constant, gate
   inheritance theorems, and verified literature review.

## Certification Boundary

The primary theorem target remains the CTMC dispatch model with Poisson
arrivals, exponential service completions, heterogeneous rates, and online
routing decisions. Other domains, such as mixture-of-experts routing or
multi-robot task allocation, inherit the theorem only when their adapter
satisfies the same model assumptions or supplies a separate proof.

## Architecture Summary

The z2 CertiQ-Net policy is:

\[
\pi^\Theta
=
(1-\eta)p^{\mathrm{base}}
+
\eta p^{\mathrm{nn}},
\]

where \(p^{\mathrm{base}}\) is the analytic stable backbone,
\(p^{\mathrm{nn}}\) is a backbone-plus-bounded-residual neural proposal, and
\(\eta\) is certified by either hard tail fallback or pointwise Lyapunov
drift-envelope projection.

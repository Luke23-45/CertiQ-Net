# Research References

This file records external and project-local sources that support the formal
definitions in this directory.

## 1. Set Processing And Permutation Symmetry

Relevant sources:

- Ashish Vaswani et al., "Attention Is All You Need," 2017.
- Manzil Zaheer et al., "Deep Sets," NeurIPS 2017.
- Juho Lee et al., "Set Transformer," ICML 2019.

Supported use:

These works motivate shared local maps, invariant aggregation, and
equivariant set processing.

Unsupported use:

They do not prove queueing stability, certificate validity, or any CertiQ
dispatch guarantee.

## 2. Queueing Stability Language

Relevant sources:

- Sean P. Meyn and Richard L. Tweedie, *Markov Chains and Stochastic Stability*.
- Leandros Tassiulas and Anthony Ephremides, "Stability Properties of
  Constrained Queueing Systems and Scheduling Policies for Maximum Throughput
  in Multihop Radio Networks," 1992.

Supported use:

These works provide the standard language for drift, recurrence, and
queue-aware scheduling.

Unsupported use:

They do not by themselves prove the CertiQ index model or its KL projection.

## 3. Delay Geometry

Relevant sources:

- Classical shortest-expected-delay routing results for heterogeneous queues.
- Comparative heterogeneous-server routing literature using SED-style rules.

Supported use:

These sources justify SED and QMD as comparison geometries.

Unsupported use:

They do not establish optimality of the learned proposal.

## 4. Exact Projection

The KL projection in the CertiQ index model is a state-dependent convex
optimization operator. It supplies a clean certificate boundary, but it does
not itself imply positive recurrence or optimality.

## 5. Differentiable Convex Optimization Layers

Relevant sources:

- Brandon Amos and J. Zico Kolter, "OptNet: Differentiable Optimization as a
  Layer in Neural Networks," 2017.
- C. Katyal, "Differentiable Convex Optimization Layers in Neural Architectures:
  Foundations and Perspectives," 2024.

Supported use:

These works support embedding exact convex optimization operators, such as the
CertiQ KL projection, directly into the forward path.

Unsupported use:

They do not provide the queueing geometry, dispatch certificate, or subcritical
load statements required by CertiQ.

## 6. Project-Local Proof Package

Relevant sources:

- `docs/z2/formal_math/01_backbone_stability_and_constant.md`
- `docs/z2/formal_math/02_gate_inheritance_theorems.md`

Supported use:

These files support the analytic base envelope and its certified inheritance
statements.

Unsupported use:

They do not automatically extend to arbitrary learned residuals or adapters
that violate the stated CTMC model.

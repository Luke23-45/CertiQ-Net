# Architecture Principles

## 1. Design Target

CertiQ Dispatcher is designed for one primitive:

> assign one arriving unit of work to one resource, online, under heterogeneous
> service capacity, while preserving an auditable stability certificate.

The architecture should be judged by whether each component helps this
primitive. A component is valid only if it has a clear role in at least one of:

1. representing resources without depending on arbitrary labels,
2. comparing current workload against capacity,
3. proposing a performance-improving assignment,
4. enforcing a certificate before dispatch,
5. exposing diagnostics that allow the certificate to be audited.

This replaces the old pattern of adding a backbone, neural residual, and gate
as separate named pieces without a single architectural invariant.

## 2. Architectural Discipline

The lesson to borrow from the Transformer is not the attention mechanism
itself. The lesson is architectural compression: define a small number of
operations that directly match the task primitive, then reuse those operations
consistently.

For CertiQ Dispatcher, the primitive is not token-to-token contextualization.
It is capacity-aware assignment. Therefore the central operation is:

\[
\text{certified assignment}
=
\text{proposal}
\quad\text{filtered through}\quad
\text{state-dependent admissibility}.
\]

Every z3 block must answer:

1. What mathematical object does this block produce?
2. What invariance or constraint does it preserve?
3. What failure mode does it prevent?
4. What diagnostic can verify that it behaved correctly?

## 3. Three-Layer Architecture

The architecture has three layers.

### 3.1 Geometry Layer

The geometry layer maps queue state and capacity into a stable dispatch energy.
It defines the coordinates in which backlog is compared:

\[
y_i(Q)=Q_i/\mu_i^\beta.
\]

It also defines the certified base policy \(p^{\mathrm{cert}}\). This policy
must have a theorem-facing arrival envelope:

\[
A_{p^{\mathrm{cert}}}(Q)\le m(Q)+C.
\]

### 3.2 Proposal Layer

The proposal layer learns performance corrections. It may use context
\(\xi\), attention, pooling, learned costs, compatibility features, or other
domain signals.

The proposal layer is not the source of certification. It is the source of
adaptivity and performance.

### 3.3 Certificate Layer

The certificate layer maps an unconstrained or partially constrained proposal
into a certified final distribution.

The certificate layer is where the architecture earns the name CertiQ
Dispatcher. If the final policy violates the certificate, the architecture has
failed, even if training loss improves.

## 4. Required Invariances

Resource labels have no intrinsic meaning. The architecture must be
permutation equivariant:

\[
\pi(\sigma Q,\sigma\mu,\sigma\xi)=\sigma\pi(Q,\mu,\xi).
\]

This requirement affects all blocks:

1. local feature maps must be shared across resources,
2. global summaries must be permutation invariant,
3. resource scores must be produced equivariantly,
4. the certificate operator must commute with resource permutation,
5. diagnostics must reorder consistently with the state.

## 5. First-Class Certificate Boundary

z3 forbids treating certification as a training regularizer.

The final action distribution must satisfy the certificate at inference time by
construction. A drift penalty may help training, but it cannot substitute for
projection, fallback, or another exact certificate operator.

## 6. What Makes The Design Elegant

An elegant CertiQ Dispatcher architecture should have:

1. one canonical forward equation,
2. one resource-set interface,
3. one diagnostics object,
4. pluggable proposal encoders,
5. pluggable certificate operators,
6. theorem claims stated only at the certificate boundary.

The implementation may expose multiple model presets, but those presets must
be parameterizations of the same architecture.

